"""Replayable weighted task sampling and conservative RGB augmentation.

Installed only by the isolated continuation entrypoint; historical runs are untouched.
"""
import bisect
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

LOW_TASKS = ['move_stapler_pad', 'put_object_cabinet', 'blocks_ranking_size',
             'hanging_mug', 'click_bell', 'place_mouse_pad', 'stamp_seal',
             'click_alarmclock', 'open_microwave', 'place_a2b_right']
GAP_TASKS = ['lift_pot', 'handover_mic', 'handover_block', 'move_pillbottle_pad',
             'move_can_pot', 'rotate_qrcode', 'beat_block_hammer', 'open_laptop',
             'place_object_scale', 'place_phone_stand']
CROSS_TASKS = ['move_stapler_pad', 'put_object_cabinet', 'click_bell',
               'lift_pot', 'handover_mic', 'handover_block']


def learning_rate(step):
    if step <= 15200:
        return 1.25e-5 + (7.5e-5 - 1.25e-5) * max(0., (step - 15000) / 200)
    q = min(1., (step - 15200) / 9800)
    return 2.5e-5 + 2.5e-5 * (1 + math.cos(math.pi * q))


def mix64(value):
    mask = (1 << 64) - 1
    value = (value + 0x9e3779b97f4a7c15) & mask
    value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & mask
    value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & mask
    return value ^ (value >> 31)


class DrawDataset(Dataset):
    """Per-draw CPU augmentation RNG is independent of worker prefetch/restarts."""
    def __init__(self, dataset):
        self.base = dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, key):
        frame, seed = key
        np_state, py_state = np.random.get_state(), random.getstate()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                np.random.seed(seed % 2**32)
                random.seed(seed)
                return self.base[frame]
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)


class TaskSampler(Sampler):
    """Rank-sharded counter-based global draws; includes a fingerprint in resume state."""
    def __init__(self, dataset, num_replicas, rank, shuffle=True, seed=42,
                 drop_last=False, mapping=None):
        import os
        self.seed, self.rank, self.world = seed, rank, num_replicas
        self.epoch, self.yielded, self.pending = 0, 0, None
        self.num_samples = math.ceil(len(dataset) / num_replicas)
        if drop_last:
            self.num_samples = len(dataset) // num_replicas
        path = Path(mapping or os.environ['LINGBOT_TASK_MAPPING'])
        self.fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        self.mapping = json.loads(path.read_text())
        self.tasks = sorted(self.mapping['tasks'])
        self.tickets, self.ends, self.intervals = [], {}, {}
        for task in self.tasks:
            self.tickets.extend([task] * (3 if task in LOW_TASKS else 2))
            intervals = self.mapping['tasks'][task]['intervals']
            ends, total = [], 0
            for start, end in intervals:
                total += end - start
                ends.append(total)
            self.ends[task], self.intervals[task] = ends, intervals
        if len(self.tasks) != 50 or len(self.tickets) != 110:
            raise ValueError('Expected 50 tasks, 10 weighted tasks, 110 tickets')

    def draw(self, local_index, epoch=None):
        epoch = self.epoch if epoch is None else epoch
        draw_id = epoch * self.num_samples * self.world + local_index * self.world + self.rank
        key = mix64(draw_id ^ mix64(self.seed))
        task = self.tickets[key % len(self.tickets)]
        offset = mix64(key) % self.ends[task][-1]
        ep = bisect.bisect_right(self.ends[task], offset)
        prior = self.ends[task][ep - 1] if ep else 0
        frame = self.intervals[task][ep][0] + offset - prior
        return task, (frame, mix64(key ^ 0xa5a5) % (2**63 - 1))

    def __iter__(self):
        self.yielded = self.pending if self.pending is not None else 0
        self.pending = None
        while self.yielded < self.num_samples:
            _, key = self.draw(self.yielded)
            self.yielded += 1
            yield key

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

    def state_dict(self):
        return dict(yielded=self.yielded, epoch=self.epoch, seed=self.seed,
                    rank=self.rank, world=self.world, mapping_sha256=self.fingerprint,
                    policy='task_weight_1.5_frame_uniform_v1')

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in ['seed', 'rank', 'world', 'mapping_sha256', 'policy']:
            if state[key] != expected[key]:
                raise ValueError(f'Sampler resume mismatch: {key}')
        if not 0 <= state['yielded'] <= self.num_samples:
            raise ValueError('Invalid sampler cursor')
        self.pending, self.epoch = state['yielded'], state['epoch']


def sample_augmentation(reference):
    p = torch.rand(()).item()
    branch = 'original' if p < .3 else 'photometric' if p < .7 else 'degraded'
    params = dict(branch=branch, brightness=.7 + .6 * torch.rand(()).item(),
                  saturation=.9 + .2 * torch.rand(()).item())
    if branch == 'degraded':
        params.update(degradation='noise' if torch.rand(()).item() < .5 else 'blur',
                      sigma_noise=(1 + 2 * torch.rand(()).item()) / 255,
                      sigma_blur=.1 + .4 * torch.rand(()).item(),
                      noise_seed=torch.randint(0, 2**31, ()).item())
    return params


def apply_augmentation(visual, params):
    if params['branch'] == 'original':
        return visual
    if visual.ndim not in (3, 4):
        raise ValueError('Expected C,H,W or T,C,H,W')
    dtype = visual.dtype
    scale = 255. if dtype == torch.uint8 or visual.max() > 1 else 1.
    x = visual.float() / scale
    x = x * params['brightness']
    # Luma weighting keeps the intended RGB hue while changing saturation slightly.
    weights = x.new_tensor([.2989, .5870, .1140]).view(*([1] * (x.ndim - 3)), 3, 1, 1)
    gray = (x * weights).sum(dim=-3, keepdim=True)
    x = gray + (x - gray) * params['saturation']
    if params['branch'] == 'degraded':
        if params['degradation'] == 'noise':
            generator = torch.Generator(device=x.device).manual_seed(params['noise_seed'])
            # Replay exactly for matching current/future pixels and repeated calls.
            x = x + torch.randn(x.shape, generator=generator, device=x.device) * params['sigma_noise']
        else:
            one = x.ndim == 3
            if one: x = x.unsqueeze(0)
            a = torch.arange(-1, 2, device=x.device).float()
            k = torch.exp(-a.square() / (2 * params['sigma_blur']**2))
            k = k / k.sum()
            kernel = (k[:, None] * k[None, :]).expand(x.shape[1], 1, 3, 3)
            x = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), kernel, groups=x.shape[1])
            if one: x = x.squeeze(0)
    x = x.clamp(0, 1) * scale
    return x.round().to(dtype) if dtype == torch.uint8 else x.to(dtype)


def build_stage_dataloader(**kwargs):
    from lingbotvla.data import data_loader
    kwargs['dataset'] = DrawDataset(kwargs['dataset'])
    original = data_loader.StatefulDistributedSampler
    data_loader.StatefulDistributedSampler = TaskSampler
    try:
        return data_loader.build_dataloader(**kwargs)
    finally:
        data_loader.StatefulDistributedSampler = original


def install_augmentation():
    from lingbotvla.data.vla_data import transform
    transform.sample_visual_augmentation_params = sample_augmentation
    transform.apply_visual_augmentation = apply_augmentation


def execution_length(action, current_state=None, grippers=(6, 13)):
    """Use 25 only after stable early transitions in both predicted gripper traces."""
    a = np.asarray(action)
    if a.ndim != 2 or len(a) != 50 or a.shape[1] < 14 or not np.isfinite(a).all():
        raise ValueError('Expected finite, unnormalized action[50,14+]')
    changed = False
    for col in grippers:
        x = a[:, col]
        initial=x[0] if current_state is None else np.asarray(current_state)[col]
        state = 0 if initial <= .3 else 1 if initial >= .7 else None
        if state is None: return 50
        last_complete = 0
        for i in range(1 if current_state is None else 0, 25):
            target = 1-state
            crossed = x[i] >= .7 if target else x[i] <= .3
            if crossed:
                if i+2 >= 25:
                    return 50
                stable = np.all(x[i:i+3] >= .7) if target else np.all(x[i:i+3] <= .3)
                if not stable: return 50
                state = target
                last_complete = i + 3  # one-based index of final confirmation
                changed = True
            # Midrange after a completed switch is not a stable holding period.
            if last_complete and .3 < x[i] < .7: return 50
        if last_complete > 15: return 50
    return 25 if changed else 50
