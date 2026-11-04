"""Additional restoration checks for a new sampling stage and exact later restarts."""
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
from stage_data import learning_rate

POLICY = '15000_lr75_warm200_aug304030_task15_v1'


def restore_stage_loader(loader, extra, args, step):
    output = Path(args.train.output_dir)
    if step == 15000:
        if 'stage_policy' in extra:
            raise RuntimeError('Unexpected sampling stage already present at step15000')
        torch.save(extra['train_dataloader'], output/f'previous_loader_rank{dist.get_rank()}.pt')
        transition = dict(previous_global_step=step, old_cursor_archived=True,
                          new_sampling_stage=True, policy=POLICY,
                          mapping_sha256=hashlib.sha256(Path(os.environ['LINGBOT_TASK_MAPPING']).read_bytes()).hexdigest())
        (output/f'sampling_transition_rank{dist.get_rank()}.json').write_text(json.dumps(transition,indent=2)+'\n')
        return 0, 0
    if extra.get('stage_policy') != POLICY:
        raise RuntimeError('Checkpoint belongs to another sampling/augmentation stage')
    loader.load_state_dict(extra['train_dataloader'])
    elapsed = step - 15000
    return elapsed // args.train.train_steps, elapsed % args.train.train_steps


def stage_rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cuda=torch.cuda.get_rng_state())


def record_stage_restore(extra, args, step):
    rng = extra.get('stage_rng')
    if step > 15000:
        if rng is None: raise RuntimeError('Missing new-stage RNG state')
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.cuda.set_rng_state(rng['cuda'])
    output=Path(args.train.output_dir)
    p=output/f'resume_audit_rank{dist.get_rank()}.json'
    audit=json.loads(p.read_text())
    audit.update(cpu_rng_state_restored=torch.equal(torch.get_rng_state(),extra['torch_rng_state']),
                 environ_meter_state_restored=True,
                 dataloader_state_restored=step>15000, new_sampling_stage=step==15000,
                 cuda_rng_restored=rng is not None,
                 original_cuda_rng_available=step>15000, stage_policy=POLICY)
    if not audit['cpu_rng_state_restored']:raise RuntimeError('CPU RNG restoration failed')
    p.write_text(json.dumps(audit,indent=2)+'\n')


def ensure_finite_before_step(loss, grad_norm):
    g = grad_norm.full_tensor() if hasattr(grad_norm,'full_tensor') else grad_norm
    ok = math.isfinite(float(loss)) and math.isfinite(float(g))
    flag=torch.tensor(int(ok),device=torch.cuda.current_device())
    dist.all_reduce(flag,op=dist.ReduceOp.MIN)
    if not flag.item():raise FloatingPointError('Non-finite loss/gradient before optimizer.step')


def configure_scheduler(scheduler,optimizer,step,args):
    if not 15000<=step<=25000 or scheduler.last_epoch!=step:
        raise RuntimeError(f'Unexpected checkpoint/scheduler step: {step}/{scheduler.last_epoch}')
    previous=[float(g['lr']) for g in optimizer.param_groups]
    if any(not math.isclose(x,learning_rate(step),abs_tol=1e-12) for x in previous):
        raise RuntimeError(f'Wrong checkpoint LR at {step}: {previous}')
    inner=getattr(optimizer,'optimizers',[optimizer]);counts=[len(o.state) for o in inner]
    if counts != [935,737]:raise RuntimeError(f'Optimizer state counts mismatch: {counts}')
    scheduler.base_lrs=[1.25e-5]*len(optimizer.param_groups)
    scheduler.last_epoch=step
    scheduler._last_lr=[learning_rate(step)]*len(optimizer.param_groups)
    scheduler.continuation_policy=dict(start_step=15000,warmup_end_step=15200,end_step=25000,
                                      start_lr=1.25e-5,peak_lr=7.5e-5,end_lr=2.5e-5)
    for group in optimizer.param_groups:
        group['initial_lr']=1.25e-5;group['lr']=learning_rate(step)
    audit=dict(rank=dist.get_rank(),restored_global_step=step,optimizer_state_counts=counts,
               checkpoint_lrs=previous,continuation_lrs=scheduler.get_last_lr(),
               schedule=scheduler.continuation_policy,checkpoint=args.train.load_checkpoint_path,
               model_load_strict=True,optimizer_load_strict=True)
    output=Path(args.train.output_dir)
    (output/f'resume_audit_rank{dist.get_rank()}.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('FULL_RESTORE_AUDIT '+json.dumps(audit),flush=True)
