"""Verify saved training step/LR on all ranks without allocating GPU memory."""
import json
import math
from pathlib import Path
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / 'plan.json').read_text())
checkpoint = Path(PLAN['checkpoint']).parent
report = []
weight_map = json.loads((checkpoint / 'hf_ckpt/model.safetensors.index.json').read_text())['weight_map']
for shard in sorted(set(weight_map.values())):
    with safe_open(checkpoint / 'hf_ckpt' / shard, framework='pt', device='cpu') as weights:
        actual = set(weights.keys())
        expected = {key for key, value in weight_map.items() if value == shard}
        assert actual == expected, (shard, len(actual), len(expected))
        for key in actual:
            weights.get_slice(key).get_shape()
for rank in range(4):
    state = torch.load(checkpoint / 'extra_state' / f'extra_state_rank_{rank}.pt',
                       map_location='cpu', weights_only=False)
    assert state['global_step'] == PLAN['checkpoint_step'], (rank, state['global_step'])
    scheduler = state['lr_scheduler']
    assert scheduler['last_epoch'] == PLAN['checkpoint_step'], (rank, scheduler['last_epoch'])
    assert all(math.isclose(lr, PLAN['expected_checkpoint_lr'], rel_tol=1e-6) for lr in scheduler['_last_lr']), scheduler
    report.append(dict(rank=rank, step=state['global_step'], lr=scheduler['_last_lr']))
(HERE / 'checkpoint_rank_validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(f'All four rank states validated: step {PLAN["checkpoint_step"]}, LR {PLAN["expected_checkpoint_lr"]}')
