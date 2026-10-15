"""CPU-only validation of saved checkpoint, actual argument parser and continuation scheduler."""
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
import pickle
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from types import SimpleNamespace
from unittest import mock

import torch
import yaml
from safetensors import safe_open
from lingbotvla.utils.arguments import (EvalArguments, DataArguments, ModelArguments,
                                      TrainingArguments, parse_args)

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / 'plan.json').read_text())
sys.path.insert(0, str(HERE / 'source'))
import resume_support as support
import run_pipeline as pipeline

checks = []
for file in list((HERE / 'source').glob('*.py')) + list(HERE.glob('*.py')) + list((HERE / 'evaluations').rglob('*.py')):
    ast.parse(file.read_text(), filename=str(file))
checks.append('all training, controller and evaluation Python files parse')
pipeline.check_original_checkpoint()
checkpoint = Path(PLAN['full_checkpoint'])
rank_states = []
for rank in range(4):
    state = torch.load(checkpoint / 'extra_state' / f'extra_state_rank_{rank}.pt', map_location='cpu', weights_only=False)
    assert state['global_step'] == 10000 and state['lr_scheduler']['last_epoch'] == 10000
    assert all(math.isclose(v, 2.5e-5, abs_tol=1e-12) for v in state['lr_scheduler']['_last_lr'])
    assert all(k in state for k in ['train_dataloader', 'environ_meter', 'torch_rng_state'])
    rank_states.append(dict(rank=rank, step=state['global_step'], lr=state['lr_scheduler']['_last_lr']))
for component in ['model', 'optimizer']:
    with (checkpoint / component / '.metadata').open('rb') as stream:
        metadata = pickle.load(stream)
    for storage in metadata.storage_data.values():
        assert (checkpoint / component / storage.relative_path).stat().st_size >= storage.offset + storage.length
mapping = json.loads((checkpoint / 'hf_ckpt/model.safetensors.index.json').read_text())['weight_map']
for shard in sorted(set(mapping.values())):
    with safe_open(checkpoint / 'hf_ckpt' / shard, framework='pt', device='cpu') as weights:
        assert set(weights.keys()) == {key for key, value in mapping.items() if value == shard}
checks.append('all four saved rank states at step10000/LR2.5e-5; DCP storage extents and HF shard headers valid')

assets = HERE / 'inference_assets'
manifest = json.loads((assets / 'normalization_manifest.json').read_text())
for name, entry in manifest['assets'].items():
    assert hashlib.sha256((assets / name).read_bytes()).hexdigest() == entry['sha256']
assert (Path(PLAN['repo']) / 'configs/robot_configs/robotwin.yaml').read_bytes() == (assets / 'robotwin.yaml').read_bytes()
checks.append('archived official clean normalization and robot mapping hashes unchanged')

# Extract only dataclass definitions; run the real repository parser without importing model initialization.
tree = ast.parse((HERE / 'source/train_continuation.py').read_text())
classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and
           node.name in ['MyTrainingArguments', 'MyDataArguments', 'Arguments']]
namespace = dict(dataclass=dataclass, field=field, Optional=Optional, Literal=Literal, Any=Any,
                 List=List, Dict=Dict, TrainingArguments=TrainingArguments,
                 DataArguments=DataArguments, ModelArguments=ModelArguments, EvalArguments=EvalArguments)
exec(compile(ast.Module(body=classes, type_ignores=[]), '<training dataclasses>', 'exec'), namespace)
with mock.patch.object(sys, 'argv', ['train_continuation.py', str(HERE / 'training/config.yaml')]):
    parsed = parse_args(namespace['Arguments'])
actual = asdict(parsed)
previous = yaml.safe_load((Path(PLAN['original_run']) / 'lingbotvla_cli.yaml').read_text())
allowed = {'data.norm_stats_file', 'train.lr', 'train.lr_min', 'train.max_steps',
           'train.load_checkpoint_path', 'train.output_dir'}
changes = {}
for section, entries in previous.items():
    for key, value in entries.items():
        observed = actual[section].get(key)
        if observed != value:
            name = section + '.' + key
            changes[name] = dict(before=value, after=observed)
            assert name in allowed, (name, value, observed)
assert parsed.train.micro_batch_size == 20 and parsed.train.gradient_accumulation_steps == 2
assert parsed.train.global_batch_size == 160 and parsed.train.enable_gradient_checkpointing
assert parsed.train.enable_resume and parsed.train.max_steps == 15000
assert parsed.train.optimizer == 'dist_muon' and parsed.train.weight_decay == 0
assert parsed.model.vlm_repo_id is None and parsed.model.basic_modules == []
(HERE / 'effective_arguments_preflight.json').write_text(json.dumps(actual, indent=2) + '\n')
checks.append('real argument parser matches previous effective arguments except the six intended continuation fields')

rates = {step: support.continuation_lr(step) for step in [10000, 10001, 12000, 12500, 15000]}
assert math.isclose(rates[10000], 2.5e-5) and math.isclose(rates[12500], 1.875e-5)
assert math.isclose(rates[15000], 1.25e-5)
parameter = torch.nn.Parameter(torch.ones(1))
optimizer = torch.optim.SGD([parameter], lr=5e-5, momentum=.95)
parameter.grad = torch.ones_like(parameter)
optimizer.step()
old_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1)
old_state = old_scheduler.state_dict()
old_state.update(last_epoch=10000, _last_lr=[2.5e-5],
    continuation_policy=dict(start_step=5000,end_step=10000,start_lr=5e-5,end_lr=2.5e-5))
scheduler = support.build_lr_scheduler(optimizer)
scheduler.load_state_dict(old_state)
optimizer.param_groups[0]['lr'] = 2.5e-5
momentum = optimizer.state[parameter]['momentum_buffer'].clone()
with tempfile.TemporaryDirectory() as temporary:
    args = SimpleNamespace(train=SimpleNamespace(output_dir=temporary,load_checkpoint_path=PLAN['full_checkpoint']))
    with mock.patch.object(support.dist,'get_rank',return_value=0), mock.patch.object(support,'archive_inference_assets'):
        support.configure_resume_scheduler(scheduler,optimizer,10000,args)
        assert scheduler.continuation_policy['start_step'] == 10000
        assert scheduler.continuation_policy['end_step'] == 15000
        assert torch.equal(momentum,optimizer.state[parameter]['momentum_buffer'])
        assert math.isclose(optimizer.param_groups[0]['lr'],2.5e-5)
        for step in range(10001,15001):
            optimizer.step();scheduler.step()
            assert math.isclose(optimizer.param_groups[0]['lr'],support.continuation_lr(step),abs_tol=1e-12)
        assert scheduler.last_epoch == 15000
checks.append('loaded old scheduler metadata is replaced by the new stage policy; optimizer momentum retained; all 5000 LR updates verified')

for directory in map(Path,PLAN['eval_roots']):
    plan = json.loads((directory/'plan.json').read_text())
    assert plan['eval_video_log'] is False and plan['episodes_per_task'] == 10
    assert len(set(plan['tasks'])) == 50 and len(plan['slots']) == 8
    assert [(s['task_config'],s['instruction_type']) for s in plan['settings']] == [('demo_clean','seen'),('demo_randomized','unseen')]
    assert math.isclose(plan['expected_checkpoint_lr'],support.continuation_lr(plan['checkpoint_step']))
    spec = importlib.util.spec_from_file_location('evaluation',directory/'run_scheduled.py')
    evaluation = importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluation)
    for path, digest in json.loads((directory/'source_hashes.json').read_text()).items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    with tempfile.TemporaryDirectory() as temporary:
        fake=Path(temporary)
        (fake/'training_status.json').write_text(json.dumps({'phase':'training','last_step':10100}))
        (fake/'launcher.pid').write_text('123')
        with mock.patch.object(evaluation,'PLAN',{**plan,'train_root':str(fake)}),mock.patch.object(evaluation,'alive',return_value=True):
            ready,detail=evaluation.readiness()
        assert not ready and detail['reason']=='training_not_complete'
        (fake/'training_status.json').write_text(json.dumps({'phase':'failed','error':'test'}))
        with mock.patch.object(evaluation,'PLAN',{**plan,'train_root':str(fake)}):
            try:
                evaluation.readiness()
            except RuntimeError:
                pass
            else:
                raise AssertionError('Evaluation must stop on training failure')
checks.append('step12000/15000 no-video evaluation plans, hashes and incomplete/failed-training gates verified')
assert not torch.cuda.is_initialized(), 'Preflight must not allocate any GPU resources'
pipeline.check_original_checkpoint()
report = dict(passed=True,checks=checks,checkpoint_rank_states=rank_states,
              learning_rates=rates,intended_argument_changes=changes,gpu_resources_used=False)
(HERE/'preflight_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
