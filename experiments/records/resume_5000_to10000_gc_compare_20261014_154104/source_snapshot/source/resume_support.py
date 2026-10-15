"""Isolated continuation policy and measurement; original repository stays unchanged."""
import importlib.util
import hashlib
import json
import math
import os
import sys
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR

START_STEP = 5000
END_STEP = 10000
START_LR = 5e-5
END_LR = 2.5e-5
_metric_stream = None


def record_gradient_checkpointing_state(model, args):
    enabled = [name for name, module in model.named_modules()
               if getattr(module, 'gradient_checkpointing', False) is True]
    requested = args.train.enable_gradient_checkpointing
    if bool(enabled) != requested:
        raise RuntimeError(f'Gradient checkpointing flag mismatch: requested={requested}, enabled={enabled}')
    output = Path(args.train.output_dir) / f'gradient_checkpointing_rank{dist.get_rank()}.json'
    output.write_text(json.dumps({'requested': requested, 'enabled_module_count': len(enabled),
                                 'enabled_modules': enabled}, indent=2) + '\n')


def record_failure(exc):
    directory = os.environ.get('LINGBOT_STAGE_OUTPUT')
    if not directory:
        return
    rank = int(os.environ.get('RANK', '0'))
    data = {'rank': rank, 'type': type(exc).__name__, 'error': str(exc)}
    if torch.cuda.is_initialized():
        data['peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
        data['peak_reserved_mib'] = torch.cuda.max_memory_reserved() / 2**20
    (Path(directory) / f'failure_rank{rank}.json').write_text(json.dumps(data, indent=2) + '\n')


def continuation_lr(step):
    progress = min(1.0, max(0.0, (step - START_STEP) / (END_STEP - START_STEP)))
    return END_LR + (START_LR - END_LR) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_lr_scheduler(optimizer, **kwargs):
    # The schedule horizon is 10000 even when the short test stops at 5050.
    scheduler = LambdaLR(optimizer, lambda step: continuation_lr(step) / START_LR)
    scheduler.continuation_policy = {
        'start_step': START_STEP, 'end_step': END_STEP,
        'start_lr': START_LR, 'end_lr': END_LR,
    }
    return scheduler


def build_strict_checkpointer(**kwargs):
    name = 'lingbotvla.checkpoint.continuation_strict_checkpointer'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name('strict_checkpointer.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].build_checkpointer(**kwargs)


def configure_resume_scheduler(scheduler, optimizer, global_step, args):
    if global_step != START_STEP:
        raise RuntimeError(f'Expected full step-5000 restore, got {global_step}')
    if scheduler.last_epoch != global_step:
        raise RuntimeError('Restored scheduler and global step disagree')
    previous_lrs = [float(group['lr']) for group in optimizer.param_groups]
    if any(not math.isclose(lr, START_LR, abs_tol=1e-12) for lr in previous_lrs):
        raise RuntimeError(f'Checkpoint LR is not 5e-5: {previous_lrs}')
    inner = getattr(optimizer, 'optimizers', [optimizer])
    counts = [len(opt.state) for opt in inner]
    if any(count == 0 for count in counts):
        raise RuntimeError(f'Optimizer state was not restored: {counts}')
    # Preserve all loaded optimizer state; change only its continuation LR policy.
    scheduler.base_lrs = [START_LR] * len(optimizer.param_groups)
    scheduler.last_epoch = global_step
    scheduler._last_lr = [continuation_lr(global_step)] * len(optimizer.param_groups)
    for group, lr in zip(optimizer.param_groups, scheduler._last_lr):
        group['initial_lr'] = START_LR
        group['lr'] = lr
    audit = {
        'rank': dist.get_rank(), 'restored_global_step': global_step,
        'optimizer_state_counts': counts, 'checkpoint_lrs': previous_lrs,
        'continuation_lrs': scheduler.get_last_lr(), 'schedule': scheduler.continuation_policy,
        'checkpoint': args.train.load_checkpoint_path,
        'model_load_strict': True, 'optimizer_load_strict': True,
    }
    path = Path(args.train.output_dir) / f'resume_audit_rank{dist.get_rank()}.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(audit, stream, indent=2)
    archive_inference_assets(Path(args.train.output_dir))
    print('FULL_RESTORE_AUDIT ' + json.dumps(audit), flush=True)


def archive_inference_assets(output):
    if dist.get_rank() != 0:
        return
    run_root = Path(__file__).resolve().parents[1]
    target = output / 'inference_assets'
    shutil.copytree(run_root / 'inference_assets', target)
    manifest_path = target / 'normalization_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['run_directory'] = str(output)
    config = output / 'lingbotvla_cli.yaml'
    manifest['training_config'] = {'path': str(config), 'sha256': hashlib.sha256(config.read_bytes()).hexdigest()}
    manifest['inference']['norm_stats_snapshot'] = str(target / 'norm_stats.json')
    manifest['continuation'] = json.loads((run_root / 'plan.json').read_text())
    flag_file = output / 'gradient_checkpointing_rank0.json'
    if flag_file.exists():
        manifest['continuation']['actual_gradient_checkpointing'] = json.loads(flag_file.read_text())['requested']
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


def record_resume_state(state, args, global_step):
    audit_path = Path(args.train.output_dir) / f'resume_audit_rank{dist.get_rank()}.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    audit.update(
        dataloader_state_restored=bool(args.train.resume_dataloader_state),
        cpu_rng_state_restored=torch.equal(torch.get_rng_state(), state['extra_state']['torch_rng_state']),
        environ_meter_state_restored=True,
    )
    audit_path.write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    if not audit['cpu_rng_state_restored']:
        raise RuntimeError('CPU RNG state restore failed')


def begin_measured_step(micro_batches, global_step):
    # Global batch remains fixed, including after loading the old dataloader cursor.
    sizes = [int(batch['lang_masks'].shape[0]) for batch in micro_batches]
    if len(sizes) != 2 or sizes != [20, 20]:
        raise RuntimeError(f'Expected two micro batches of 20, got {sizes} at {global_step}')
    torch.cuda.reset_peak_memory_stats()


def record_measured_step(global_step, seconds, loss, vla_loss, grad_norm, lr, micro_batches, args):
    global _metric_stream
    numeric = {'seconds': float(seconds), 'loss': float(loss), 'vla_loss': float(vla_loss),
               'grad_norm': float(grad_norm), 'lr_after_step': float(lr)}
    if not all(math.isfinite(value) for value in numeric.values()):
        raise RuntimeError(f'Non-finite training metrics at {global_step}: {numeric}')
    if not math.isclose(numeric['lr_after_step'], continuation_lr(global_step), abs_tol=1e-12):
        raise RuntimeError(f'LR discontinuity at {global_step}: {numeric["lr_after_step"]}')
    if _metric_stream is None:
        directory = Path(args.train.output_dir) / 'metrics'
        directory.mkdir(exist_ok=True)
        _metric_stream = (directory / f'rank{dist.get_rank()}.jsonl').open('x', encoding='utf-8', buffering=1)
    row = {'step': global_step, 'rank': dist.get_rank(), **numeric,
           'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
           'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
           'allocated_mib': torch.cuda.memory_allocated() / 2**20,
           'global_batch_size': args.train.global_batch_size,
           'micro_batch_size': args.train.micro_batch_size,
           'accumulation_steps': len(micro_batches)}
    _metric_stream.write(json.dumps(row) + '\n')
