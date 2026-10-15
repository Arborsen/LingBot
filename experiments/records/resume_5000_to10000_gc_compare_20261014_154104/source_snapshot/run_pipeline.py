"""Benchmark full restoration, then train only if the benchmark passes."""
import json
import math
import os
import signal
import statistics
import subprocess
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / 'plan.json').read_text())
ACTIVE = None
STOP = threading.Event()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def gpu_status():
    rows = subprocess.check_output(['nvidia-smi', '--id=0,1,2,3',
        '--query-gpu=index,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    return [{'gpu': int(parts[0]), 'used_mib': int(parts[1]), 'total_mib': int(parts[2]),
             'utilization_percent': int(parts[3])}
            for parts in (row.split(',') for row in rows.splitlines())]


def check_original_checkpoint():
    original = Path(PLAN['original_run'])
    manifest = json.loads((HERE / 'original_checkpoint_manifest.json').read_text())
    for name, metadata in manifest.items():
        path = original / name
        stat = path.stat()
        if stat.st_size != metadata['size'] or stat.st_mtime_ns != metadata['mtime_ns']:
            raise RuntimeError(f'Original checkpoint changed: {path}')


def environment():
    env = os.environ.copy()
    prefix = Path(PLAN['env'])
    for key in list(env):
        if key.lower().endswith('_proxy'):
            env.pop(key)
    env.update(CUDA_VISIBLE_DEVICES='0,1,2,3', CONDA_PREFIX=str(prefix),
        PATH=str(prefix / 'bin') + ':' + env.get('PATH', ''),
        PYTHONPATH=PLAN['repo'] + ':' + env.get('PYTHONPATH', ''),
        PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1', DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        TORCH_EXTENSIONS_DIR=str(HERE / 'extensions_cache'),
        MPLCONFIGDIR=str(HERE / 'matplotlib_cache'),
        TORCHINDUCTOR_CACHE_DIR=str(Path(PLAN['baseline_run']) / 'compile_cache'),
        TRITON_CACHE_DIR=str(Path(PLAN['baseline_run']) / 'triton_cache'))
    return env


def latest_metrics(stage):
    values = []
    for rank in range(4):
        path = HERE / stage / 'metrics' / f'rank{rank}.jsonl'
        if path.exists():
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size - 8192))
                lines = stream.read().decode().splitlines()
            for line in reversed(lines):
                try:
                    values.append(json.loads(line))
                    break
                except json.JSONDecodeError:
                    continue
    return values


def monitor(stage, process, finished):
    with (HERE / stage / 'gpu_samples.jsonl').open('x', encoding='utf-8', buffering=1) as stream:
        while not finished.is_set():
            readings = gpu_status()
            stream.write(json.dumps({'time': time.time(), 'gpus': readings}) + '\n')
            metrics = latest_metrics(stage)
            save_json(HERE / 'status.json', {'phase': stage, 'child_pid': process.pid,
                'target_step': 5050 if stage == 'benchmark' else 10000,
                'last_step': min((m['step'] for m in metrics), default=5000),
                'latest_rank_metrics': metrics, 'gpus': readings, 'updated_at': time.time()})
            finished.wait(5)


def run_stage(stage):
    global ACTIVE
    check_original_checkpoint()
    deadline = time.monotonic() + 60
    while any(g['used_mib'] > 1024 for g in gpu_status()):
        if time.monotonic() > deadline:
            raise RuntimeError('GPU 0-3 are not free; no competing processes will be stopped')
        time.sleep(3)
    command = [str(Path(PLAN['env']) / 'bin/python'), '-m', 'torch.distributed.run',
        '--standalone', '--nproc_per_node=4', str(HERE / 'source/train_continuation.py'),
        str(HERE / stage / 'config.yaml')]
    with (HERE / stage / 'train.log').open('x', encoding='utf-8') as log:
        ACTIVE = subprocess.Popen(command, cwd=PLAN['repo'], env={**environment(), 'LINGBOT_STAGE_OUTPUT': str(HERE / stage)},
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save_json(HERE / stage / 'process.json', {'pid': ACTIVE.pid, 'command': command,
        'checkpoint': PLAN['full_checkpoint'], 'start_step': 5000})
    print(f'START {stage}: pid={ACTIVE.pid}; full restore from step 5000', flush=True)
    finished = threading.Event()
    watcher = threading.Thread(target=monitor, args=(stage, ACTIVE, finished), daemon=True)
    watcher.start()
    try:
        code = ACTIVE.wait()
    finally:
        finished.set()
        watcher.join(timeout=15)
    (HERE / stage / 'exit_code.txt').write_text(str(code) + '\n')
    if code:
        raise RuntimeError(f'{stage} exited with {code}; see {HERE / stage / "train.log"}')
    check_original_checkpoint()


def check_checkpoint(stage, step):
    directory = HERE / stage / 'checkpoints' / f'global_step_{step}'
    for name in ['model/.metadata', 'optimizer/.metadata']:
        path = directory / name
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f'Incomplete distributed checkpoint: {path}')
    for rank in range(4):
        path = directory / 'extra_state' / f'extra_state_rank_{rank}.pt'
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f'Missing rank checkpoint state: {path}')
    index = directory / 'hf_ckpt/model.safetensors.index.json'
    mapping = json.loads(index.read_text())['weight_map']
    shards = sorted(set(mapping.values()))
    if not shards or any(not (index.parent / name).is_file() or not (index.parent / name).stat().st_size for name in shards):
        raise RuntimeError('HF checkpoint export is incomplete')
    return {'directory': str(directory), 'hf_shards': shards}


def validate_benchmark():
    measurements = []
    audits = []
    for rank in range(4):
        audit = json.loads((HERE / 'benchmark' / f'resume_audit_rank{rank}.json').read_text())
        if audit['restored_global_step'] != 5000 or not all(audit.get(key) for key in
                ['model_load_strict', 'optimizer_load_strict', 'dataloader_state_restored',
                 'cpu_rng_state_restored', 'environ_meter_state_restored']):
            raise RuntimeError(f'Incomplete restoration on rank {rank}: {audit}')
        audits.append(audit)
        rows = [json.loads(line) for line in (HERE / 'benchmark/metrics' / f'rank{rank}.jsonl').read_text().splitlines()]
        if [row['step'] for row in rows] != list(range(5001, 5051)):
            raise RuntimeError(f'Expected 50 consecutive updates on rank {rank}')
        if any(row['micro_batch_size'] != 20 or row['accumulation_steps'] != 2 or row['global_batch_size'] != 160 for row in rows):
            raise RuntimeError('The actual batch sizes changed')
        if any(not math.isfinite(row[k]) for row in rows for k in ['seconds', 'loss', 'vla_loss', 'grad_norm', 'lr_after_step']):
            raise RuntimeError('Non-finite training metrics')
        measurements.append(rows)
    warm = PLAN['warmup_steps_excluded']
    times = [max(rows[index]['seconds'] for rows in measurements) for index in range(warm, 50)]
    gpu_samples = [json.loads(line) for line in (HERE / 'benchmark/gpu_samples.jsonl').read_text().splitlines()]
    peak_gpus = {gpu: max(reading['used_mib'] for sample in gpu_samples for reading in sample['gpus'] if reading['gpu'] == gpu) for gpu in range(4)}
    totals = {row['gpu']: row['total_mib'] for row in gpu_samples[0]['gpus']}
    report = {'passed': False, 'updates': 50, 'measured_updates': 50 - warm,
        'mean_step_seconds': statistics.mean(times), 'median_step_seconds': statistics.median(times),
        'samples_per_second': 160 / statistics.mean(times),
        'baseline_step_seconds': PLAN['baseline_step_seconds'],
        'speedup': PLAN['baseline_step_seconds'] / statistics.mean(times),
        'mean_vla_loss': statistics.mean(row['vla_loss'] for row in measurements[0][warm:]),
        'max_grad_norm': max(row['grad_norm'] for rows in measurements for row in rows),
        'peak_nvidia_memory_mib': peak_gpus,
        'peak_torch_allocated_mib': {rank: max(row['peak_allocated_mib'] for row in rows) for rank, rows in enumerate(measurements)},
        'peak_torch_reserved_mib': {rank: max(row['peak_reserved_mib'] for row in rows) for rank, rows in enumerate(measurements)},
        'full_restoration_verified': True, 'checkpoint': check_checkpoint('benchmark', 5050),
        'errors': []}
    if report['mean_step_seconds'] > PLAN['baseline_step_seconds'] * PLAN['max_step_seconds_ratio']:
        report['errors'].append('GC off gives less than 3% speed improvement versus GC on')
    if any(max(peak_gpus[gpu], report['peak_torch_reserved_mib'][gpu]) >
           totals[gpu] * PLAN['max_gpu_memory_fraction'] for gpu in range(4)):
        report['errors'].append('Peak memory leaves less than 10% headroom on at least one GPU')
    if report['mean_vla_loss'] > 0.1:
        report['errors'].append('Mean action loss is unexpectedly high for this restored checkpoint')
    report['passed'] = not report['errors']
    save_json(HERE / 'benchmark/report.json', report)
    print('BENCHMARK_REPORT ' + json.dumps(report), flush=True)
    return report


def main():
    with (HERE / 'launcher.pid').open('x') as stream:
        stream.write(str(os.getpid()) + '\n')
    report = None
    try:
        run_stage('benchmark')
        report = validate_benchmark()
    except RuntimeError as exc:
        log = (HERE / 'benchmark/train.log').read_text(encoding='utf-8', errors='replace')
        if 'out of memory' not in log.lower():
            raise
        gpu_file = HERE / 'benchmark/gpu_samples.jsonl'
        samples = [json.loads(line) for line in gpu_file.read_text().splitlines()] if gpu_file.exists() else []
        failures = [json.loads(p.read_text()) for p in (HERE / 'benchmark').glob('failure_rank*.json')]
        report = {'passed': False, 'reason': 'OOM with GC disabled', 'error': str(exc),
                  'rank_failures': failures, 'gpu_samples': samples[-3:]}
        save_json(HERE / 'benchmark/report.json', report)
    use_gc = not report['passed']
    import yaml
    path = HERE / 'training/config.yaml'
    config = yaml.safe_load(path.read_text())
    config['train']['enable_gradient_checkpointing'] = use_gc
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    decision = {'gradient_checkpointing': use_gc, 'gc_off_passed': report['passed'],
                'reason': 'GC off improves speed with memory headroom' if not use_gc else report.get('errors', report.get('reason')),
                'gc_on_mean_seconds': PLAN['baseline_step_seconds'],
                'gc_off_report': str(HERE / 'benchmark/report.json'),
                'formal_restart_step': 5000, 'formal_stop_step': 10000,
                'micro_batch_size': 20, 'accumulation_steps': 2, 'global_batch_size': 160}
    save_json(HERE / 'decision.json', decision)
    print('GC_DECISION ' + json.dumps(decision), flush=True)
    run_stage('training')
    checkpoint = check_checkpoint('training', 10000)
    save_json(HERE / 'status.json', {'phase': 'complete', 'final_step': 10000,
        'additional_updates': 5000, 'final_lr': 2.5e-5, 'checkpoint': checkpoint,
        'gradient_checkpointing': use_gc, 'original_checkpoint_unchanged': True})
    (HERE / 'exit_code.txt').write_text('0\n')


def interrupted(signum, frame):
    raise KeyboardInterrupt(f'Signal {signum}')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        main()
    except BaseException as exc:
        if ACTIVE is not None and ACTIVE.poll() is None:
            os.killpg(ACTIVE.pid, signal.SIGTERM)
            try:
                ACTIVE.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(ACTIVE.pid, signal.SIGKILL)
                ACTIVE.wait()
        save_json(HERE / 'status.json', {'phase': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        (HERE / 'exit_code.txt').write_text('1\n')
        print(f'FAILED {type(exc).__name__}: {exc}', flush=True)
        raise
