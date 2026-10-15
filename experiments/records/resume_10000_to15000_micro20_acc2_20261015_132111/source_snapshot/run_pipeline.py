"""Wait for four free GPUs, fully resume step10000, then evaluate saved step12000 and step15000."""
import json
import math
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / 'plan.json').read_text())
ACTIVE = None
LOCK = threading.RLock()
STOP = threading.Event()
PHASE = 'waiting_for_free_gpus'


def save_json(path, value):
    with LOCK:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
        temporary.replace(path)


def gpu_status():
    output = subprocess.check_output(['nvidia-smi', '--id=0,1,2,3',
        '--query-gpu=index,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    rows = [{'gpu': int(p[0]), 'used_mib': int(p[1]), 'total_mib': int(p[2]),
             'utilization_percent': int(p[3])} for p in (line.split(',') for line in output.splitlines())]
    if len(rows) != 4 or {g['gpu'] for g in rows} != {0, 1, 2, 3}:
        raise RuntimeError('Expected all four GPUs 0-3')
    return rows


def check_original_checkpoint():
    directory = Path(PLAN['original_run'])
    for name, expected in json.loads((HERE / 'original_checkpoint_manifest.json').read_text()).items():
        stat = (directory / name).stat()
        if stat.st_size != expected['size'] or stat.st_mtime_ns != expected['mtime_ns']:
            raise RuntimeError(f'Original checkpoint changed: {directory / name}')


def environment():
    env = os.environ.copy()
    prefix = Path(PLAN['env'])
    for key in list(env):
        if key.lower().endswith('_proxy'):
            env.pop(key)
    env.update(CUDA_VISIBLE_DEVICES='0,1,2,3', CONDA_PREFIX=str(prefix),
        PATH=str(prefix / 'bin') + ':' + env.get('PATH', ''), PYTHONPATH=PLAN['repo'],
        PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1', DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_HOME='/usr/local/cuda',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        TORCH_EXTENSIONS_DIR=str(HERE / 'extensions_cache'), MPLCONFIGDIR=str(HERE / 'matplotlib_cache'),
        TORCHINDUCTOR_CACHE_DIR=str(HERE / 'compile_cache'), TRITON_CACHE_DIR=str(HERE / 'triton_cache'))
    return env


def latest_metrics():
    rows = []
    for rank in range(4):
        path = HERE / 'training/metrics' / f'rank{rank}.jsonl'
        if not path.exists():
            continue
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 8192))
            lines = stream.read().decode().splitlines()
        for line in reversed(lines):
            try:
                rows.append(json.loads(line))
                break
            except json.JSONDecodeError:
                continue
    return rows


def validate_initial_restore():
    audits = []
    for rank in range(4):
        file = HERE / 'training' / f'resume_audit_rank{rank}.json'
        if not file.exists():
            return False
        audit = json.loads(file.read_text())
        if 'cpu_rng_state_restored' not in audit:
            return False
        if audit['restored_global_step'] != 10000 or audit['optimizer_state_counts'] != [935, 737]:
            raise RuntimeError(f'Unexpected full restore state: {audit}')
        flags = ['model_load_strict', 'optimizer_load_strict', 'dataloader_state_restored',
                 'cpu_rng_state_restored', 'environ_meter_state_restored']
        if not all(audit.get(key) is True for key in flags):
            raise RuntimeError(f'Full restore checks failed: {audit}')
        expected = {k: v for k, v in PLAN['learning_rate'].items() if k != 'style'}
        if audit['schedule'] != expected:
            raise RuntimeError(f'Wrong continuation policy: {audit["schedule"]}')
        if any(not math.isclose(v, 2.5e-5, abs_tol=1e-12) for v in audit['continuation_lrs']):
            raise RuntimeError('Restored learning rate does not start at 2.5e-5')
        gc = json.loads((HERE / 'training' / f'gradient_checkpointing_rank{rank}.json').read_text())
        if not gc['requested'] or not gc['enabled_module_count']:
            raise RuntimeError(f'Gradient checkpointing disabled on rank {rank}')
        audits.append(audit)
    save_json(HERE / 'restore_verified.json', {'passed': True, 'ranks': audits, 'verified_at': time.time()})
    return True


def train_monitor(process, finished):
    verified = False
    try:
        with (HERE / 'training/gpu_samples.jsonl').open('x', buffering=1) as stream:
            while not finished.is_set():
                gpus = gpu_status()
                stream.write(json.dumps({'time': time.time(), 'gpus': gpus}) + '\n')
                metrics = latest_metrics()
                if not verified:
                    verified = validate_initial_restore()
                status = {'phase': 'training', 'child_pid': process.pid, 'controller_pid': os.getpid(),
                    'start_step': 10000, 'target_step': 15000,
                    'last_step': min((m['step'] for m in metrics), default=10000),
                    'full_restore_verified': verified, 'latest_rank_metrics': metrics,
                    'gpus': gpus, 'updated_at': time.time()}
                save_json(HERE / 'training_status.json', status)
                save_json(HERE / 'status.json', status)
                finished.wait(5)
    except BaseException as exc:
        save_json(HERE / 'monitor_failure.json', {'error': f'{type(exc).__name__}: {exc}'})
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)


def check_checkpoint(step):
    directory = HERE / 'training/checkpoints' / f'global_step_{step}'
    required = ['model/.metadata', 'optimizer/.metadata', 'hf_ckpt/model.safetensors.index.json']
    required += [f'extra_state/extra_state_rank_{r}.pt' for r in range(4)]
    for name in required:
        path = directory / name
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f'Incomplete checkpoint: {path}')
    shards = sorted(set(json.loads((directory / required[2]).read_text())['weight_map'].values()))
    if not shards or any(not (directory / 'hf_ckpt' / name).is_file() or
                         not (directory / 'hf_ckpt' / name).stat().st_size for name in shards):
        raise RuntimeError(f'HF export incomplete at step {step}')
    return {'step': step, 'directory': str(directory), 'hf_shards': shards}


def wait_for_resources():
    stable = 0
    with (HERE / 'queue_gpu_samples.jsonl').open('x', buffering=1) as stream:
        while stable < 3:
            check_original_checkpoint()
            readings = gpu_status()
            stream.write(json.dumps({'time': time.time(), 'gpus': readings}) + '\n')
            stable = stable + 1 if all(g['used_mib'] < 1024 for g in readings) else 0
            save_json(HERE / 'status.json', {'phase': 'waiting_for_free_gpus', 'controller_pid': os.getpid(),
                'start_step': 10000, 'target_step': 15000, 'last_step': 10000,
                'gpus': readings, 'consecutive_free_checks': stable,
                'reason': 'Waiting for all four GPUs 0-3 to be free; existing processes are retained',
                'updated_at': time.time()})
            if stable < 3:
                STOP.wait(30)


def stop_active():
    STOP.set()
    if ACTIVE is not None and ACTIVE.poll() is None:
        # Includes nested evaluation servers/clients, which create their own sessions.
        try:
            os.killpg(ACTIVE.pid, signal.SIGTERM)
            ACTIVE.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(ACTIVE.pid, signal.SIGKILL)
            ACTIVE.wait()
        except ProcessLookupError:
            pass


def main():
    global ACTIVE, PHASE
    with (HERE / 'launcher.pid').open('x') as stream:
        stream.write(str(os.getpid()) + '\n')
    report = json.loads((HERE / 'preflight_report.json').read_text())
    if report.get('passed') is not True:
        raise RuntimeError('Preflight must pass before launching')
    check_original_checkpoint()
    save_json(HERE / 'training_status.json', {'phase': 'waiting_for_free_gpus', 'last_step': 10000})
    print('QUEUED: full step10000 restore ->15000; cosine 2.5e-5 ->1.25e-5; GPUs0-3', flush=True)
    wait_for_resources()
    PHASE = 'training'
    command = [str(Path(PLAN['env']) / 'bin/python'), '-B', '-m', 'torch.distributed.run',
        '--standalone', '--nproc_per_node=4', str(HERE / 'source/train_continuation.py'),
        str(HERE / 'training/config.yaml')]
    with (HERE / 'training/train.log').open('x') as stream:
        ACTIVE = subprocess.Popen(command, cwd=PLAN['repo'], stdin=subprocess.DEVNULL,
            env={**environment(), 'LINGBOT_STAGE_OUTPUT': str(HERE / 'training')},
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    save_json(HERE / 'training/process.json', {'pid': ACTIVE.pid, 'command': command})
    print(f'TRAINING: pid={ACTIVE.pid}; full step10000 restore', flush=True)
    finished = threading.Event()
    watcher = threading.Thread(target=train_monitor, args=(ACTIVE, finished), daemon=True)
    watcher.start()
    try:
        code = ACTIVE.wait()
    finally:
        finished.set()
        watcher.join(timeout=15)
    (HERE / 'training/exit_code.txt').write_text(str(code) + '\n')
    if code or (HERE / 'monitor_failure.json').exists():
        raise RuntimeError(f'Training failed, exit={code}; inspect training/train.log and monitor_failure.json')
    if not validate_initial_restore():
        raise RuntimeError('Full restore audits are incomplete')
    check_original_checkpoint()
    checkpoints = [check_checkpoint(s) for s in range(11000, 15001, 1000)]
    metrics = latest_metrics()
    if len(metrics) != 4 or any(m['step'] != 15000 or not math.isclose(m['lr_after_step'], 1.25e-5, abs_tol=1e-12) for m in metrics):
        raise RuntimeError('Training did not reach step15000 with final LR 1.25e-5 on all ranks')
    (HERE / 'training_exit_code.txt').write_text('0\n')
    save_json(HERE / 'training_status.json', {'phase': 'complete', 'final_step': 15000,
        'final_lr': 1.25e-5, 'additional_updates': 5000, 'checkpoints': checkpoints,
        'full_restore_verified': True, 'completed_at': time.time()})
    evaluations = []
    for directory in map(Path, PLAN['eval_roots']):
        step = json.loads((directory / 'plan.json').read_text())['checkpoint_step']
        PHASE = f'evaluating_step{step}'
        save_json(HERE / 'status.json', {'phase': PHASE, 'training_complete': True, 'last_step': 15000,
            'evaluation_root': str(directory), 'completed_evaluations': evaluations, 'updated_at': time.time()})
        with (directory / 'scheduler.log').open('x') as stream:
            ACTIVE = subprocess.Popen(['/usr/bin/python3', '-B', '-u', str(directory / 'run_scheduled.py')],
                cwd=directory, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        code = ACTIVE.wait()
        if code or (directory / 'exit_code.txt').read_text().strip() != '0':
            raise RuntimeError(f'Evaluation step{step} failed: exit={code}; {directory}')
        evaluations.append({'step': step, 'directory': str(directory), 'summaries': {
            setting: json.loads((directory / setting / 'summary.json').read_text())
            for setting in ['clean2clean', 'clean2random']}})
        save_json(HERE / 'evaluation_results.json', evaluations)
    check_original_checkpoint()
    PHASE = 'complete'
    save_json(HERE / 'status.json', {'phase': PHASE, 'final_step': 15000, 'final_lr': 1.25e-5,
        'training_complete': True, 'evaluation_complete': True,
        'evaluation_results': str(HERE / 'evaluation_results.json'), 'completed_at': time.time()})
    (HERE / 'exit_code.txt').write_text('0\n')


def interrupted(signum, frame):
    raise KeyboardInterrupt(f'Signal {signum}')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        main()
    except BaseException as exc:
        stop_active()
        error = f'{type(exc).__name__}: {exc}'
        save_json(HERE / 'status.json', {'phase': 'failed', 'stage': PHASE, 'error': error,
            'updated_at': time.time()})
        if PHASE in ['waiting_for_free_gpus', 'training']:
            save_json(HERE / 'training_status.json', {'phase': 'failed', 'error': error})
            (HERE / 'training_exit_code.txt').write_text('1\n')
        (HERE / 'exit_code.txt').write_text('1\n')
        print('FAILED ' + error, flush=True)
        raise
