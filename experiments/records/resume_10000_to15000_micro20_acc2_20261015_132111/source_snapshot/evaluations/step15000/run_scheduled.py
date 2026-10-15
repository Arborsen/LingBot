"""Wait for the requested time and complete checkpoint, then evaluate in a new directory."""
import concurrent.futures
import csv
import datetime
import hashlib
import json
import os
import queue
import signal
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / 'plan.json').read_text())
LOCK = threading.RLock()
STOP = threading.Event()
OWNED = []
SERVERS = {}
ACTIVE = {}
SOURCES = {}
RESULTS = {}
PHASE = 'waiting_for_schedule'
CURRENT_SETTING = None
EVAL_START = None


def save_json(name, value):
    with LOCK:
        path = HERE / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
        temporary.replace(path)


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1][0] != 'Z'
    except FileNotFoundError:
        return False


def readiness():
    root = Path(PLAN['train_root'])
    state = json.loads((root / 'training_status.json').read_text())
    if state['phase'] == 'failed':
        raise RuntimeError(f'Training failed: {state.get("error")}')
    for name in ['training_exit_code.txt', 'training/exit_code.txt']:
        path = root / name
        if path.exists() and path.read_text().strip() != '0':
            raise RuntimeError(f'Training failed: {path.read_text().strip()} at {path}')
    if state['phase'] != 'complete':
        pid = int((root / 'launcher.pid').read_text())
        if not alive(pid):
            raise RuntimeError('Training controller exited without completing; evaluation not started')
        return False, {'reason': 'training_not_complete', 'last_step': state.get('last_step')}
    for name in ['training_exit_code.txt', 'training/exit_code.txt']:
        if not (root / name).exists():
            return False, {'reason': 'waiting_for_training_exit_marker'}
    checkpoint = Path(PLAN['checkpoint']).parent
    required = ['model/.metadata', 'optimizer/.metadata']
    required += [f'extra_state/extra_state_rank_{r}.pt' for r in range(4)]
    required += ['hf_ckpt/model.safetensors.index.json', 'hf_ckpt/config.json']
    for name in required:
        path = checkpoint / name
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f'Final checkpoint incomplete: {path}')
    index = json.loads((checkpoint / 'hf_ckpt/model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    if not shards or any(not (checkpoint / 'hf_ckpt' / n).is_file() or
                         not (checkpoint / 'hf_ckpt' / n).stat().st_size for n in shards):
        raise RuntimeError('Final checkpoint shards missing or empty')
    if state.get('final_step') != PLAN['training_final_step']:
        raise RuntimeError('Training has not reached the requested final step')
    check_assets()
    return True, {'checkpoint': str(checkpoint), 'hf_shards': shards, 'checkpoint_step': PLAN['checkpoint_step']}


def check_assets():
    train = Path(PLAN['train'])
    assets = train / 'inference_assets'
    manifest = json.loads((assets / 'normalization_manifest.json').read_text())
    for name, entry in manifest['assets'].items():
        if hashlib.sha256((assets / name).read_bytes()).hexdigest() != entry['sha256']:
            raise RuntimeError(f'Normalization asset changed: {name}')
    if hashlib.sha256((train / 'lingbotvla_cli.yaml').read_bytes()).hexdigest() != manifest['training_config']['sha256']:
        raise RuntimeError('Training config does not match archived normalization manifest')
    if (Path(PLAN['repo']) / 'configs/robot_configs/robotwin.yaml').read_bytes() != (assets / 'robotwin.yaml').read_bytes():
        raise RuntimeError('Robot mapping changed')
    for path, digest in json.loads((HERE / 'source_hashes.json').read_text()).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Simulator source/config changed: {path}')


def environment(slot, name):
    env = os.environ.copy()
    prefix = Path(PLAN['env_root']) / name
    for key in list(env):
        if key.lower().endswith('_proxy'):
            env.pop(key)
    env.update(PATH=str(prefix / 'bin') + ':' + env.get('PATH', ''), CONDA_PREFIX=str(prefix),
        CUDA_VISIBLE_DEVICES=str(slot['gpu']), CUDA_HOME='/usr/local/cuda',
        NO_PROXY='127.0.0.1,localhost,0.0.0.0', no_proxy='127.0.0.1,localhost,0.0.0.0',
        PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', ROBOTWIN_TEST_NUM='10',
        PYTHONWARNINGS='ignore::UserWarning',
        TORCHINDUCTOR_CACHE_DIR=str(HERE / 'compile_cache' / f'slot{slot["slot"]}'),
        TRITON_CACHE_DIR=str(HERE / 'triton_cache' / f'slot{slot["slot"]}'),
        MPLCONFIGDIR=str(HERE / 'matplotlib_cache' / f'slot{slot["slot"]}'))
    if name == 'RoboTwin':
        env['PYTHONPATH'] = ':'.join([str(Path(PLAN['root']) / 'reproduce_artifacts/robotwin_eval10/python_overlay'),
            str(prefix / 'lib/python3.10/site-packages'), PLAN['sim']])
    else:
        env['PYTHONPATH'] = PLAN['repo']
        env['QWEN3VL_PATH'] = '/data1/hs/User/Program/Compete/LingBot/models/modelscope/Qwen3-VL-4B-Instruct'
    return env


def launch(command, log, slot, name, cwd):
    with log.open('x', encoding='utf-8') as stream:
        proc = subprocess.Popen(command, cwd=cwd, env=environment(slot, name),
                                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    with LOCK:
        OWNED.append(proc)
        save_json('pids.json', {'controller': os.getpid(), 'children': [p.pid for p in OWNED]})
    return proc


def client_command(task, setting, slot, source):
    return [str(Path(PLAN['env_root']) / 'RoboTwin/bin/python'), '-u', str(HERE / 'eval_no_video.py'),
        '--config', 'policy/ACT/deploy_policy.yml', '--overrides', '--task_name', task,
        '--task_config', setting['task_config'], '--instruction_type', setting['instruction_type'],
        '--train_config_name', '0', '--seed', '0', '--policy_name', 'ACT', '--port', str(slot['port']),
        '--robo_name', 'robotwin', '--eval_video_log', 'False', '--output_dir', str(source / 'eval_results')]


def validate_task(task, setting, source, code, seconds):
    directory = source / 'eval_results' / task
    errors = []
    rows = []
    try:
        rows = [json.loads(line) for line in (directory / 'episodes.jsonl').read_text().splitlines()]
        assert len(rows) == 10 and [r['episode'] for r in rows] == list(range(10)), 'Expected consecutive 10 episodes'
        assert len({r['seed'] for r in rows}) == 10, 'Duplicate seeds'
        total = 0
        for row in rows:
            assert type(row['success']) is bool and row['video_recording'] is False, 'Invalid outcome/video flag'
            assert row['task'] == task and row['task_config'] == setting['task_config'], 'Wrong task/scene'
            assert row['instruction_type'] == setting['instruction_type'], 'Wrong instruction type'
            total += int(row['success'])
            assert row['cumulative_successes'] == total, 'Outcome/counter mismatch'
        result = (directory / '_result.txt').read_text()
        assert f'Instruction Type: {setting["instruction_type"]}' in result, 'Wrong result instructions'
        assert abs(float(result.splitlines()[-1]) - total / 10) < 1e-8, 'Final success ratio mismatch'
        assert not list(source.rglob('*.mp4')), 'Unexpected video recording'
    except (OSError, ValueError, KeyError, AssertionError) as exc:
        errors.append(str(exc))
    if code != 0:
        errors.append(f'Client exited with {code}')
    return dict(task=task, **setting, source=str(source), episodes=len(rows),
                successes=sum(r.get('success') is True for r in rows), seconds=seconds,
                validated=not errors, errors=errors)


def progress(extra=None):
    with LOCK:
        by_setting = {}
        for setting in PLAN['settings']:
            key = setting['setting']
            outcomes = [r for k, r in RESULTS.items() if k.startswith(key + '/')]
            rows = []
            for k, source in SOURCES.items():
                if not k.startswith(key + '/'):
                    continue
                task = k.split('/', 1)[1]
                ledger = source / 'eval_results' / task / 'episodes.jsonl'
                if ledger.exists():
                    for line in ledger.read_text().splitlines():
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            by_setting[key] = dict(**setting, completed_episodes=len(rows), expected_episodes=500,
                successes_so_far=sum(r['success'] for r in rows), completed_tasks=len(outcomes),
                validated_tasks=sum(r['validated'] for r in outcomes), results=outcomes)
        save_json('status.json', dict(phase=PHASE, scheduled_for=PLAN['schedule'],
            controller_pid=os.getpid(), current_setting=CURRENT_SETTING, video_recording=False,
            expected_episodes=1000, elapsed_eval_seconds=None if EVAL_START is None else round(time.time()-EVAL_START),
            active_slots=dict(ACTIVE), settings=by_setting, updated_at=time.time(), **(extra or {})))


def evaluate(task, setting, slot):
    key = setting['setting'] + '/' + task
    attempts = []
    for attempt in range(1, 4):
        if STOP.is_set() or SERVERS[slot['slot']].poll() is not None:
            raise RuntimeError(f'Inference server/worker stopped: {slot}')
        source = HERE / setting['setting'] / 'attempts' / task / f'attempt{attempt}'
        source.mkdir(parents=True, exist_ok=False)
        check_assets()
        started = time.time()
        with LOCK:
            SOURCES[key] = source
            ACTIVE[str(slot['slot'])] = dict(**slot, task=task, setting=setting['setting'], source=str(source))
        proc = launch(client_command(task, setting, slot, source), source / 'client.log', slot, 'RoboTwin', PLAN['sim'])
        with LOCK:
            ACTIVE[str(slot['slot'])]['client_pid'] = proc.pid
        progress()
        print(f'START {key}: GPU {slot["gpu"]} slot {slot["slot"]} attempt {attempt}', flush=True)
        code = proc.wait()
        result = validate_task(task, setting, source, code, time.time()-started)
        attempts.append(result)
        save_json(str(source.parent / 'attempts.json'), attempts)
        if result['validated']:
            with LOCK:
                RESULTS[key] = result
                ACTIVE.pop(str(slot['slot']), None)
            progress()
            print(f'DONE {key}: {result["successes"]}/10, validated', flush=True)
            return
        print(f'RETRY {key}: {result["errors"]}', flush=True)
    raise RuntimeError(f'All attempts failed: {key}: {result["errors"]}')


def worker(slot, setting, tasks):
    while not STOP.is_set():
        try:
            task = tasks.get_nowait()
        except queue.Empty:
            return
        evaluate(task, setting, slot)


def cleanup():
    STOP.set()
    for proc in OWNED:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for proc in OWNED:
        if proc.poll() is None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


def start_servers():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for slot in PLAN['slots']:
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', slot['port']))
    for slot in PLAN['slots']:
        command = [str(Path(PLAN['env_root']) / 'lerobot/bin/python'), '-u', str(HERE / 'policy_server.py'),
                   '--port', str(slot['port'])]
        SERVERS[slot['slot']] = launch(command, HERE / f'server_slot{slot["slot"]}.log', slot, 'lerobot', PLAN['repo'])
    deadline = time.monotonic() + 1200
    pending = dict(SERVERS)
    while pending:
        for slot in PLAN['slots']:
            if slot['slot'] not in pending:
                continue
            if pending[slot['slot']].poll() is not None:
                raise RuntimeError(f'Inference server exited: slot {slot["slot"]}')
            try:
                with opener.open(f'http://127.0.0.1:{slot["port"]}/healthz', timeout=2) as response:
                    if response.status == 200:
                        pending.pop(slot['slot'])
            except OSError:
                pass
        if pending and time.monotonic() > deadline:
            raise RuntimeError(f'Inference startup timed out: {list(pending)}')
        if pending:
            time.sleep(5)


def main():
    global PHASE, CURRENT_SETTING, EVAL_START
    assert len(PLAN['tasks']) == len(set(PLAN['tasks'])) == 50
    assert PLAN['eval_video_log'] is False and len(PLAN['slots']) == 8
    with (HERE / 'controller.pid').open('x') as stream:
        stream.write(str(os.getpid()) + '\n')
    when = datetime.datetime.fromisoformat(PLAN['schedule']).timestamp()
    progress()
    print(f'SCHEDULED {PLAN["schedule"]}; step{PLAN["checkpoint_step"]} only, clean/seen + randomized/unseen, no video', flush=True)
    while time.time() < when:
        STOP.wait(min(30, when-time.time()))
    PHASE = 'waiting_for_training'
    while True:
        ready, detail = readiness()
        progress({'readiness': detail})
        if ready:
            break
        STOP.wait(30)
    save_json('checkpoint_validation.json', detail)
    # Read saved rank states on CPU after training exits, without loading model weights.
    check = subprocess.run([str(Path(PLAN['env_root']) / 'lerobot/bin/python'), str(HERE / 'verify_checkpoint.py')],
                           capture_output=True, text=True)
    (HERE / 'checkpoint_validation.log').write_text(check.stdout + check.stderr)
    if check.returncode:
        raise RuntimeError('Final checkpoint state validation failed; see checkpoint_validation.log')
    PHASE = 'waiting_for_free_gpus'
    while True:
        usage = subprocess.check_output(['nvidia-smi', '--id=0,1,2,3', '--query-gpu=memory.used',
            '--format=csv,noheader,nounits'], text=True)
        readings = [int(v.strip()) for v in usage.splitlines()]
        progress({'gpu_memory_mib': readings})
        if all(v < 1024 for v in readings):
            break
        STOP.wait(30)
    EVAL_START = time.time()
    PHASE = 'starting_servers'
    progress()
    start_servers()
    def monitor():
        while not STOP.wait(15):
            progress()
    threading.Thread(target=monitor, daemon=True).start()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    try:
        for setting in PLAN['settings']:
            CURRENT_SETTING = setting['setting']
            PHASE = 'evaluating'
            tasks = queue.Queue()
            for task in PLAN['tasks']:
                tasks.put(task)
            futures = [executor.submit(worker, slot, setting, tasks) for slot in PLAN['slots']]
            for future in concurrent.futures.as_completed(futures):
                future.result()
            selected = [r for r in RESULTS.values() if r['setting'] == CURRENT_SETTING]
            assert len(selected) == 50 and all(r['validated'] and r['episodes'] == 10 for r in selected)
            successes = sum(r['successes'] for r in selected)
            save_json(f'{CURRENT_SETTING}/summary.json', dict(**setting, successes=successes, episodes=500,
                success_rate=successes/500, validated=True, video_recording=False, tasks=selected))
            with (HERE / CURRENT_SETTING / 'stats.txt').open('x') as stream:
                stream.write(f'{CURRENT_SETTING}: {successes}/500 = {successes/5:.1f}%\nVideo recording: OFF\nAll 50 tasks validated.\n')
    finally:
        cleanup()
        executor.shutdown(wait=True, cancel_futures=True)
    PHASE = 'complete'
    progress()
    (HERE / 'exit_code.txt').write_text('0\n')


def interrupted(signum, frame):
    raise KeyboardInterrupt(f'Signal {signum}')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        main()
    except BaseException as exc:
        cleanup()
        PHASE = 'failed'
        progress({'error': f'{type(exc).__name__}: {exc}'})
        (HERE / 'exit_code.txt').write_text('1\n')
        print(f'FAILED {type(exc).__name__}: {exc}', flush=True)
        raise
