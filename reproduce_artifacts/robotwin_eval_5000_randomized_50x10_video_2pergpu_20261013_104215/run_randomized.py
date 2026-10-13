"""Run randomized evaluation after clean finishes; retain clean results and videos."""
import concurrent.futures
import csv
import hashlib
import importlib.util
import json
import os
import queue
import re
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'robotwin_eval_5000_10x10_video_2pergpu_20261013_093256'
SPEC = importlib.util.spec_from_file_location('previous_parallel_runner', PREVIOUS / 'run_parallel.py')
PREV = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREV)
BASE = PREV.BASE
BASE.HERE = HERE
PLAN = json.loads((HERE / 'adoption_plan.json').read_text(encoding='utf-8'))
TASKS = PLAN['tasks']
SLOTS = PLAN['slots']
SOURCES = {task: Path(PLAN['task_sources'].get(task, str(HERE))) for task in TASKS}
RESULTS = {}
ATTEMPTS = {}
ACTIVE = {}
OWNED = []
LOCK = threading.RLock()
STOP = threading.Event()
START = time.time()


def save_json(name, value):
    with LOCK:
        temporary = HERE / (name + '.tmp')
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        temporary.replace(HERE / name)


def check_assets():
    assets = BASE.TRAIN / 'inference_assets'
    manifest = json.loads((assets / 'normalization_manifest.json').read_text(encoding='utf-8'))
    for name, entry in manifest['assets'].items():
        assert hashlib.sha256((assets / name).read_bytes()).hexdigest() == entry['sha256'], name
    assert (BASE.ROOT / 'lingbot-vla-v2/configs/robot_configs/robotwin.yaml').read_bytes() == (assets / 'robotwin.yaml').read_bytes()
    assert hashlib.sha256((BASE.TRAIN / 'lingbotvla_cli.yaml').read_bytes()).hexdigest() == manifest['training_config']['sha256']


def progress():
    with LOCK:
        source_pairs = [(task, SOURCES[task]) for task in TASKS]
        active = dict(ACTIVE)
        results = dict(RESULTS)
    episodes = {}
    duplicates = []
    for task, source in source_pairs:
        for video in sorted((source / 'eval_results' / task).glob('episode*_*.mp4')):
            match = re.fullmatch(r'episode(\d+)_(success|failure)\.mp4', video.name)
            if not match or int(match[1]) not in range(10):
                continue
            key = (task, int(match[1]))
            if key in episodes:
                duplicates.append({'task': task, 'episode': key[1]})
            episodes[key] = {'task': task, 'episode': key[1], 'video': str(video), 'success': match[2] == 'success'}
    save_json('progress.json', {'completed_episodes': len(episodes), 'expected_episodes': 500,
        'completed_tasks': len(results), 'validated_tasks': sum(r['validated'] for r in results.values()),
        'total_tasks': 50, 'successes_so_far': sum(e['success'] for e in episodes.values()),
        'elapsed_seconds': round(time.time() - START), 'task_config': 'demo_randomized', 'phase': 'running', 'active_slots': active,
        'duplicate_episode_ids': duplicates, 'episodes': list(episodes.values()), 'results': results})


def record(result):
    with LOCK:
        RESULTS[result['task']] = result
        SOURCES[result['task']] = Path(result['source'])
    progress()
    print(f'DONE {result["task"]}: {result["successes"]}/10, videos={result["episodes"]}, validated={result["validated"]}', flush=True)


def spawn(task, slot, source):
    log = source / 'eval_logs' / f'{task}.log'
    (source / 'eval_logs').mkdir(parents=True, exist_ok=True)
    (source / 'eval_results').mkdir(parents=True, exist_ok=True)
    assert not (source / 'eval_results' / task).exists(), f'Output already exists: {task}'
    cmd = [str(BASE.ENV_ROOT / 'RoboTwin/bin/python'), '-u', str(HERE / 'eval_with_video.py'),
           '--config', 'policy/ACT/deploy_policy.yml', '--overrides', '--task_name', task,
           '--task_config', 'demo_randomized', '--train_config_name', '0', '--seed', '0',
           '--policy_name', 'ACT', '--port', str(slot['port']), '--robo_name', 'robotwin',
           '--video_fps', '10', '--eval_video_log', 'True', '--output_dir', str(source / 'eval_results')]
    env = BASE.environment(slot['gpu'], 'RoboTwin')
    env['PYTHONUTF8'] = '1'
    with log.open('x', encoding='utf-8') as stream:
        proc = subprocess.Popen(cmd, env=env, cwd=BASE.SIM, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
    with LOCK:
        OWNED.append(proc)
        ACTIVE[str(slot['slot'])]['client_pid'] = proc.pid
        save_json('pids.json', {'launcher': os.getpid(), 'paused_controllers': PLAN['paused_controllers'],
                              'slots': SLOTS, 'owned_children': [p.pid for p in OWNED]})
    return proc


def evaluate(task, slot, previous_attempt=None):
    attempts = [previous_attempt] if previous_attempt else []
    result = None
    for number in range(1, 4):
        if STOP.is_set():
            raise RuntimeError('Evaluation stopped')
        check_assets()
        state, _ = PREV.process_state(slot['server_pid'])
        if state is None or state == 'Z':
            raise RuntimeError(f'Inference slot {slot["slot"]} exited')
        source = HERE if number == 1 and previous_attempt is None else HERE / 'retry_attempts' / task / f'attempt{number}'
        begin = time.time()
        with LOCK:
            SOURCES[task] = source
            ACTIVE[str(slot['slot'])] = {**slot, 'task': task, 'source': str(source), 'adopted': False}
        progress()
        print(f'START {task}: GPU {slot["gpu"]}, port {slot["port"]}, slot {slot["slot"]}, attempt {number}', flush=True)
        proc = spawn(task, slot, source)
        code = proc.wait()
        result = PREV.validate_task(task, source, code, time.time() - begin)
        attempts.append(result)
        with LOCK:
            ATTEMPTS[task] = attempts
            save_json('attempts.json', ATTEMPTS)
        if result['validated']:
            break
        print(f'RETRY {task}: {result["errors"]}; files retained at {source}', flush=True)
    record(result)


def slot_worker(slot, tasks):
    PREV.wait_ready(slot['port'], slot['server_pid'])
    current_task = slot.get('task')
    if current_task:
        source = Path(slot['source'])
        with LOCK:
            ACTIVE[str(slot['slot'])] = {**slot, 'adopted': True}
        print(f'ADOPT {current_task}: GPU {slot["gpu"]}, port {slot["port"]}, client {slot["client_pid"]}; no restart', flush=True)
        while not STOP.is_set():
            state, code = PREV.process_state(slot['client_pid'])
            if state is None or state == 'Z':
                if code is None:
                    code = 0 if (source / 'eval_results' / current_task / '_result.txt').exists() else -1
                result = PREV.validate_task(current_task, source, code, time.time() - slot['started_at'])
                if result['validated']:
                    record(result)
                else:
                    evaluate(current_task, slot, previous_attempt=result)
                with LOCK:
                    ACTIVE.pop(str(slot['slot']), None)
                break
            time.sleep(3)
    while not STOP.is_set():
        try:
            task = tasks.get_nowait()
        except queue.Empty:
            return
        evaluate(task, slot)
        with LOCK:
            ACTIVE.pop(str(slot['slot']), None)


def monitor():
    while not STOP.wait(15):
        progress()


def cleanup(success):
    # Own only randomized processes; never signal the clean coordinator.
    STOP.set()
    PREV.STOP.set()
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


def wait_for_clean_and_resources():
    check_assets()
    prerequisite = Path(PLAN['prerequisite_clean'])
    clean_plan = json.loads((prerequisite / 'adoption_plan.json').read_text(encoding='utf-8'))
    old_servers = [slot['server_pid'] for slot in clean_plan['slots']]
    print(f'QUEUED: randomized 50 x 10 episodes after {prerequisite}; clean processes unchanged', flush=True)
    while not (prerequisite / 'exit_code').exists():
        coordinator = int((prerequisite / 'coordinator.pid').read_text())
        if PREV.process_state(coordinator)[0] in (None, 'Z'):
            # The completion marker might be written at the same instant as exit.
            if not (prerequisite / 'exit_code').exists():
                raise RuntimeError('Clean coordinator exited without its completion marker')
        save_json('queue_state.json', {'phase': 'waiting_for_clean', 'clean_source': str(prerequisite),
                                     'expected_episodes': 500, 'task_config': 'demo_randomized'})
        time.sleep(15)
    report = json.loads((prerequisite / 'results.json').read_text(encoding='utf-8'))
    if (prerequisite / 'exit_code').read_text().strip() != '0' or not report['complete']:
        raise RuntimeError('Clean evaluation did not finish successfully; randomized remains unstarted')
    # Snapshot every completed clean video before launching randomized work.
    hashes = {video: hashlib.sha256(Path(video).read_bytes()).hexdigest()
              for result in report['results'].values() for video in result['videos']}
    assert len(hashes) == 500
    save_json('preserved_videos.json', hashes)
    while True:
        servers_stopped = all(PREV.process_state(pid)[0] in (None, 'Z') for pid in old_servers)
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                       '--format=csv,noheader,nounits'], text=True)
        free = {int(row.split(',')[0]): int(row.split(',')[1]) for row in rows.splitlines()}
        ports_free = True
        for slot in SLOTS:
            with socket.socket() as sock:
                try:
                    sock.bind(('127.0.0.1', slot['port']))
                except OSError:
                    ports_free = False
        if servers_stopped and ports_free and all(free[gpu] >= 65000 for gpu in range(4)):
            break
        save_json('queue_state.json', {'phase': 'waiting_for_gpu_resources', 'old_servers_stopped': servers_stopped,
                                     'ports_free': ports_free, 'gpu_free_mib': free})
        time.sleep(15)
    save_json('queue_state.json', {'phase': 'starting_randomized', 'clean_source': str(prerequisite)})


def start_servers():
    try:
        for slot in SLOTS:
            log = HERE / 'inference_logs' / f'slot{slot["slot"]}.log'
            log.parent.mkdir(exist_ok=True)
            env = BASE.environment(slot['gpu'], 'lerobot')
            env['PYTHONUTF8'] = '1'
            # Share the clean compile cache on the same GPU to reduce warm-up time.
            env['TORCHINDUCTOR_CACHE_DIR'] = str(Path(PLAN['prerequisite_clean']) / 'compile_cache' / str(slot['gpu']))
            env['TRITON_CACHE_DIR'] = str(Path(PLAN['prerequisite_clean']) / 'triton_cache' / str(slot['gpu']))
            with log.open('x', encoding='utf-8') as stream:
                proc = subprocess.Popen([str(BASE.ENV_ROOT / 'lerobot/bin/python'), '-u',
                    str(HERE / 'policy_server.py'), '--port', str(slot['port'])],
                    env=env, cwd=BASE.ROOT / 'lingbot-vla-v2', stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
            OWNED.append(proc)
            slot['server_pid'] = proc.pid
        save_json('pids.json', {'launcher': os.getpid(), 'slots': SLOTS,
                              'owned_children': [proc.pid for proc in OWNED]})
    except BaseException:
        cleanup(False)
        raise


def main():
    with (HERE / 'coordinator.pid').open('x', encoding='utf-8') as file:
        file.write(str(os.getpid()) + '\n')
    assert len(TASKS) == len(set(TASKS)) == 50
    assert len(SLOTS) == 8 and all(sum(s['gpu'] == gpu for s in SLOTS) == 2 for gpu in range(4))
    assert all(PREV.process_state(pid)[0] in ('T', 't') for pid in PLAN['paused_controllers'])
    check_assets()
    assert (BASE.SIM / 'task_config/demo_randomized.yml').read_bytes() == (HERE / 'demo_randomized.snapshot.yml').read_bytes()
    start_servers()
    save_json('queue_state.json', {'phase': 'running', 'task_config': 'demo_randomized', 'expected_episodes': 500})
    save_json('run_config.json', {'checkpoint': str(BASE.TRAIN / 'checkpoints/global_step_5000/hf_ckpt'),
        'normalization': str(BASE.TRAIN / 'inference_assets/norm_stats.json'), 'tasks': TASKS,
        'episodes_per_task': 10, 'expected_episodes': 500, 'task_config': 'demo_randomized',
        'instruction_type': 'unseen', 'seed': 0, 'precision': 'fp32', 'compile': True,
        'chunk_length': 50, 'video_fps': 10, 'gpus': [0, 1, 2, 3], 'slots_per_gpu': 2,
        'slots': SLOTS, 'task_list_source': PLAN['task_list_source'],
        'preserved_sources': PLAN['preserved_sources'], 'new_output': str(HERE),
        'training_data': 'clean only; same archived clean normalization', 'official_final_protocol': False})
    adopted = {slot['task'] for slot in SLOTS if slot.get('task')}
    for task, source in PLAN['completed_tasks'].items():
        result = PREV.validate_task(task, Path(source), 0, 0)
        assert result['validated'], result
        record(result)
    tasks = queue.Queue()
    pending = [task for task in BASE.TASKS if task not in adopted and task not in RESULTS]
    pending += [task for task in TASKS if task not in BASE.TASKS and task not in adopted and task not in RESULTS]
    assert len(set(pending) | adopted | set(RESULTS)) == 50
    assert not (set(pending) & adopted or set(pending) & set(RESULTS) or adopted & set(RESULTS))
    for task in pending:
        tasks.put(task)
    print(f'50 tasks: {len(RESULTS)} complete, {len(adopted)} already running, {len(pending)} queued; 500 episodes total', flush=True)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    error = None
    try:
        futures = [executor.submit(slot_worker, slot, tasks) for slot in SLOTS]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        print(error, flush=True)
    valid = len(RESULTS) == 50 and all(r['validated'] for r in RESULTS.values()) and error is None
    cleanup(valid)
    executor.shutdown(wait=True, cancel_futures=True)
    monitor_thread.join()
    preserved_ok = True
    for path, digest in json.loads((HERE / 'preserved_videos.json').read_text(encoding='utf-8')).items():
        if not Path(path).exists() or hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            preserved_ok = False
            error = f'Previously completed video changed: {path}'
            valid = False
    if valid:
        for result in RESULTS.values():
            for video in result['videos']:
                try:
                    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', video, '-f', 'null', '-'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, timeout=120)
                except Exception as exc:
                    result['validated'] = False
                    result['errors'].append(f'Video decode failed: {exc}')
                    valid = False
    total = sum(r['successes'] for r in RESULTS.values() if r['validated'])
    report = {'complete': valid, 'successes': total, 'expected_episodes': 500,
              'success_rate': total / 500 if valid else None, 'seconds_since_expansion': round(time.time() - START, 1),
              'preserved_videos_unchanged': preserved_ok, 'error': error, 'results': RESULTS}
    save_json('results.json', report)
    progress()
    lines = ['Checkpoint: global_step_5000', 'Scene: demo_randomized; instruction_type: unseen; seed: 0',
             'Tasks: 50 distinct tasks x 10 episodes; GPU 0-3 x 2 slots; FP32; compile=True; chunk=50; video=10 fps',
             f'Complete and validated: {valid}', f'Original completed videos unchanged: {preserved_ok}']
    with (HERE / 'task_results.csv').open('x', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['task', 'successes', 'episodes', 'success_rate', 'validated', 'source'])
        for task in TASKS:
            result = RESULTS.get(task)
            if result:
                writer.writerow([task, result['successes'], result['episodes'], result['successes'] / 10, result['validated'], result['source']])
                lines.append(f'{task}: {result["successes"]}/10, videos={result["episodes"]}, validated={result["validated"]}')
            else:
                writer.writerow([task, '', '', '', False, ''])
                lines.append(f'{task}: incomplete')
    lines.append(f'Total: {total}/500 ({total / 5:.1f}%)' if valid else 'Total: incomplete; see results.json')
    lines.append(f'Elapsed since expansion: {report["seconds_since_expansion"]} seconds')
    (HERE / 'stats.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    save_json('queue_state.json', {'phase': 'complete' if valid else 'failed', 'error': error})
    (HERE / 'exit_code').write_text('0\n' if valid else '1\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return 0 if valid else 1


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, PREV.stop_signal)
    signal.signal(signal.SIGINT, PREV.stop_signal)
    try:
        wait_for_clean_and_resources()
        START = time.time()
        code = main()
    except BaseException as exc:
        cleanup(False)
        save_json('queue_state.json', {'phase': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        (HERE / 'exit_code').write_text('1\n', encoding='utf-8')
        raise
    raise SystemExit(code)
