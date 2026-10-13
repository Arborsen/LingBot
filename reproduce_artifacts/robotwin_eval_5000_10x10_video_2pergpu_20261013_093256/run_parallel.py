"""Adopt four running tasks and add four GPU slots without replacing outputs."""
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import queue
import re
import signal
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'robotwin_eval_5000_10x10_video_20261013_091713'
ROOT = HERE.parents[1]
SPEC = importlib.util.spec_from_file_location('previous_runner', OLD / 'run_eval.py')
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
BASE.HERE = HERE
TASKS = BASE.TASKS
PLAN = json.loads((HERE / 'adoption_plan.json').read_text())
RESULTS = {}
LOCK = threading.RLock()
STOP = threading.Event()
OWNED = []
ADOPTED = PLAN['old_slots']
START = time.time()
ACTIVE = {}
ERRORS = []


def save_json(name, value):
    with LOCK:
        tmp = HERE / (name + '.tmp')
        tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
        tmp.replace(HERE / name)


def process_state(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
    except FileNotFoundError:
        return None, None
    state = fields[0]
    code = os.waitstatus_to_exitcode(int(fields[49])) if state == 'Z' else None
    return state, code


def spawn(command, env, cwd, log):
    # Exclusive create prevents accidental replacement even on relaunch.
    with log.open('x') as stream:
        proc = subprocess.Popen(command, env=env, cwd=cwd, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
    with LOCK:
        OWNED.append(proc)
        save_json('pids.json', {'launcher': os.getpid(), 'old_launcher': PLAN['old_launcher_pid'],
                              'adopted_slots': ADOPTED, 'owned_children': [p.pid for p in OWNED]})
    return proc


def validate_task(task, source, code, seconds):
    folder = source / 'eval_results' / task
    videos = sorted(folder.glob('episode*_*.mp4'))
    errors = []
    if code != 0:
        errors.append(f'Client exit code {code}')
    if len(videos) != 10:
        errors.append(f'Expected 10 videos, found {len(videos)}')
    if {int(re.match(r'episode(\d+)_', p.name).group(1)) for p in videos} != set(range(10)):
        errors.append('Episode IDs are incomplete or duplicated')
    metadata = []
    for video in videos:
        try:
            info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                 '-show_entries', 'stream=nb_frames,width,height,duration,r_frame_rate', '-of', 'json', str(video)], text=True))['streams'][0]
            if int(info['nb_frames']) < 2 or float(info['duration']) <= 0:
                errors.append(f'Empty video: {video.name}')
            metadata.append({'path': str(video), **info})
        except Exception as exc:
            errors.append(f'Invalid video {video.name}: {exc}')
    log = source / 'eval_logs' / f'{task}.log'
    text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', log.read_text(errors='replace'))
    rates = re.findall(r'Success rate:\s*(\d+)/(\d+).*?current seed:\s*(\d+)', text)
    successes = sum(p.name.endswith('_success.mp4') for p in videos)
    result_file = folder / '_result.txt'
    if len(rates) != 10 or rates[-1][:2] != (str(successes), '10') or not result_file.exists():
        errors.append('Final client result does not match the ten episode videos')
    elif abs(float(result_file.read_text().splitlines()[-1]) - successes / 10) > 1e-8:
        errors.append('Result file success rate mismatch')
    if len({r[2] for r in rates}) != 10:
        errors.append('Evaluation seeds missing or duplicated')
    return {'task': task, 'successes': successes, 'episodes': len(videos), 'seconds': round(seconds, 1),
            'exit_code': code, 'validated': not errors, 'errors': errors, 'source': str(source),
            'seeds': [int(r[2]) for r in rates], 'videos': [str(v) for v in videos], 'video_metadata': metadata}


def progress():
    sources = {slot['task']: OLD for slot in ADOPTED}
    episodes = []
    for task in TASKS:
        source = sources.get(task, HERE)
        for video in sorted((source / 'eval_results' / task).glob('episode*_*.mp4')):
            episodes.append({'task': task, 'video': str(video), 'success': video.name.endswith('_success.mp4')})
    with LOCK:
        save_json('progress.json', {'completed_episodes': len(episodes), 'expected_episodes': 100,
            'completed_tasks': len(RESULTS), 'active_slots': dict(ACTIVE), 'successes_so_far': sum(e['success'] for e in episodes),
            'elapsed_seconds': round(time.time() - START), 'episodes': episodes, 'results': dict(RESULTS)})


def record(result):
    with LOCK:
        RESULTS[result['task']] = result
    progress()
    print(f'DONE {result["task"]}: {result["successes"]}/10, videos={result["episodes"]}, validated={result["validated"]}', flush=True)


def wait_ready(port, pid):
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline and not STOP.is_set():
        state, code = process_state(pid)
        if state is None or state == 'Z':
            raise RuntimeError(f'Server {pid} on port {port} exited: {code}')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f'Server on port {port} was not ready')


def run_queue(slot, gpu, port, server_pid, tasks):
    while not STOP.is_set():
        try:
            task = tasks.get_nowait()
        except queue.Empty:
            return
        state, _ = process_state(server_pid)
        if state is None or state == 'Z':
            raise RuntimeError(f'Inference slot {slot} died')
        begin = time.time()
        with LOCK:
            ACTIVE[str(slot)] = {'gpu': gpu, 'port': port, 'task': task, 'source': str(HERE)}
        progress()
        print(f'START {task}: GPU {gpu}, port {port}, slot {slot}', flush=True)
        cmd = [str(BASE.ENV_ROOT / 'RoboTwin/bin/python'), '-u', str(HERE / 'eval_with_video.py'),
               '--config', 'policy/ACT/deploy_policy.yml', '--overrides', '--task_name', task,
               '--task_config', 'demo_clean', '--train_config_name', '0', '--seed', '0',
               '--policy_name', 'ACT', '--port', str(port), '--robo_name', 'robotwin',
               '--video_fps', '10', '--eval_video_log', 'True', '--output_dir', str(HERE / 'eval_results')]
        proc = spawn(cmd, BASE.environment(gpu, 'RoboTwin'), BASE.SIM, HERE / 'eval_logs' / f'{task}.log')
        code = proc.wait()
        record(validate_task(task, HERE, code, time.time() - begin))
        with LOCK:
            ACTIVE.pop(str(slot), None)


def adopt_worker(slot, tasks):
    gpu, port, pid, task = slot['gpu'], slot['port'], slot['client_pid'], slot['task']
    with LOCK:
        ACTIVE[str(gpu)] = {'gpu': gpu, 'port': port, 'task': task, 'source': str(OLD), 'adopted': True}
    print(f'ADOPT {task}: GPU {gpu}, port {port}, existing client {pid}; no restart', flush=True)
    while not STOP.is_set():
        state, code = process_state(pid)
        if state is None or state == 'Z':
            if code is None:
                code = 0 if (OLD / 'eval_results' / task / '_result.txt').exists() else -1
            record(validate_task(task, OLD, code, time.time() - START))
            with LOCK:
                ACTIVE.pop(str(gpu), None)
            run_queue(gpu, gpu, port, slot['server_pid'], tasks)
            return
        time.sleep(3)


def new_worker(gpu, tasks):
    slot, port = gpu + 4, 9380 + gpu
    proc = spawn([str(BASE.ENV_ROOT / 'lerobot/bin/python'), '-u', str(HERE / 'policy_server.py'),
                  '--port', str(port)], BASE.environment(gpu, 'lerobot'), ROOT / 'lingbot-vla-v2',
                 HERE / 'inference_logs' / f'slot{slot}_gpu{gpu}_port{port}.log')
    wait_ready(port, proc.pid)
    print(f'Slot {slot} ready: GPU {gpu}, port {port}', flush=True)
    run_queue(slot, gpu, port, proc.pid, tasks)


def monitor():
    while not STOP.wait(15):
        progress()


def cleanup(success):
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
    old_pid = PLAN['old_launcher_pid']
    try:
        if success:
            # Old task workers were paused to prevent duplicate queue dispatch.
            # All their original tasks and the added tasks are now complete.
            os.kill(old_pid, signal.SIGTERM)
        os.kill(old_pid, signal.SIGCONT)
    except ProcessLookupError:
        pass


def stop_signal(signum, frame):
    raise KeyboardInterrupt(f'Signal {signum}')


def main():
    (HERE / 'coordinator.pid').write_text(str(os.getpid()) + '\n')
    assert process_state(PLAN['old_launcher_pid'])[0] in ('T', 't'), 'Old dispatcher must be paused before adoption'
    for directory in ['eval_results', 'eval_logs', 'inference_logs']:
        (HERE / directory).mkdir(exist_ok=True)
    tasks = queue.Queue()
    old_tasks = {slot['task'] for slot in ADOPTED}
    for task in TASKS:
        if task not in old_tasks:
            tasks.put(task)
    save_json('run_config.json', {'checkpoint': str(BASE.TRAIN / 'checkpoints/global_step_5000/hf_ckpt'),
        'normalization': str(BASE.TRAIN / 'inference_assets/norm_stats.json'), 'tasks': TASKS,
        'episodes_per_task': 10, 'task_config': 'demo_clean', 'instruction_type': 'unseen', 'seed': 0,
        'precision': 'fp32', 'compile': True, 'chunk_length': 50, 'video_fps': 10,
        'gpus': [0, 1, 2, 3], 'slots_per_gpu': 2, 'ports': [9370, 9371, 9372, 9373, 9380, 9381, 9382, 9383],
        'preserved_source': str(OLD), 'adopted_tasks': sorted(old_tasks), 'new_output': str(HERE)})
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    error = None
    try:
        futures = [executor.submit(adopt_worker, slot, tasks) for slot in ADOPTED]
        futures += [executor.submit(new_worker, gpu, tasks) for gpu in range(4)]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        print(error, flush=True)
    valid = len(RESULTS) == 10 and all(r['validated'] for r in RESULTS.values()) and error is None
    cleanup(valid)
    executor.shutdown(wait=True, cancel_futures=True)
    monitor_thread.join()
    preserved = json.loads((HERE / 'preserved_videos.json').read_text())
    for path, digest in preserved.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
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
    report = {'complete': valid, 'successes': total, 'expected_episodes': 100,
              'success_rate': total / 100 if valid else None, 'seconds_since_expansion': round(time.time() - START, 1),
              'preserved_videos_unchanged': error is None, 'error': error, 'results': RESULTS}
    save_json('results.json', report)
    progress()
    lines = ['Checkpoint: global_step_5000', 'Scene: demo_clean; instruction_type: unseen; seed: 0',
             'GPUs: 0,1,2,3; 2 inference/simulation slots per GPU; FP32; compile=True; chunk=50; video=10 fps',
             f'Complete and validated: {valid}', f'Original results preserved at: {OLD}', f'New results at: {HERE}']
    for task in TASKS:
        r = RESULTS.get(task)
        lines.append(f'{task}: {r["successes"]}/10, videos={r["episodes"]}, validated={r["validated"]}' if r else f'{task}: incomplete')
    lines.append(f'Total: {total}/100 ({total}%)' if valid else 'Total: incomplete; see results.json')
    lines.append(f'Elapsed after expansion: {report["seconds_since_expansion"]} seconds')
    (HERE / 'stats.txt').write_text('\n'.join(lines) + '\n')
    (HERE / 'exit_code').write_text('0\n' if valid else '1\n')
    print('\n'.join(lines), flush=True)
    return 0 if valid else 1


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    raise SystemExit(main())
