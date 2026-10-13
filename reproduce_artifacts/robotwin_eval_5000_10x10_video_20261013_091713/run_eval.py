"""Four GPU slots; ten distinct tasks; ten valid episodes and videos per task."""
import concurrent.futures
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
ROOT = HERE.parents[1]
TRAIN = ROOT / 'reproduce_artifacts/official_dist_muon_batch160_seed42_5000_20261011_163354'
SIM = ROOT / 'RoboTwin'
ENV_ROOT = Path('/data1/hs/miniconda3/envs')
TASKS = ['lift_pot', 'click_bell', 'click_alarmclock', 'turn_switch', 'stamp_seal',
         'place_fan', 'put_object_cabinet', 'open_microwave', 'stack_blocks_three', 'place_shoe']
GPUS = [0, 1, 2, 3]
PORTS = [9370, 9371, 9372, 9373]
CHILDREN = []
LOCK = threading.RLock()
RESULTS = {}
SERVERS = {}
START = time.time()
STOP = threading.Event()


def save_json(name, value):
    tmp = HERE / (name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(HERE / name)


def environment(gpu, name):
    env = os.environ.copy()
    for key in list(env):
        if key.lower().endswith('_proxy'):
            env.pop(key)
    env['NO_PROXY'] = '127.0.0.1,localhost,0.0.0.0'
    prefix = ENV_ROOT / name
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), CONDA_PREFIX=str(prefix),
               PATH=str(prefix / 'bin') + ':' + env.get('PATH', ''),
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               SETUPTOOLS_SCM_PRETEND_VERSION='0.0.0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               CUDA_HOME='/usr/local/cuda', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TORCHINDUCTOR_CACHE_DIR=str(HERE / 'compile_cache' / str(gpu)),
               TRITON_CACHE_DIR=str(HERE / 'triton_cache' / str(gpu)))
    if name == 'lerobot':
        env['QWEN3VL_PATH'] = '/data1/hs/User/Program/Compete/LingBot/models/modelscope/Qwen3-VL-4B-Instruct'
    else:
        env.update(ROBOTWIN_TEST_NUM='10', PYTHONWARNINGS='ignore::UserWarning', XLA_PYTHON_CLIENT_MEM_FRACTION='0.9')
        env['PYTHONPATH'] = ':'.join([str(ROOT / 'reproduce_artifacts/robotwin_eval10/python_overlay'),
                                    str(prefix / 'lib/python3.10/site-packages'), str(SIM),
                                    env.get('PYTHONPATH', '')])
    return env


def spawn(command, cwd, env, log):
    with log.open('w') as handle:
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=handle,
                                stderr=subprocess.STDOUT, start_new_session=True)
    with LOCK:
        CHILDREN.append(proc)
        save_json('pids.json', {'launcher': os.getpid(), 'children': [p.pid for p in CHILDREN]})
    return proc


def cleanup():
    STOP.set()
    for proc in CHILDREN:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 5
    for proc in CHILDREN:
        if proc.poll() is None:
            try:
                proc.wait(timeout=max(.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


def interrupted(signum, frame):
    raise KeyboardInterrupt(f'Signal {signum}')


def wait_ready(slot):
    server = SERVERS[slot]
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if STOP.is_set():
            raise RuntimeError('Evaluation stopped')
        if server.poll() is not None:
            raise RuntimeError(f'Inference slot {slot} exited with {server.returncode}; see inference_logs/slot{slot}.log')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORTS[slot]}/healthz', timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f'Inference slot {slot} was not ready after 600 seconds')


def validate_task(task, code, elapsed):
    folder = HERE / 'eval_results' / task
    videos = sorted(folder.glob('episode*_*.mp4'))
    errors = []
    if code != 0:
        errors.append(f'Client exit code {code}')
    if len(videos) != 10:
        errors.append(f'Expected 10 videos, found {len(videos)}')
    if {re.match(r'episode(\d+)_', p.name).group(1) for p in videos} != {str(i) for i in range(10)}:
        errors.append('Episode IDs are incomplete or duplicated')
    for video in videos:
        try:
            info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=nb_frames,width,height,duration', '-of', 'json', str(video)], text=True))['streams'][0]
            if int(info['nb_frames']) < 2 or float(info['duration']) <= 0:
                errors.append(f'Empty video: {video.name}')
        except Exception as exc:
            errors.append(f'Invalid video {video.name}: {exc}')
    text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', (HERE / 'eval_logs' / f'{task}.log').read_text(errors='replace'))
    rates = re.findall(r'Success rate:\s*(\d+)/(\d+)', text)
    successes = sum(p.name.endswith('_success.mp4') for p in videos)
    result_file = folder / '_result.txt'
    if not rates or rates[-1] != (str(successes), '10') or not result_file.exists():
        errors.append('Final client result does not match the 10 episode videos')
    else:
        try:
            if abs(float(result_file.read_text().splitlines()[-1]) - successes / 10) > 1e-8:
                errors.append('Result file success rate mismatch')
        except Exception as exc:
            errors.append(f'Invalid result file: {exc}')
    return {'task': task, 'successes': successes, 'episodes': len(videos), 'seconds': round(elapsed, 1),
            'exit_code': code, 'validated': not errors, 'errors': errors, 'videos': [str(p) for p in videos]}


def worker(slot, tasks):
    wait_ready(slot)
    print(f'Slot {slot} ready: GPU {GPUS[slot]}, port {PORTS[slot]}', flush=True)
    while not STOP.is_set():
        try:
            task = tasks.get_nowait()
        except queue.Empty:
            return
        if SERVERS[slot].poll() is not None:
            raise RuntimeError(f'Inference slot {slot} died')
        begin = time.time()
        print(f'START {task}: GPU {GPUS[slot]}', flush=True)
        cmd = [str(ENV_ROOT / 'RoboTwin/bin/python'), '-u', str(HERE / 'eval_with_video.py'),
               '--config', 'policy/ACT/deploy_policy.yml', '--overrides', '--task_name', task,
               '--task_config', 'demo_clean', '--train_config_name', '0', '--seed', '0',
               '--policy_name', 'ACT', '--port', str(PORTS[slot]), '--robo_name', 'robotwin',
               '--video_fps', '10', '--eval_video_log', 'True', '--output_dir', str(HERE / 'eval_results')]
        proc = spawn(cmd, SIM, environment(GPUS[slot], 'RoboTwin'), HERE / 'eval_logs' / f'{task}.log')
        code = proc.wait()
        result = validate_task(task, code, time.time() - begin)
        with LOCK:
            RESULTS[task] = result
            save_json('progress.json', {'completed_tasks': len(RESULTS), 'total_tasks': 10, 'results': RESULTS})
        print(f'DONE {task}: {result["successes"]}/10, videos={result["episodes"]}, validated={result["validated"]}', flush=True)


def main():
    for directory in ['inference_logs', 'eval_logs', 'eval_results']:
        (HERE / directory).mkdir(exist_ok=True)
    save_json('run_config.json', {'checkpoint': str(TRAIN / 'checkpoints/global_step_5000/hf_ckpt'),
        'normalization': str(TRAIN / 'inference_assets/norm_stats.json'), 'tasks': TASKS,
        'episodes_per_task': 10, 'task_config': 'demo_clean', 'instruction_type': 'unseen', 'seed': 0,
        'precision': 'fp32', 'compile': True, 'chunk_length': 50, 'video_fps': 10, 'gpus': GPUS, 'ports': PORTS})
    tasks = queue.Queue()
    for task in TASKS:
        tasks.put(task)
    for slot, gpu in enumerate(GPUS):
        SERVERS[slot] = spawn([str(ENV_ROOT / 'lerobot/bin/python'), '-u', str(HERE / 'policy_server.py'),
                               '--port', str(PORTS[slot])], ROOT / 'lingbot-vla-v2', environment(gpu, 'lerobot'),
                              HERE / 'inference_logs' / f'slot{slot}.log')
    error = None
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    try:
        futures = [executor.submit(worker, slot, tasks) for slot in range(4)]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        print(error, flush=True)
    finally:
        cleanup()
        executor.shutdown(wait=True, cancel_futures=True)
    valid = len(RESULTS) == 10 and all(r['validated'] for r in RESULTS.values()) and error is None
    total = sum(r['successes'] for r in RESULTS.values() if r['validated'])
    report = {'complete': valid, 'successes': total, 'expected_episodes': 100,
              'success_rate': total / 100 if valid else None, 'seconds': round(time.time() - START, 1),
              'error': error, 'results': RESULTS}
    save_json('results.json', report)
    lines = [f'Checkpoint: global_step_5000', 'Scene: demo_clean; instruction_type: unseen; seed: 0',
             'Precision: FP32; compile: True; chunk: 50; video: 10 fps', f'Complete and validated: {valid}']
    for task in TASKS:
        r = RESULTS.get(task)
        lines.append(f'{task}: {r["successes"]}/10, videos={r["episodes"]}, validated={r["validated"]}' if r else f'{task}: incomplete')
    lines.append(f'Total: {total}/100 ({total}%)' if valid else 'Total: incomplete; see results.json for errors')
    lines.append(f'Elapsed: {report["seconds"]} seconds')
    (HERE / 'stats.txt').write_text('\n'.join(lines) + '\n')
    (HERE / 'exit_code').write_text('0\n' if valid else '1\n')
    print('\n'.join(lines), flush=True)
    return 0 if valid else 1


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    raise SystemExit(main())
