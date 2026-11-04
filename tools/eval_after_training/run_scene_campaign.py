"""Durable queue: remaining22k -> repaired25k -> repaired22k -> comparison."""
from datetime import datetime
import ast
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

HERE=Path(sys.argv[1]).resolve();CONFIG=json.loads((HERE/'campaign.json').read_text())
OWNED=[]


def read(path):return json.loads(Path(path).read_text()) if Path(path).exists() else None


def status(phase,**extra):
    value=dict(phase=phase,controller_pid=os.getpid(),updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        gpus=[1,2],video_recording=False,**extra)
    temporary=HERE/'status.json.tmp';temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(HERE/'status.json')


def start(run):
    run=Path(run);env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1,2',PYTHONUTF8='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
    for key in list(env):
        if key.lower().endswith('_proxy'):env.pop(key)
    plan=read(run/'plan.json')
    assert plan['gpus']==[1,2] and all(s['gpu'] in [1,2] for s in plan['slots'])
    assert plan['eval_video_log'] is False and plan['episodes_per_task']==50
    with (run/'scheduler.log').open('x') as stream:
        child=subprocess.Popen([sys.executable,'-u',str(run/'run_scheduled.py')],
            cwd=CONFIG.get('repo_root',str(Path(CONFIG['remaining22000']).parents[1])),env=env,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    OWNED.append(child)
    with (run/'profiler.log').open('x') as stream:
        profiler=subprocess.Popen([sys.executable,'-u',str(run/'profile_eval.py'),'--run-root',str(run),'--watch'],
            env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    OWNED.append(profiler)
    (run/'campaign_launch.json').write_text(json.dumps(dict(controller_pid=child.pid,profiler_pid=profiler.pid),indent=2)+'\n')
    return child


def apply_native_fix():
    """Update shared source only after the existing frozen-source run has exited."""
    plan=read(Path(CONFIG['repair25000'])/'plan.json');source=Path(plan['sim'])/'envs/_base_task.py'
    original=source.read_text();before=hashlib.sha256(source.read_bytes()).hexdigest()
    signature='    def get_cluttered_table(self, cluttered_numbers=10, xlim=[-0.59, 0.59], ylim=[-0.34, 0.34], zlim=[0.741]):\n'
    assert original.count(signature)==1,'Simulator source differs from reviewed patch'
    replacement='''    def get_cluttered_table(self, cluttered_numbers=10, xlim=None, ylim=None, zlim=None):
        # Each scene owns its bounds; nonzero table offsets must not accumulate.
        xlim = list([-0.59, 0.59] if xlim is None else xlim)
        ylim = list([-0.34, 0.34] if ylim is None else ylim)
        zlim = list([0.741] if zlim is None else zlim)
'''
    patched=original.replace(signature,replacement);ast.parse(patched)
    manifests=[]
    for step in [22000,25000]:
        path=Path(CONFIG[f'repair{step}'])/'source_hashes.json';manifest=read(path)
        assert manifest[str(source)]==before,'Unexpected simulator source mutation'
        manifests.append((path,manifest))
    with (HERE/'simulator_source_before.py').open('x') as stream:stream.write(original)
    with (HERE/'simulator_source_after.py').open('x') as stream:stream.write(patched)
    temporary=source.with_name('_base_task.scene_repair.tmp')
    with temporary.open('x') as stream:stream.write(patched)
    temporary.chmod(source.stat().st_mode & 0o777);temporary.replace(source)
    after=hashlib.sha256(source.read_bytes()).hexdigest()
    for path,manifest in manifests:
        with path.with_name('source_hashes_before_native_fix.json').open('x') as stream:stream.write(json.dumps(manifest,indent=2)+'\n')
        manifest[str(source)]=after;path.write_text(json.dumps(manifest,indent=2)+'\n')
    (HERE/'native_source_fix.json').write_text(json.dumps(dict(source=str(source),sha256_before=before,
        sha256_after=after,applied_after_remaining22000=True,old_result_manifests_unchanged=True),indent=2)+'\n')


def main():
    with (HERE/'controller.pid').open('x') as stream:stream.write(str(os.getpid())+'\n')
    preflight=read(HERE/'scene_preflight.json');assert preflight and preflight['passed'],'Scene preflight must pass before queuing retests'
    cross=read(HERE/'cross_gpu_preflight.json');assert cross and cross['passed'],'Cross-process scene replay must pass'
    remaining=Path(CONFIG['remaining22000'])
    while True:
        s=read(remaining/'status.json')
        if s and s['phase'] in ['complete','needs_scene_repair']:
            expected=set(CONFIG['repaired_tasks'])
            clean=s['settings']['clean2clean'];randomized=s['settings']['clean2random']
            assert clean['validated_tasks']==50 and randomized['validated_tasks']==48,'Unexpected missing results in remaining22k run'
            actual={r['task'] for r in randomized['results'] if r['validated']}
            tasks=set(read(Path(CONFIG['baseline25000'])/'plan.json')['tasks'])
            assert actual==tasks-expected
            break
        if s and s['phase']=='failed':raise RuntimeError('Remaining22k evaluation failed: '+str(s.get('error')))
        status('waiting_for_remaining22000',remaining_run=str(remaining),
            progress={k:{a:v.get(a) for a in ['validated_tasks','completed_episodes','successes_so_far']} for k,v in (s or {}).get('settings',{}).items()})
        time.sleep(30)
    # The old scheduler has already cleaned up its model/simulator processes
    # before publishing its terminal phase. Its original source manifest stays intact.
    apply_native_fix()
    for step in [25000,22000]:
        run=Path(CONFIG[f'repair{step}']);status(f'retesting_step{step}',active_run=str(run))
        child=start(run)
        while child.poll() is None:
            detail=read(run/'status.json');status(f'retesting_step{step}',active_run=str(run),evaluation_status=detail)
            time.sleep(30)
        assert child.returncode==0 and read(run/'status.json')['phase']=='complete',f'Repaired step{step} failed; see {run}'
    status('validating_and_aggregating')
    from aggregate_scene_campaign import aggregate
    final=aggregate(HERE)
    status('complete',final_directory=str(final),report=str(final/'对比汇总.md'))
    (HERE/'exit_code.txt').write_text('0\n')


def stop(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')


if __name__=='__main__':
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:main()
    except BaseException as exc:
        for child in OWNED:
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        status('failed',error=f'{type(exc).__name__}: {exc}')
        (HERE/'exit_code.txt').write_text('1\n')
        raise
