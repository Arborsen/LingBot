"""Fail-closed training completion and immutable per-task pairing barriers."""
import hashlib
import json
from pathlib import Path


def read(path):
    p=Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def training_ready(plan):
    status=read(plan['source_training_status'])
    if status is None:return False,dict(reason='training_status_not_created')
    if status.get('failed') or status.get('phase')=='failed':
        raise RuntimeError('Training failed; evaluation will not start: '+str(status.get('error')))
    exit_path=Path(plan['source_training_exit_code'])
    if exit_path.exists() and exit_path.read_text().strip()!='0':
        raise RuntimeError('Training controller exited unsuccessfully')
    if status.get('phase')!='complete':
        return False,dict(reason='training_in_progress',last_step=status.get('last_step'))
    final_step=plan.get('training_final_step',plan['checkpoint_step'])
    if status.get('final_step')!=final_step:raise RuntimeError('Wrong training final checkpoint')
    if not exit_path.exists():return False,dict(reason='waiting_for_training_exit_marker')
    validation=read(plan['source_training_validation'])
    if validation is None:return False,dict(reason='waiting_for_training_validation')
    if not validation.get('passed') or validation.get('final_step')!=final_step:
        raise RuntimeError('Training final checkpoint validation failed')
    if plan.get('preceding_evaluation_status'):
        previous=read(plan['preceding_evaluation_status'])
        marker=Path(plan['preceding_evaluation_exit_code'])
        if previous and (previous.get('phase')=='failed' or previous.get('failed')):
            raise RuntimeError('Preceding evaluation failed; queued evaluation will not start')
        if marker.exists() and marker.read_text().strip()!='0':
            raise RuntimeError('Preceding evaluation exited unsuccessfully')
        if not previous or previous.get('phase')!='complete' or not marker.exists():
            return False,dict(reason='waiting_for_preceding_evaluation',preceding_status=plan['preceding_evaluation_status'])
    cp=Path(plan['checkpoint']).parent
    required=['model/.metadata','optimizer/.metadata']+[
        f'extra_state/extra_state_rank_{r}.pt' for r in range(plan['checkpoint_world_size'])]
    missing=[name for name in required if not (cp/name).is_file()]
    index=read(cp/'hf_ckpt/model.safetensors.index.json')
    if index is None:missing.append('HF weight index')
    else:
        files=set(index['weight_map'].values())
        if len(files)!=6:raise RuntimeError('Expected all6 HF weight shards')
        missing.extend(name for name in files if not (cp/'hf_ckpt'/name).is_file())
    if missing:return False,dict(reason='checkpoint_not_complete',missing=missing)
    return True,dict(checkpoint_step=plan['checkpoint_step'],checkpoint=str(cp),training_complete=True,
        full_training_validation_passed=True,checkpoint_shards_complete=True)


def manifest_rows(plan,setting,task):
    path=Path(plan['manifest_source_directory'])/setting['task_config']/(task+'.json')
    rows=read(path);n=plan['episodes_per_task']
    if rows is None or len(rows)<n:return None
    rows=rows[:n]
    if any('initial_actors' not in r or not r['initial_actors'] for r in rows):return None
    for i,row in enumerate(rows):
        if row.get('episode')!=i or type(row.get('seed')) is not int:
            raise RuntimeError(f'Invalid producer episode ordering: {path}')
        if 'expert_info' not in row or not all(isinstance(row.get('instructions',{}).get(k),str) for k in ['seen','unseen']):
            raise RuntimeError(f'Invalid frozen instruction/expert info: {path}')
    if len({r['seed'] for r in rows})!=n:raise RuntimeError(f'Duplicate producer seeds: {path}')
    return rows


def freeze_manifest(plan,setting,task,output):
    rows=manifest_rows(plan,setting,task)
    if rows is None:return False
    dest=Path(plan['manifest_directory'])/setting['task_config']/(task+'.json')
    existing=read(dest)
    if existing is not None and existing!=rows:
        raise RuntimeError(f'Paired manifest changed; will not overwrite: {dest}')
    body=json.dumps(rows,ensure_ascii=False,indent=2)+'\n'
    if existing is None:
        dest.parent.mkdir(parents=True,exist_ok=True)
        temporary=dest.with_suffix('.json.tmp');temporary.write_text(body);temporary.replace(dest)
    audit=Path(output)/'manifest_audits'/setting['task_config']/(task+'.json');audit.parent.mkdir(parents=True,exist_ok=True)
    detail=dict(task=task,task_config=setting['task_config'],episodes=len(rows),
        source=str(Path(plan['manifest_source_directory'])/setting['task_config']/(task+'.json')),
        destination=str(dest),sha256=hashlib.sha256(body.encode()).hexdigest(),paired_episode_conditions_verified=True)
    audit.write_text(json.dumps(detail,indent=2)+'\n')
    return True


def take_ready_task(tasks,plan,setting,output):
    """Caller holds scheduler lock: skip unready tasks without dropping them."""
    if tasks.empty():return None,True
    if plan.get('manifest_mode')=='record':return tasks.get_nowait(),False
    for _ in range(tasks.qsize()):
        task=tasks.get_nowait()
        if freeze_manifest(plan,setting,task,output):return task,False
        tasks.put(task)
    source=read(plan['manifest_source_status'])
    if source and (source.get('phase') in ['failed','complete'] or source.get('failed')):
        raise RuntimeError('Seed producer stopped before all paired50-episode manifests were complete')
    return None,False
