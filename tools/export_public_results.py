#!/usr/bin/env python3
"""Export verified task counts and hashes, without publishing raw rollout data."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from eval_after_training.aggregate_scene_campaign import load_result

ROOT=Path(__file__).resolve().parents[1]
SETTINGS=['clean2clean','clean2random']
REPAIRED={'dump_bin_bigbin','put_bottles_dustbin'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def relative(path,root):
    try:return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:raise ValueError(f'Evidence must belong to the project: {Path(path).name}')


def export(root,output):
    root=Path(root);output=Path(output)
    policy=read(root/'reproduce_artifacts/evaluation_gpu_policy.json')
    base25=Path(policy['first']);base22=Path(policy['second']);campaign=Path(policy['scene_repair_campaign'])
    repaired25=campaign/'step25000_retest';repaired22=campaign/'step22000_retest'
    sources={22000:(base22,read(base22/'status.json')),25000:(base25,read(base25/'status.json'))}
    repair=read(repaired25/'status.json');assert repair['phase']=='complete'
    plan=read(base25/'plan.json');tasks=plan['tasks'];assert len(set(tasks))==len(tasks)==50
    assert plan['precision']=='fp32' and plan['chunk_length']==50 and not plan['dynamic_execution']
    summaries={};evidence=[];counts={}
    for step,(run,status) in sources.items():
        summaries[str(step)]={}
        for setting in SETTINGS:
            selected={r['task']:r for r in status['settings'][setting]['results'] if r['validated']}
            if step==25000 and setting=='clean2random':
                replacements={r['task']:r for r in repair['settings'][setting]['results'] if r['validated']}
                assert set(replacements)==REPAIRED
                selected.update(replacements)
            assert set(selected)<=set(tasks)
            for task,result in selected.items():
                replaced=step==25000 and setting=='clean2random' and task in REPAIRED
                rows,audit=load_result(result,setting,task,repaired=replaced)
                evidence.append(dict(checkpoint_step=step,setting=setting,task=task,
                    source=relative(audit.pop('source'),root),**audit))
                counts[step,setting,task]=sum(r['success'] for r in rows)
            successes=sum(r['successes'] for r in selected.values());episodes=50*len(selected)
            summaries[str(step)][setting]=dict(status='complete' if len(selected)==50 else 'partial',
                validated_tasks=len(selected),expected_tasks=50,validated_episodes=episodes,
                expected_episodes=2500,successes=successes,success_rate=successes/episodes if episodes else None,
                missing_tasks=sorted(set(tasks)-selected.keys()),
                corrected_tasks=sorted(REPAIRED) if step==25000 and setting=='clean2random' else [])
    rows=[]
    for setting in SETTINGS:
        for task in sorted(tasks):
            a=counts.get((22000,setting,task));b=counts.get((25000,setting,task))
            rows.append(dict(task=task,setting=setting,episodes_per_task=50,
                step22000_successes=a,step25000_successes=b,
                step22000_rate=a/50 if a is not None else None,
                step25000_rate=b/50 if b is not None else None,
                step22000_minus25000_pp=2*(a-b) if a is not None and b is not None else None))
    common={}
    for setting in SETTINGS:
        matched=[r for r in rows if r['setting']==setting and r['step22000_successes'] is not None and r['step25000_successes'] is not None]
        common[setting]=dict(tasks=len(matched),episodes_per_checkpoint=50*len(matched),
            step22000_successes=sum(r['step22000_successes'] for r in matched),
            step25000_successes=sum(r['step25000_successes'] for r in matched))
    training=Path(plan['source_training_validation']);training_report=read(training)
    report=dict(captured_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        checkpoints=summaries,matched_task_counts=common,
        protocol=dict(tasks=50,episodes_per_task=50,precision='fp32',prediction_horizon=50,
            execution_horizon=50,flow_matching_iterations=10,rtc_enabled=False,video_recording=False,
            clean='demo_clean / seen',randomized='demo_randomized / unseen',training_data='official clean only'),
        training_validation=dict(final_step=training_report['final_step'],full_restore_verified=training_report['full_restore_verified'],
            final_learning_rate=training_report['lr'],world_size=2,global_batch=160,
            mean_step_seconds=training_report['mean_step_seconds'],peak_reserved_mib=training_report['peak_reserved_mib']),
        repair_status=dict(step25000=repair['phase'],step22000=read(repaired22/'status.json')['phase'],
            corrected_randomized_tasks=sorted(REPAIRED),completed_step22000_randomized_comparison=False),
        limitations=['Step22000 randomized is partial; no complete equal-weight comparison is claimed.',
            'Step25000 randomized replaces only the two affected tasks with corrected50-episode retests.',
            'These fixed50 scores do not measure RTC. Raw rollouts and model checkpoints remain local.'])
    provenance=dict(evaluations=evidence,sources=[dict(path=relative(path,root),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for path in [base22/'status.json',base25/'status.json',repaired25/'status.json',repaired22/'status.json',
            campaign/'native_source_fix.json',training]])
    output.mkdir(parents=True,exist_ok=True)
    for name,value in [('summary.json',report),('provenance.json',provenance)]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (output/'per_task.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/benchmarks/step22000_step25000')
    args=parser.parse_args();export(args.root,args.output)
