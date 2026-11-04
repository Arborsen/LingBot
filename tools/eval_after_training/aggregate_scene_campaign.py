"""Validate and merge corrected retests without rewriting historical results."""
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys

SETTINGS={'clean2clean':('demo_clean','seen'),'clean2random':('demo_randomized','unseen')}


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def inference_seed(scene,task,seed):
    return int.from_bytes(hashlib.sha256(f'42|{scene}|{task}|{seed}'.encode()).digest()[:8],'big')%(2**31)


def load_result(result,setting,task,repaired=False):
    assert result['validated'] and result['episodes']==50,(setting,task,'not validated')
    directory=Path(result['source'])/'eval_results'/task
    path=directory/'episodes.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows)==50 and [r['episode'] for r in rows]==list(range(50)),(task,'incomplete episodes')
    assert len({r['seed'] for r in rows})==50,(task,'duplicate seed')
    scene,language=SETTINGS[setting];successes=0
    for row in rows:
        assert row['task']==task and row['task_config']==scene and row['instruction_type']==language
        assert row['inference_seed']==inference_seed(scene,task,row['seed']),(task,'inference seed')
        assert row['video_recording'] is False and type(row['success']) is bool
        if repaired:assert row['scene_repair_version']==1 and row['initial_state_verified'] is True
        successes+=row['success'];assert row['cumulative_successes']==successes
    assert successes==result['successes']
    text=(directory/'_result.txt').read_text()
    assert f'Instruction Type: {language}' in text
    assert math.isclose(float(text.splitlines()[-1]),successes/50,abs_tol=1e-10)
    assert not list(Path(result['source']).rglob('*.mp4'))
    return rows,dict(source=str(result['source']),ledger_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        result_sha256=hashlib.sha256((directory/'_result.txt').read_bytes()).hexdigest(),repaired=repaired)


def paired_summary(differences,task_differences):
    n=len(differences);mean=sum(differences)/n
    se=math.sqrt(sum((d-mean)**2 for d in differences)/(n-1)/n)
    rng=random.Random(42);bootstrap=sorted(sum(rng.choice(task_differences) for _ in task_differences)/len(task_differences)
        for _ in range(10000))
    return dict(paired_episodes=n,step22000_minus25000_percentage_points=100*mean,
        paired_episode_normal_95ci_percentage_points=[100*(mean-1.96*se),100*(mean+1.96*se)],
        task_bootstrap_95ci_percentage_points=[100*bootstrap[250],100*bootstrap[9749]],
        step22000_only_success=sum(d==1 for d in differences),step25000_only_success=sum(d==-1 for d in differences))


def aggregate(campaign):
    campaign=Path(campaign);cfg=read(campaign/'campaign.json');out=campaign/'final'
    assert not out.exists(),'Never overwrite a final report'
    base25=read(Path(cfg['baseline25000'])/'status.json')
    remaining=read(Path(cfg['remaining22000'])/'status.json')
    repaired={s:read(Path(cfg[f'repair{s}'])/'status.json') for s in [22000,25000]}
    assert base25['phase']=='complete'
    assert remaining['phase'] in ['complete','needs_scene_repair']
    assert all(s['phase']=='complete' for s in repaired.values())
    tasks=read(Path(cfg['baseline25000'])/'plan.json')['tasks'];assert len(tasks)==len(set(tasks))==50
    replacements=set(cfg['repaired_tasks']);assert replacements=={'dump_bin_bigbin','put_bottles_dustbin'}
    merged={22000:{},25000:{}};ledgers={};provenance=[];table=[]
    for step,base in [(22000,remaining),(25000,base25)]:
        for setting in SETTINGS:
            selected={r['task']:dict(r) for r in base['settings'][setting]['results'] if r['validated']}
            if setting=='clean2random':
                corrected={r['task']:dict(r) for r in repaired[step]['settings'][setting]['results'] if r['validated']}
                assert set(corrected)==replacements
                selected.update(corrected)
            assert set(selected)==set(tasks),(step,setting,'Missing tasks',set(tasks)-selected.keys())
            for task,result in selected.items():
                is_repaired=setting=='clean2random' and task in replacements
                rows,audit=load_result(result,setting,task,is_repaired)
                ledgers[(step,setting,task)]=rows;provenance.append(dict(checkpoint=step,setting=setting,task=task,**audit))
            successes=sum(r['successes'] for r in selected.values())
            merged[step][setting]=dict(setting=setting,task_config=SETTINGS[setting][0],instruction_type=SETTINGS[setting][1],
                successes=successes,episodes=2500,success_rate=successes/2500,validated=True,video_recording=False,
                corrected_tasks=sorted(replacements) if setting=='clean2random' else [],tasks=[selected[t] for t in tasks])
    comparison={};all_differences=[];paired_by_task={t:[] for t in tasks}
    for setting in SETTINGS:
        differences=[];task_diffs=[]
        for task in tasks:
            a,b=ledgers[(22000,setting,task)],ledgers[(25000,setting,task)];ds=[]
            for row22,row25 in zip(a,b):
                assert all(row22[k]==row25[k] for k in ['task','task_config','instruction_type','episode','seed','instruction','inference_seed']),('Pairing mismatch',setting,task)
                ds.append(int(row22['success'])-int(row25['success']))
            differences.extend(ds);all_differences.extend(ds);paired_by_task[task].extend(ds)
            task_diffs.append(sum(ds)/50)
            score22=sum(r['success'] for r in a);score25=sum(r['success'] for r in b)
            table.append(dict(task=task,setting=setting,step22000_successes=score22,step25000_successes=score25,
                episodes_each=50,step22000_rate=score22/50,step25000_rate=score25/50,
                step22000_minus25000_percentage_points=2*(score22-score25),retested=setting=='clean2random' and task in replacements))
        comparison[setting]=dict(step22000_rate=merged[22000][setting]['success_rate'],
            step25000_rate=merged[25000][setting]['success_rate'],
            tasks_22000_better=sum(d>0 for d in task_diffs),tasks_25000_better=sum(d<0 for d in task_diffs),
            tasks_tied=sum(d==0 for d in task_diffs),**paired_summary(differences,task_diffs))
    # Repaired references must include exact full states, not only nominal seeds.
    for task in replacements:
        manifests={step:read(Path(cfg[f'repair{step}'])/'paired_episodes/demo_randomized'/f'{task}.json') for step in repaired}
        assert manifests[22000]==manifests[25000] and len(manifests[22000])==50
        assert all(r.get('replay_state',{}).get('expert_initial_state') is True for r in manifests[22000])
    equal={step:sum(v['successes'] for v in merged[step].values())/5000 for step in merged}
    comparison['equal_weight']=dict(step22000_rate=equal[22000],step25000_rate=equal[25000],
        **paired_summary(all_differences,[sum(paired_by_task[t])/100 for t in tasks]))
    report=dict(validated=True,episodes_each_checkpoint=5000,video_recording=False,gpus=[1,2],
        comparison=comparison,per_task=table,provenance=provenance,source_config=cfg,
        note='Historical results are retained; two randomized tasks are replaced by corrected paired retests only.')
    # Create the final directory only after every source and pair has passed.
    out.mkdir()
    for step,settings in merged.items():
        for setting,summary in settings.items():save(out/f'step{step}'/setting/'summary.json',summary)
    save(out/'comparison.json',report)
    with (out/'task_success_rates.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    lines=['# step22000 与 step25000 修复后评测对比','',
        '50个任务，每任务每设置50回合；clean+seen、randomized+unseen。仅GPU1、2，关闭录像。',
        '原结果均保留；randomized 的 dump_bin_bigbin、put_bottles_dustbin 使用修复后的各50回合重测结果替换。','',
        '| 设置 | step22000 | step25000 | 22k−25k（百分点） |','|---|---:|---:|---:|']
    for name,label in [('clean2clean','clean+seen'),('clean2random','randomized+unseen'),('equal_weight','两设置等权平均')]:
        c=comparison[name];lines.append(f'| {label} | {100*c["step22000_rate"]:.2f}% | {100*c["step25000_rate"]:.2f}% | {c["step22000_minus25000_percentage_points"]:+.2f} |')
    lines.extend(['','所有10000条回合记录已校验；5000组种子、完整指令及推理种子逐项配对一致。',
        '两项重测还核验了初始物体姿态与速度、机器人关节状态、驱动目标、场景配置及随机状态。','',
        '| 设置 | 22k更好任务数 | 25k更好任务数 | 持平 | 配对差值95%区间（百分点） |',
        '|---|---:|---:|---:|---|'])
    for name in SETTINGS:
        c=comparison[name];low,high=c['paired_episode_normal_95ci_percentage_points']
        lines.append(f'| {name} | {c["tasks_22000_better"]} | {c["tasks_25000_better"]} | {c["tasks_tied"]} | [{low:+.2f}, {high:+.2f}] |')
    lo,hi=comparison['equal_weight']['task_bootstrap_95ci_percentage_points']
    best=22000 if equal[22000]>equal[25000] else 25000
    lines.extend(['',f'等权点估计较高的是 step{best}。按任务重采样的等权差值95%区间为 [{lo:+.2f}, {hi:+.2f}] 个百分点；',
        '若区间覆盖0，当前样本尚不能确认稳定优势，应结合弱任务和泛化表现选权重。','',
        '逐任务结果见 task_success_rates.csv；来源、哈希、配对校验及两种不确定性估计见 comparison.json。'])
    (out/'对比汇总.md').write_text('\n'.join(lines)+'\n')
    save(out/'validation.json',dict(passed=True,validated_episode_records=10000,paired_episodes=5000,
        corrected_retest_episodes=200,video_recording=False))
    return out


if __name__=='__main__':print(aggregate(sys.argv[1]))
