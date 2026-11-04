"""Frozen episode manifests, isolated instruction RNG, trajectory and video audits."""
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess

import numpy as np


def seed_setup(environment_seed):
    # Base_Task seeds NumPy/Torch itself but historically omitted Python random.
    random.seed(int(environment_seed))


def manifest_path(task_config,task):
    return Path(os.environ['ROBOTWIN_PAIR_MANIFEST'])/task_config/(task+'.json')


def read_manifest(task_config,task):
    p=manifest_path(task_config,task)
    return json.loads(p.read_text()) if p.exists() else []


def planned_episode(task_config,task,index):
    rows=read_manifest(task_config,task)
    return rows[index] if index<len(rows) else None


def freeze_instruction(task_config,task,index,environment_seed,episode_info,generator,kind):
    rows=read_manifest(task_config,task)
    seed=int.from_bytes(hashlib.sha256(f'42|language|{task_config}|{task}|{environment_seed}'.encode()).digest()[:4],'big')
    if index>=len(rows):
        if os.environ.get('ROBOTWIN_MANIFEST_MODE')!='record':
            raise RuntimeError('Replay is missing an episode; never replace it with a new seed')
        if index!=len(rows):raise RuntimeError('Manifest episode gap')
        py,npstate=random.getstate(),np.random.get_state()
        try:
            random.seed(seed);np.random.seed(seed)
            generated=generator(task,[episode_info],100)[0]
            instructions={k:generated[k][np.random.randint(len(generated[k]))] for k in ['seen','unseen']}
        finally:
            random.setstate(py);np.random.set_state(npstate)
        row=dict(episode=index,seed=environment_seed,instructions=instructions,language_seed=seed,
                 expert_info=episode_info)
        rows.append(row)
        p=manifest_path(task_config,task);p.parent.mkdir(parents=True,exist_ok=True)
        temp=p.with_suffix('.tmp');temp.write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n');temp.replace(p)
    row=rows[index]
    if row['seed']!=environment_seed:raise RuntimeError('Frozen environment seed mismatch')
    return row['instructions'][kind]


def freeze_initial_state(task_config,task,index,env):
    # Save/check realized initial scene, beyond merely checking nominal seeds.
    signature={}
    for i,actor in enumerate(env.scene.get_all_actors()):
        pose=actor.get_pose()
        signature[f'{i}:{actor.get_name()}']=np.r_[pose.p,pose.q].tolist()
    rows=read_manifest(task_config,task);row=rows[index]
    if 'initial_actors' not in row:
        if os.environ.get('ROBOTWIN_MANIFEST_MODE')!='record':raise RuntimeError('Missing initial scene reference')
        row['initial_actors']=signature
        p=manifest_path(task_config,task);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
    else:
        old=row['initial_actors']
        if old.keys()!=signature.keys() or any(not np.allclose(old[k],v,atol=2e-4,rtol=0) for k,v in signature.items()):
            raise RuntimeError('Realized initial scene differs from frozen reference')


def validate_videos(directory,rows):
    output=[]
    videos=sorted(Path(directory).glob('episode*.mp4'))
    if len(videos)!=len(rows):raise RuntimeError(f'Expected {len(rows)} videos; got {len(videos)}')
    for row in rows:
        p=Path(directory)/f"episode{row['episode']}_{'success' if row['success'] else 'failure'}.mp4"
        if not p.is_file() or not p.stat().st_size:raise RuntimeError(f'Missing video: {p}')
        probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
                '-show_entries','stream=width,height,nb_read_frames','-of','json',str(p)],capture_output=True,text=True,timeout=90,check=True)
        stream=json.loads(probe.stdout)['streams'][0]
        if int(stream['nb_read_frames'])<=0:raise RuntimeError('Empty video stream')
        subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(p),'-f','null','-'],
                       capture_output=True,text=True,timeout=120,check=True)
        # A compact contact sheet includes both approach and late failure/success.
        sheet=p.with_suffix('.jpg')
        subprocess.run(['ffmpeg','-v','error','-i',str(p),'-vf',
                        f"select='not(mod(n,{max(1,int(stream['nb_read_frames'])//12)}))',tpad=stop_mode=clone:stop=12,scale=320:-1,tile=4x3",
                        '-frames:v','1',str(sheet)],capture_output=True,text=True,timeout=90,check=True)
        output.append(dict(video=str(p),bytes=p.stat().st_size,frames=int(stream['nb_read_frames']),
                           decoded=True,contact_sheet=str(sheet)))
    return output
