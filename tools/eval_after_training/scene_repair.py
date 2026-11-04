"""Fresh clutter bounds and certified initial-state replay for the two offset tasks.

Installed only inside new repair clients. Existing simulator files and running
evaluations remain untouched. A state is captured before the expert acts; only
expert-valid seeds reach the policy and its immutable paired manifest.
"""
import json
import os
from pathlib import Path
import random

TASKS={'dump_bin_bigbin','put_bottles_dustbin'}


def fresh_clutter_call(original):
    def call(self,cluttered_numbers=10,xlim=None,ylim=None,zlim=None):
        return original(self,cluttered_numbers=cluttered_numbers,
            xlim=list([-0.59,0.59] if xlim is None else xlim),
            ylim=list([-0.34,0.34] if ylim is None else ylim),
            zlim=list([0.741] if zlim is None else zlim))
    return call


def plain(value):
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if hasattr(value,'tolist'):return value.tolist()
    return value


def pose_values(pose):return plain(list(pose.p)+list(pose.q))


def capture(env):
    import numpy as np
    import sapien
    import torch
    actors=[]
    for i,actor in enumerate(env.scene.get_all_actors()):
        body=actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        item=dict(key=f'{i}:{actor.get_name()}',pose=pose_values(actor.get_pose()),dynamic=body is not None)
        if body is not None:
            item.update(linear=plain(body.linear_velocity),angular=plain(body.angular_velocity),
                sleeping=body.is_sleeping,mass=body.mass,inertia=plain(body.inertia))
        actors.append(item)
    arts=[]
    for i,art in enumerate(env.scene.get_all_articulations()):
        joints=art.get_active_joints()
        arts.append(dict(key=f'{i}:{art.get_name()}',root_pose=pose_values(art.get_root_pose()),
            qpos=plain(art.get_qpos()),qvel=plain(art.get_qvel()),qf=plain(art.get_qf()),qacc=plain(art.get_qacc()),
            root_linear=plain(art.get_root_linear_velocity()),root_angular=plain(art.get_root_angular_velocity()),
            joints=[dict(name=j.get_name(),target=plain(j.get_drive_target()),
                velocity=plain(j.get_drive_velocity_target())) for j in joints]))
    npstate=np.random.get_state()
    return plain(dict(schema=1,expert_initial_state=True,actors=actors,articulations=arts,
        configuration=dict(task=env.task_name,table_z_bias=env.table_z_bias,
            table_texture=env.table_texture,wall_texture=env.wall_texture,
            clutter=env.record_cluttered_objects,deskbin_id=getattr(env,'deskbin_id',None)),
        rng=dict(python=random.getstate(),numpy=[npstate[0],npstate[1],npstate[2],npstate[3],npstate[4]],
            torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None)))


def restore(env,state):
    import numpy as np
    import sapien
    import torch
    assert state['schema']==1 and state['expert_initial_state']
    before=capture(env)
    assert before['configuration']==state['configuration'],'Recreated scene configuration differs from certified reference'
    actors=env.scene.get_all_actors();arts=env.scene.get_all_articulations()
    assert [r['key'] for r in before['actors']]==[r['key'] for r in state['actors']],'Actor identity mismatch'
    assert [r['key'] for r in before['articulations']]==[r['key'] for r in state['articulations']],'Articulation identity mismatch'
    for actor,old,current in zip(actors,state['actors'],before['actors']):
        assert old['dynamic']==current['dynamic']
        actor.set_pose(sapien.Pose(old['pose'][:3],old['pose'][3:]))
        if old['dynamic']:
            assert np.isclose(old['mass'],current['mass']) and np.allclose(old['inertia'],current['inertia'])
            body=actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            body.set_linear_velocity(old['linear']);body.set_angular_velocity(old['angular'])
            if old['sleeping']:body.put_to_sleep()
    for art,old in zip(arts,state['articulations']):
        art.set_root_pose(sapien.Pose(old['root_pose'][:3],old['root_pose'][3:]))
        art.set_qpos(old['qpos']);art.set_qvel(old['qvel']);art.set_qf(old['qf']);art.set_qacc(old['qacc'])
        art.set_root_linear_velocity(old['root_linear']);art.set_root_angular_velocity(old['root_angular'])
        joints=art.get_active_joints()
        assert [j.get_name() for j in joints]==[j['name'] for j in old['joints']]
        for joint,saved in zip(joints,old['joints']):
            joint.set_drive_target(saved['target']);joint.set_drive_velocity_target(saved['velocity'])
    def tuples(value):return tuple(tuples(v) for v in value) if isinstance(value,list) else value
    rng=state['rng'];random.setstate(tuples(rng['python']))
    n=rng['numpy'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(torch.tensor(rng['torch_cpu'],dtype=torch.uint8))
    if rng['torch_cuda'] is not None:torch.cuda.set_rng_state(torch.tensor(rng['torch_cuda'],dtype=torch.uint8))
    after=capture(env)
    for old,new in zip(state['actors'],after['actors']):
        assert np.allclose(old['pose'],new['pose'],atol=1e-6,rtol=0),'Actor pose restore failed'
        if old['dynamic']:
            assert np.allclose(old['linear'],new['linear'],atol=1e-6,rtol=0)
            assert np.allclose(old['angular'],new['angular'],atol=1e-6,rtol=0)
    for old,new in zip(state['articulations'],after['articulations']):
        for key in ['root_pose','qpos','qvel','qf','root_linear','root_angular']:
            assert np.allclose(old[key],new[key],atol=1e-6,rtol=0),key
        assert old['joints']==new['joints'],'Robot drive target restore failed'
    return after


def install(plan):
    from envs._base_task import Base_Task
    import paired_eval
    assert set(plan['tasks'])<=TASKS
    Base_Task.get_cluttered_table=fresh_clutter_call(Base_Task.get_cluttered_table)
    original_init=Base_Task._init_task_env_;original_stable=Base_Task.check_stable
    original_freeze=paired_eval.freeze_initial_state
    pending={}

    def init(self,*args,**kwargs):
        self._paired_seed=int(kwargs['seed']);self._paired_config=kwargs['task_config']
        self._paired_snapshot=None
        return original_init(self,*args,**kwargs)

    def stable(self):
        key=(self.task_name,self._paired_seed)
        rows=paired_eval.read_manifest(self._paired_config,self.task_name)
        saved=next((r for r in rows if r['seed']==self._paired_seed),None)
        state=saved.get('replay_state') if saved else pending.get(key)
        if saved is not None and state is None:
            raise RuntimeError('Certified frozen episode has no complete initial state')
        if state is not None:
            restore(self,state);self._paired_snapshot=state
            print(f'RESTORED_CERTIFIED_SCENE {self.task_name} seed={self._paired_seed}',flush=True)
            return True,[]
        result=original_stable(self)
        if result[0]:
            pending.clear();pending[key]=capture(self);self._paired_snapshot=pending[key]
        return result

    def freeze(task_config,task,index,env):
        original_freeze(task_config,task,index,env)
        rows=paired_eval.read_manifest(task_config,task);row=rows[index]
        state=env._paired_snapshot
        assert state is not None
        if 'replay_state' not in row:
            assert os.environ['ROBOTWIN_MANIFEST_MODE']=='record'
            row['replay_state']=state
            p=paired_eval.manifest_path(task_config,task);temporary=p.with_suffix('.tmp')
            temporary.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n');temporary.replace(p)
        else:assert row['replay_state']==state
        print(f'CERTIFIED_INITIAL_STATE_VERIFIED {task} episode={index}',flush=True)

    Base_Task._init_task_env_=init;Base_Task.check_stable=stable
    paired_eval.freeze_initial_state=freeze
    print('SCENE_REPAIR_ENABLED fresh clutter bounds; complete certified state replay',flush=True)
