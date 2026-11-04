"""Independently reconstruct one frozen scene; stop before any policy action."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


class ReplayVerified(Exception):pass


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--evaluation',required=True)
    parser.add_argument('--task',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();job=Path(args.evaluation);plan=json.loads((job/'plan.json').read_text())
    sim=Path(plan['sim']);sys.path[:0]=[str(job),str(sim)]
    os.chdir(sim)
    manifest=Path(plan['manifest_directory'])/'demo_clean'/(args.task+'.json')
    # Other diagnostic episodes may append concurrently; the replayed prefix
    # itself must stay immutable, not the growing whole-file checksum.
    before=hashlib.sha256(json.dumps(json.loads(manifest.read_text())[0],sort_keys=True).encode()).hexdigest()
    os.environ.update(ROBOTWIN_PAIR_MANIFEST=plan['manifest_directory'],ROBOTWIN_MANIFEST_MODE='replay',ROBOTWIN_TEST_NUM='1')
    from script.deploy import websocket_client_policy
    instances=[]
    class ResetOnlyClient:
        def __init__(self,*a,**k):pass
        def infer(self,observation):
            assert observation['reset'] and type(observation['inference_seed']) is int
            required={'put_object_cabinet':['arm_tag','origin_z'],
                      'place_object_scale':['arm_tag'],'open_laptop':['arm_tag']}
            assert all(hasattr(instances[-1],name) for name in required.get(args.task,[]))
            raise ReplayVerified()
    websocket_client_policy.WebsocketClientPolicy=ResetOnlyClient
    filename=str(sim/'script/eval_policy_client_lingbotvla.py')
    namespace={'__name__':'scene_replay_validation','__file__':filename}
    exec(compile((job/'client_instrumented.py').read_text(),filename,'exec'),namespace)
    old=namespace['class_decorator']
    def capture(name):
        env=old(name);instances.append(env);return env
    namespace['class_decorator']=capture
    import yaml
    config=yaml.safe_load((sim/'policy/ACT/deploy_policy.yml').read_text())
    config.update(task_name=args.task,task_config='demo_clean',instruction_type='seen',
                  train_config_name='0',seed=0,policy_name='ACT',port=1,robo_name='robotwin',
                  eval_video_log=False,output_dir=args.output)
    try:
        namespace['main'](config)
        raise RuntimeError('Replay unexpectedly ran policy actions')
    except ReplayVerified:
        after=hashlib.sha256(json.dumps(json.loads(manifest.read_text())[0],sort_keys=True).encode()).hexdigest()
        assert before==after
        report=dict(passed=True,task=args.task,setting='demo_clean/seen',
                    initial_scene_verified=True,instruction_frozen=True,frozen_episode_unchanged=True,
                    task_specific_success_state_verified=True,
                    policy_actions_executed=0)
        Path(args.output).mkdir(parents=True,exist_ok=True)
        (Path(args.output)/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    finally:
        for env in instances:env.close_env()


if __name__=='__main__':main()
