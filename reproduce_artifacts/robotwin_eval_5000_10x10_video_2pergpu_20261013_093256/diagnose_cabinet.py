"""Time a single expert precheck in isolation; no policy episode is evaluated."""
import ast
import os
import sys
import time
from pathlib import Path

REPO = Path('/data2/hs/User/Program/Compete/LingBot/RoboTwin')
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.extend([str(REPO / 'policy'), str(REPO / 'description/utils')])
import script.deploy.websocket_client_policy as client


class NoPolicy:
    def __init__(self, **kwargs):
        pass


client.WebsocketClientPolicy = NoPolicy
from envs._base_task import Base_Task
from envs.robot.planner import CuroboPlanner
from envs.put_object_cabinet import put_object_cabinet


def timed_method(cls, name):
    original = getattr(cls, name)
    def measured(self, *args, **kwargs):
        begin = time.monotonic()
        print(f'ENTER {cls.__name__}.{name}', flush=True)
        try:
            return original(self, *args, **kwargs)
        finally:
            print(f'EXIT {cls.__name__}.{name}: {time.monotonic() - begin:.2f}s', flush=True)
    setattr(cls, name, measured)


for cls, names in [(Base_Task, ['setup_scene', 'load_robot']),
                   (CuroboPlanner, ['__init__']),
                   (put_object_cabinet, ['load_actors', 'play_once'])]:
    for name in names:
        timed_method(cls, name)

source = REPO / 'script/eval_policy_client_lingbotvla.py'
tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == 'eval_policy':
        node.body = ast.parse('''
args['eval_mode'] = True
print(f'EXPERT PRECHECK ONLY, seed={st_seed}', flush=True)
TASK_ENV.setup_demo(now_ep_num=0, seed=st_seed, is_test=True, **args)
print('SETUP COMPLETE', flush=True)
TASK_ENV.play_once()
print(f'EXPERT RESULT: plan_success={TASK_ENV.plan_success}, success={TASK_ENV.check_success()}', flush=True)
TASK_ENV.close_env()
raise SystemExit(0)
''').body
ast.fix_missing_locations(tree)
exec(compile(tree, str(source), 'exec'), {'__name__': '__main__', '__file__': str(source)})
