import os
import runpy
import sys
from pathlib import Path
from video_recording import wrap_take_action

repo = Path('/data2/hs/User/Program/Compete/LingBot/RoboTwin')
os.chdir(repo)
sys.path.insert(0, str(repo))
from envs._base_task import Base_Task
import script.deploy.websocket_client_policy as client_module


class LocalClient(client_module.WebsocketClientPolicy):
    def __init__(self, host='127.0.0.1', port=None, api_key=None):
        super().__init__(host=host, port=port, api_key=api_key)


client_module.WebsocketClientPolicy = LocalClient

Base_Task.take_action = wrap_take_action(Base_Task.take_action)
print('Video recording: fresh frame after every action, 10 fps', flush=True)
runpy.run_path(str(repo / 'script/eval_policy_client_lingbotvla.py'), run_name='__main__')
