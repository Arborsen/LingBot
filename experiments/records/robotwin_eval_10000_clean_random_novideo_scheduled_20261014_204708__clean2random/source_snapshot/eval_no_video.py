"""Run the simulator without the per-action video recording patch."""
import os
import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / 'plan.json').read_text())
SIM = Path(PLAN['sim'])
os.chdir(SIM)
sys.path.insert(0, str(SIM))
from script.deploy import websocket_client_policy as client_module


class LocalClient(client_module.WebsocketClientPolicy):
    def __init__(self, host='127.0.0.1', port=None, api_key=None):
        super().__init__(host=host, port=port, api_key=api_key)


client_module.WebsocketClientPolicy = LocalClient
print('Video recording OFF; original take_action; episode/timing ledger enabled', flush=True)
filename = str(SIM / 'script/eval_policy_client_lingbotvla.py')
exec(compile((HERE / 'client_instrumented.py').read_text(), filename, 'exec'),
     {'__name__': '__main__', '__file__': filename})
