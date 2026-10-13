import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('/data2/hs/User/Program/Compete/LingBot')
TRAIN = ROOT / 'reproduce_artifacts/official_dist_muon_batch160_seed42_5000_20261011_163354'
REPO = ROOT / 'lingbot-vla-v2'
ASSETS = TRAIN / 'inference_assets'
os.chdir(REPO)
sys.path.insert(0, str(REPO))

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, required=True)
args = parser.parse_args()
manifest = json.loads((ASSETS / 'normalization_manifest.json').read_text())
for name, item in manifest['assets'].items():
    assert hashlib.sha256((ASSETS / name).read_bytes()).hexdigest() == item['sha256'], name
assert (REPO / 'configs/robot_configs/robotwin.yaml').read_bytes() == (ASSETS / 'robotwin.yaml').read_bytes()
assert hashlib.sha256((TRAIN / 'lingbotvla_cli.yaml').read_bytes()).hexdigest() == manifest['training_config']['sha256']

from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
from deploy.websocket_policy_server import WebsocketPolicyServer

checkpoint = TRAIN / 'checkpoints/global_step_5000/hf_ckpt'
policy = LingbotVLAv2Server(str(checkpoint), robot_norm_path=str(ASSETS / 'norm_stats.json'),
                         use_length=50, chunk_ret=True, use_bf16=False,
                         use_fp32=True, use_compile=True)
policy.reset('robotwin')
print('READY: global_step_5000, archived normalization, FP32, chunk=50', flush=True)
WebsocketPolicyServer(policy, host='127.0.0.1', port=args.port,
                      metadata={'checkpoint': str(checkpoint), 'norm_stats_sha256': manifest['assets']['norm_stats.json']['sha256']}).serve_forever()
