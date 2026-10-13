#!/usr/bin/env bash
set -euo pipefail
source /data1/hs/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
cd /data1/hs/User/Program/Compete/LingBot/lingbot-vla-v2
export GPU_IDS=0,1,2,3
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65536
MAX_STEPS=${MAX_STEPS:-2}
MICRO_BATCH=8
ACCUM_STEPS=32
NUM_WORKERS=8
export MICRO_BATCH ACCUM_STEPS NUM_WORKERS
if [ $((1024 % (4 * MICRO_BATCH))) -ne 0 ]; then echo "[STOP] MICRO_BATCH must divide official global batch 1024 across 4 GPUs"; exit 1; fi
OUT=/data2/hs/User/Program/Compete/LingBot/reproduce_artifacts/official_dist_muon_test_micro8_fp32_workers8_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
if [ -e "$OUT" ]; then
  echo "[STOP] 输出目录已存在：$OUT"
  exit 1
fi
mkdir -p "$OUT"
export OUT
BASE_CONFIG=configs/vla/robotwin/robotwin_dist_muon.yaml
sha256sum "$BASE_CONFIG" > "$OUT/official_config.sha256"
cp -- "$BASE_CONFIG" "$OUT/official_robotwin_dist_muon.yaml"
python - <<'PY'
import os
from pathlib import Path
import yaml
src = Path(os.environ['BASE_CONFIG']) if os.environ.get('BASE_CONFIG') else Path('configs/vla/robotwin/robotwin_dist_muon.yaml')
out = Path(os.environ['OUT']) / 'robotwin_official_4gpu_adapted.yaml'
cfg = yaml.safe_load(src.read_text())
align = cfg['train']['align_params']
align['depth']['moge_path'] = '/data1/hs/User/Program/Compete/LingBot/models/moge-2-vitb-normal/model.pt'
align['depth']['morgbd_path'] = '/data1/hs/User/Program/Compete/LingBot/models/modelscope/lingbot-vla-v2-6b/depth/model.pt'
align['video']['ckpt_path'] = '/data1/hs/User/Program/Compete/LingBot/models/modelscope/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth'
align['video']['config_path'] = '/data1/hs/User/Program/Compete/LingBot/models/modelscope/lingbot-vla-v2-6b/dino_video/config.yaml'
# Paths and data-loader settings are environment/path adaptations, not training hyperparameters.
cfg['data']['train_path'] = '/data1/hs/User/Program/Compete/LingBot/datasets/RoboTwin_lerobot_v30'
cfg['data']['robot_config_root'] = '/data1/hs/User/Program/Compete/LingBot/lingbot-vla-v2/configs/robot_configs'
cfg['data']['num_workers'] = 8
cfg['data']['pin_memory'] = False
cfg['data']['drop_last'] = False
# Official optimization/model hyperparameters remain from robotwin_dist_muon.yaml.
# 4-GPU hardware adaptation: preserve official global batch 1024 with micro batch 8.
cfg['train']['data_parallel_mode'] = 'fsdp2'
cfg['train']['data_parallel_shard_size'] = 4
cfg['train']['data_parallel_replicate_size'] = 1
cfg['train']['micro_batch_size'] = 8
cfg['train']['global_batch_size'] = 1024
cfg['train']['gradient_accumulation_steps'] = 32
cfg['train']['output_dir'] = str(Path(os.environ['OUT']))
cfg['train']['max_steps'] = int(os.environ.get('MAX_STEPS', '2'))
cfg['train']['save_steps'] = 1000
cfg['train']['enable_resume'] = False
cfg['train']['init_device'] = 'cuda'
# Required memory/runtime adaptations for 4 x A100 80GB; official objective hyperparameters unchanged.
cfg['train']['enable_full_shard'] = True
cfg['train']['enable_gradient_checkpointing'] = True
cfg['train']['enable_fp32'] = True
cfg['train']['use_compile'] = False
# Keep official distributed Muon and all official Muon/MoE/loss values untouched.
out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
print(out)
PY
export BASE_CONFIG
# regenerate with BASE_CONFIG visible and verify critical official values
python - <<'PY'
from pathlib import Path
import os,yaml
p=Path(os.environ['OUT'])/'robotwin_official_4gpu_adapted.yaml'
c=yaml.safe_load(p.read_text())
print({k:c['train'].get(k) for k in ['optimizer','lr','lr_min','use_moe_expert_lr','muon_momentum','muon_nesterov','muon_ns_steps','data_parallel_mode','enable_full_shard','enable_gradient_checkpointing','enable_fp32','use_compile','micro_batch_size','global_batch_size','gradient_accumulation_steps']})
PY
cp -- "${BASH_SOURCE[0]}" "$OUT/run_official_dist_muon_micro8_fp32_workers8.sh"
git rev-parse HEAD > "$OUT/git_revision.txt"
git diff > "$OUT/git_worktree.diff"
python -m pip freeze > "$OUT/pip_freeze.txt"
conda list --explicit > "$OUT/conda_explicit.txt"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "输出目录：$OUT"
echo "训练步数：$MAX_STEPS"
echo "global batch size：1024 (= $MICRO_BATCH x $ACCUM_STEPS x 4)"
echo "num_workers=$NUM_WORKERS, enable_fp32=true"
echo "官方基准：$BASE_CONFIG"
echo "适配配置：$OUT/robotwin_official_4gpu_adapted.yaml"
set +e
torchrun --standalone --nproc_per_node=4 --master_port=29502 \
  tasks/vla/train_lingbotvla.py "$OUT/robotwin_official_4gpu_adapted.yaml" \
  --model.model_path=/data1/hs/User/Program/Compete/LingBot/models/modelscope/lingbot-vla-v2-6b \
  --model.config_path=/data1/hs/User/Program/Compete/LingBot/models/modelscope/lingbot-vla-v2-6b \
  --model.tokenizer_path=/data1/hs/User/Program/Compete/LingBot/models/modelscope/Qwen3-VL-4B-Instruct \
  --data.data_name=robotwin \
  --data.train_path=/data1/hs/User/Program/Compete/LingBot/datasets/RoboTwin_lerobot_v30 \
  --data.norm_stats_file=/data1/hs/User/Program/Compete/LingBot/lingbot-vla-v2/reproduce_artifacts/robotwin_clean_v30_norm.json \
  --data.use_future_image=true \
  --data.num_workers="$NUM_WORKERS" \
  --data.pin_memory=false \
  --data.drop_last=false \
  --train.output_dir="$OUT" \
  --train.micro_batch_size="$MICRO_BATCH" \
  --train.global_batch_size=1024 \
  --train.gradient_accumulation_steps="$ACCUM_STEPS" \
  --train.max_steps="$MAX_STEPS" \
  --train.save_steps=1000 \
  --train.enable_resume=false \
  --train.data_parallel_mode=fsdp2 \
  --train.data_parallel_shard_size=4 \
  --train.data_parallel_replicate_size=1 \
  --train.enable_full_shard=true \
  --train.enable_gradient_checkpointing=true \
  --train.enable_fp32=true \
  --train.use_compile=false \
  --train.init_device=cuda \
  2>&1 | tee "$OUT/train.log"
status=${PIPESTATUS[0]}
printf '%s\n' "$status" > "$OUT/exit_code.txt"
echo "训练退出码: $status" | tee -a "$OUT/train.log"
exit "$status"
