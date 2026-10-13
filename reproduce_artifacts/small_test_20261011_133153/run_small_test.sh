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
MAX_STEPS=200
OUT=/data2/hs/User/Program/Compete/LingBot/reproduce_artifacts/small_test_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
if [ -e "$OUT" ]; then
  echo "[STOP] 输出目录已存在：$OUT"
  exit 1
fi
mkdir -p "$OUT"
export OUT
python - <<'PY'
import os
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path('configs/vla/robotwin/robotwin.yaml').read_text())
align = cfg['train']['align_params']
align['depth']['moge_path'] = '/data1/hs/User/Program/Compete/LingBot/models/moge-2-vitb-normal/model.pt'
align['depth']['morgbd_path'] = '/data1/hs/User/Program/Compete/LingBot/models/modelscope/lingbot-vla-v2-6b/depth/model.pt'
align['video']['ckpt_path'] = '/data1/hs/User/Program/Compete/LingBot/models/modelscope/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth'
align['video']['config_path'] = '/data1/hs/User/Program/Compete/LingBot/models/modelscope/lingbot-vla-v2-6b/dino_video/config.yaml'
for group in ('depth','video'):
    for key,value in align[group].items():
        if key.endswith('_path') and not Path(value).is_file():
            raise FileNotFoundError(value)
Path(os.environ['OUT'], 'robotwin_small_test.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
PY
cp /tmp/run_small_test.sh "$OUT/run_small_test.sh"
git rev-parse HEAD > "$OUT/git_revision.txt"
git diff > "$OUT/git_worktree.diff"
git diff be969b8fd117fb70550c5d4bf4bc328211b5b1b6 HEAD > "$OUT/git_from_official.diff"
python -m pip freeze > "$OUT/pip_freeze.txt"
conda list --explicit > "$OUT/conda_explicit.txt"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "输出目录：$OUT"
echo "训练步数：$MAX_STEPS"
echo "有效 global batch size：128 (=1 x 32 x 4)"
echo "配置副本：$OUT/robotwin_small_test.yaml"
set +e
torchrun \
  --standalone \
  --nproc_per_node=4 \
  --master_port=29501 \
  tasks/vla/train_lingbotvla.py \
  "$OUT/robotwin_small_test.yaml" \
  --model.model_path=/data1/hs/User/Program/Compete/LingBot/models/modelscope/lingbot-vla-v2-6b \
  --model.config_path=/data1/hs/User/Program/Compete/LingBot/models/modelscope/lingbot-vla-v2-6b \
  --model.tokenizer_path=/data1/hs/User/Program/Compete/LingBot/models/modelscope/Qwen3-VL-4B-Instruct \
  --data.data_name=robotwin \
  --data.train_path=/data1/hs/User/Program/Compete/LingBot/datasets/RoboTwin_lerobot_v30 \
  --data.norm_stats_file=/data1/hs/User/Program/Compete/LingBot/lingbot-vla-v2/reproduce_artifacts/robotwin_clean_v30_norm.json \
  --data.use_future_image=true \
  --data.num_workers=4 \
  --train.output_dir="$OUT" \
  --train.optimizer=muon \
  --train.lr=2.5e-5 \
  --train.lr_min=2.5e-6 \
  --train.micro_batch_size=1 \
  --train.global_batch_size=128 \
  --train.gradient_accumulation_steps=32 \
  --train.max_steps="$MAX_STEPS" \
  --train.save_steps=1000 \
  --train.ckpt_manager=dcp \
  --train.enable_resume=false \
  --train.enable_gradient_checkpointing=true \
  --train.enable_fp32=false \
  --train.use_compile=false \
  --train.data_parallel_mode=ddp \
  --train.vlm_fsdp=false \
  --train.module_fsdp_enable=false \
  --train.enable_full_shard=false \
  --train.freeze_vision_encoder=false \
  2>&1 | tee "$OUT/train.log"
status=${PIPESTATUS[0]}
printf '%s\n' "$status" > "$OUT/exit_code.txt"
echo "训练退出码: $status" | tee -a "$OUT/train.log"
exit "$status"
