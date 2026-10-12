#!/usr/bin/env bash
set -u
BASE=/data1/hs/User/Program/Compete/LingBot
REPO="$BASE/lingbot-vla-v2"
OUT1000=/data2/hs/User/Program/Compete/LingBot/reproduce_artifacts/single_gpu_train_1000_resume
OUT10000=/data2/hs/User/Program/Compete/LingBot/reproduce_artifacts/single_gpu_train_10000
WATCH=/data2/hs/User/Program/Compete/LingBot/reproduce_artifacts/watch_1000_resume_then_10000.log
PID=1238368
PYTHON=/data1/hs/miniconda3/envs/lerobot/bin/python

echo "[$(date)] watcher started for resumed 1000 PID=$PID" >> "$WATCH"
while kill -0 "$PID" 2>/dev/null; do
  sleep 30
done

echo "[$(date)] resumed 1000-step process exited" >> "$WATCH"
CKPT="$OUT1000/checkpoints/global_step_1000"
if [ ! -d "$CKPT" ]; then
  echo "[$(date)] ERROR: missing $CKPT; will not start 10000-step training" >> "$WATCH"
  exit 1
fi

echo "[$(date)] confirmed $CKPT; launching independent 10000-step training" >> "$WATCH"
mkdir -p "$OUT10000"
export CUDA_VISIBLE_DEVICES=4
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ALIGN='{"mode":"query","num_task_tokens":8,"depth_loss_weight":0.004,"future_depth_loss_weight":0.004,"use_future_video":true,"llm":{"dim_out":2560,"image_token_size":8,"image_input_size":224},"depth":{"model_type":"MoRGBD","moge_path":"/data1/hs/User/Program/Compete/LingBot/models/moge-2-vitb-normal/model.pt","morgbd_path":"/data1/hs/User/Program/Compete/LingBot/models/modelscope/lingbot-vla-v2-6b/depth/model.pt","num_layers":1,"num_heads":4,"dim_head":32,"ff_mult":1,"num_backbone_tokens":256,"token_size":16,"dim_out":1024,"input_size":224,"use_future_depth":true,"block_future_depth_to_action":false,"detach_future_image_feats":true},"video":{"ckpt_path":"/data1/hs/User/Program/Compete/LingBot/models/modelscope/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth","config_path":"/data1/hs/User/Program/Compete/LingBot/models/modelscope/lingbot-vla-v2-6b/dino_video/config.yaml","attention_mode":"flex_block_causal","input_size":256,"block_suffix_to_future_video":false,"share_future_depth_query":true,"use_shared_future_task_proj":true,"use_current_shared_task_proj":true,"num_future_frames":1,"use_warmup_frame":true,"effective_fps":1.0,"n_blocks":1,"cls_pool":"last","detach_image_feats":true,"num_layers":1,"num_heads":4,"dim_head":32,"ff_mult":1,"num_backbone_tokens":256,"dim_out":1024,"future_video_loss_weight":0.004,"use_smooth_l1_loss":false,"use_mse_loss":true,"mse_loss_weight":1.0,"use_patch_loss":true,"use_current_patch_loss":true,"use_cosine_loss":true,"cosine_loss_weight":0.2,"use_cls_loss":false,"cls_loss_type":"mse","cls_loss_weight":0.2,"log_max_samples":32,"log_scale":16},"visual_steps":5000}'
cd "$REPO"
"$PYTHON" -u -m torch.distributed.run \
  --standalone \
  --nproc_per_node=1 \
  tasks/vla/train_lingbotvla.py \
  configs/vla/robotwin/robotwin.yaml \
  --model.model_path="$BASE/models/modelscope/lingbot-vla-v2-6b" \
  --model.config_path="$BASE/models/modelscope/lingbot-vla-v2-6b" \
  --model.tokenizer_path="$BASE/models/modelscope/Qwen3-VL-4B-Instruct" \
  --data.data_name=robotwin \
  --data.train_path="$BASE/datasets/RoboTwin_lerobot_v30" \
  --data.robot_config_root="$REPO/configs/robot_configs" \
  --data.num_workers=0 \
  --data.drop_last=false \
  --data.pin_memory=false \
  --data.use_future_image=true \
  --train.output_dir="$OUT10000" \
  --train.data_parallel_mode=ddp \
  --train.micro_batch_size=1 \
  --train.global_batch_size=1 \
  --train.gradient_accumulation_steps=1 \
  --train.max_steps=10000 \
  --train.save_steps=1000 \
  --train.num_train_epochs=1 \
  --train.optimizer=muon \
  --train.lr=1e-7 \
  --train.lr_min=1e-7 \
  --train.enable_resume=false \
  --train.enable_gradient_checkpointing=true \
  --train.enable_mixed_precision=true \
  --train.enable_fp32=false \
  --train.use_compile=false \
  --train.module_fsdp_enable=false \
  --train.rmpad=false \
  --train.rmpad_with_pos_ids=false \
  --train.vlm_fsdp=false \
  --train.align_params="$ALIGN" \
  > "$OUT10000/train.log" 2>&1
status=$?
echo "[$(date)] 10000-step process exited status=$status" >> "$WATCH"
exit "$status"
