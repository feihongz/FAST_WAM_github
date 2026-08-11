#!/usr/bin/env bash
set -euo pipefail

cd /root/feihong/FAST_WAM_github

export DIFFSYNTH_MODEL_BASE_PATH=/dev/shm/fastwam_model_cache
export DIFFSYNTH_SKIP_DOWNLOAD=true
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export LIBERO_CONFIG_PATH=/root/feihong/FastWAM/libero_config_fastwam_local
export PYTHONPATH=/root/feihong/LIBERO:/root/feihong/FAST_WAM_github:/root/feihong/FAST_WAM_github/src
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=23456

MODEL_DIR="$DIFFSYNTH_MODEL_BASE_PATH/Wan-AI/Wan2.2-TI2V-5B"
if [[ ! -f "$MODEL_DIR/.copy_complete" ]]; then
  echo "Staging Wan2.2 model to local tmpfs before smoke test..."
  mkdir -p /dev/shm/fastwam_model_cache/Wan-AI
  rm -rf /dev/shm/fastwam_model_cache/Wan-AI/Wan2.2-TI2V-5B.tmp
  cp -a /root/feihong/FastWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B /dev/shm/fastwam_model_cache/Wan-AI/Wan2.2-TI2V-5B.tmp
  mv /dev/shm/fastwam_model_cache/Wan-AI/Wan2.2-TI2V-5B.tmp "$MODEL_DIR"
  touch "$MODEL_DIR/.copy_complete"
fi
TOKENIZER_DIR="$DIFFSYNTH_MODEL_BASE_PATH/Wan-AI/Wan2.1-T2V-1.3B"
if [[ ! -f "$TOKENIZER_DIR/google/umt5-xxl/tokenizer_config.json" ]]; then
  echo "Staging Wan2.1 tokenizer to local tmpfs..."
  rm -rf /dev/shm/fastwam_model_cache/Wan-AI/Wan2.1-T2V-1.3B.tmp
  cp -a /root/feihong/FastWAM/checkpoints/Wan-AI/Wan2.1-T2V-1.3B /dev/shm/fastwam_model_cache/Wan-AI/Wan2.1-T2V-1.3B.tmp
  mv /dev/shm/fastwam_model_cache/Wan-AI/Wan2.1-T2V-1.3B.tmp "$TOKENIZER_DIR"
fi

CKPT=/root/feihong/FastWAM/formal_runs/FAST_WAM_github/libero_unified_shared_action_shift1_2cam224_1e-4/2026-08-03_22-44-39/checkpoints/weights/latest.pt
STATS=/root/feihong/FastWAM/formal_runs/FAST_WAM_github/libero_unified_shared_action_shift1_2cam224_1e-4/2026-08-03_22-44-39/dataset_stats.json
OUT=/root/feihong/FastWAM/evaluate_results/libero_single_smoke_action_shift1/$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

echo "Starting single-task LIBERO smoke test"
echo "GPU=$CUDA_VISIBLE_DEVICES suite=libero_10 task_id=0 trials=1"
echo "Output: $OUT"

exec /root/.venvs/fastwam/bin/python experiments/libero/eval_libero_single.py \
  task=libero_unified_shared_2cam224_1e-4 \
  ckpt="$CKPT" \
  gpu_id=0 \
  EVALUATION.task_suite_name=libero_10 \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=1 \
  EVALUATION.output_dir="$OUT" \
  EVALUATION.dataset_stats_path="$STATS" \
  EVALUATION.inference_mode=prefix \
  EVALUATION.video_prefix_steps=0 \
  EVALUATION.num_inference_steps=10 \
  EVALUATION.save_videos=false \
  EVALUATION.retry_invalid_episodes=false \
  EVALUATION.black_screen_filter=true \
  model.redirect_common_files=false \
  model.action_scheduler.train_shift=1.0 \
  model.action_scheduler.infer_shift=1.0 \
  model.video_scheduler.train_shift=5.0 \
  model.video_scheduler.infer_shift=5.0
