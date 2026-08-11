#!/usr/bin/env bash
set -euo pipefail

cd /root/feihong/FAST_WAM_github

# Stage the 32GB Wan2.2 asset on the local 100GB tmpfs. /root/feihong is a
# Ceph network filesystem; loading eight workers directly from it can stall.
MODEL_SOURCE_ROOT=/root/feihong/FastWAM/checkpoints
MODEL_CACHE_ROOT=/dev/shm/fastwam_model_cache
MODEL_SOURCE_DIR="$MODEL_SOURCE_ROOT/Wan-AI/Wan2.2-TI2V-5B"
MODEL_CACHE_DIR="$MODEL_CACHE_ROOT/Wan-AI/Wan2.2-TI2V-5B"
TOKENIZER_SOURCE_DIR="$MODEL_SOURCE_ROOT/Wan-AI/Wan2.1-T2V-1.3B"
TOKENIZER_CACHE_DIR="$MODEL_CACHE_ROOT/Wan-AI/Wan2.1-T2V-1.3B"
if [[ "${FASTWAM_STAGE_MODEL:-1}" == "1" ]]; then
  if [[ ! -f "$MODEL_CACHE_DIR/.copy_complete" ]]; then
    echo "[$(date '+%F %T')] Staging Wan2.2 model to $MODEL_CACHE_ROOT"
    mkdir -p "$MODEL_CACHE_ROOT/Wan-AI"
    rm -rf "$MODEL_CACHE_ROOT/Wan-AI/Wan2.2-TI2V-5B.tmp"
    cp -a "$MODEL_SOURCE_DIR" "$MODEL_CACHE_ROOT/Wan-AI/Wan2.2-TI2V-5B.tmp"
    mv "$MODEL_CACHE_ROOT/Wan-AI/Wan2.2-TI2V-5B.tmp" "$MODEL_CACHE_DIR"
    touch "$MODEL_CACHE_DIR/.copy_complete"
  else
    echo "[$(date '+%F %T')] Reusing staged Wan2.2 model at $MODEL_CACHE_ROOT"
  fi
  if [[ ! -f "$TOKENIZER_CACHE_DIR/google/umt5-xxl/tokenizer_config.json" ]]; then
    echo "[$(date '+%F %T')] Staging Wan2.1 tokenizer assets"
    rm -rf "$MODEL_CACHE_ROOT/Wan-AI/Wan2.1-T2V-1.3B.tmp"
    cp -a "$TOKENIZER_SOURCE_DIR" "$MODEL_CACHE_ROOT/Wan-AI/Wan2.1-T2V-1.3B.tmp"
    mv "$MODEL_CACHE_ROOT/Wan-AI/Wan2.1-T2V-1.3B.tmp" "$TOKENIZER_CACHE_DIR"
  fi
  export DIFFSYNTH_MODEL_BASE_PATH="$MODEL_CACHE_ROOT"
else
  export DIFFSYNTH_MODEL_BASE_PATH="$MODEL_SOURCE_ROOT"
fi
export DIFFSYNTH_SKIP_DOWNLOAD=true
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export LIBERO_CONFIG_PATH=/root/feihong/FastWAM/libero_config_fastwam_local
export PYTHONPATH=/root/feihong/LIBERO:/root/feihong/FAST_WAM_github:/root/feihong/FAST_WAM_github/src
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TASK_LAUNCH_DELAY_SECONDS=120
export PYTHON_BIN=/root/.venvs/fastwam/bin/python
export USE_TMUX=0
export MONITORING_INTERVAL=30
export STATUS_INTERVAL=120

CKPT=/root/feihong/FastWAM/formal_runs/FAST_WAM_github/libero_unified_shared_action_shift1_2cam224_1e-4/2026-08-03_22-44-39/checkpoints/weights/latest.pt
STATS=/root/feihong/FastWAM/formal_runs/FAST_WAM_github/libero_unified_shared_action_shift1_2cam224_1e-4/2026-08-03_22-44-39/dataset_stats.json
BASE_OUT=/root/feihong/FastWAM/evaluate_results/libero_prefix_shared_action_shift1_T10_8xh100/$(date +%Y%m%d_%H%M%S)

mkdir -p "${BASE_OUT}"

for n in 0 1 2 3 4 5 6 7 8 9 10; do
  echo "[$(date '+%F %T')] ===== Starting prefix n=${n} ====="
  /root/.venvs/fastwam/bin/python experiments/libero/run_libero_manager.py \
    task=libero_unified_shared_2cam224_1e-4 \
    ckpt="${CKPT}" \
    model.redirect_common_files=false \
    EVALUATION.dataset_stats_path="${STATS}" \
    EVALUATION.output_dir="${BASE_OUT}/n${n}" \
    EVALUATION.inference_mode=prefix \
    EVALUATION.video_prefix_steps="${n}" \
    EVALUATION.num_inference_steps=10 \
    model.action_scheduler.train_shift=1.0 \
    model.action_scheduler.infer_shift=1.0 \
    model.video_scheduler.train_shift=5.0 \
    model.video_scheduler.infer_shift=5.0 \
    EVALUATION.num_trials=50 \
    EVALUATION.save_videos=false \
    EVALUATION.retry_invalid_episodes=true \
    EVALUATION.max_invalid_episode_retries=100 \
    EVALUATION.black_screen_filter=true \
    MULTIRUN.num_gpus=8 \
    MULTIRUN.max_tasks_per_gpu=1
  echo "[$(date '+%F %T')] ===== Finished prefix n=${n} ====="
done
