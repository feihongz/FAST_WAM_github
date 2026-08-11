#!/usr/bin/env bash
set -euo pipefail

cd /root/feihong/FAST_WAM_github

export DIFFSYNTH_MODEL_BASE_PATH=/root/public/models
export DIFFSYNTH_SKIP_DOWNLOAD=true
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export LIBERO_CONFIG_PATH=/root/feihong/FastWAM/libero_config_fastwam_local
export PYTHONPATH=/root/feihong/LIBERO:/root/feihong/FAST_WAM_github:/root/feihong/FAST_WAM_github/src
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

PYTHON_BIN=/root/.venvs/fastwam/bin/python
CKPT=/root/feihong/FastWAM/formal_runs/FAST_WAM_github/libero_unified_shared_2cam224_1e-4/2026-07-01_00-44-20/checkpoints/weights/latest.pt
STATS=/root/feihong/FastWAM/formal_runs/FAST_WAM_github/libero_unified_shared_2cam224_1e-4/2026-07-01_00-44-20/dataset_stats.json
OUT=/root/feihong/FastWAM/evaluate_results/libero_prefix_shared_T10/prefix_shared_T10_20260710_2218/n9
LOG_DIR="${OUT}/resume_task_logs"
mkdir -p "${LOG_DIR}"

if [ -s "${OUT}/failed_tasks.txt" ]; then
  mv "${OUT}/failed_tasks.txt" "${OUT}/failed_tasks.before_resume_$(date '+%Y%m%d_%H%M%S').txt"
fi

TASKS=(
  "libero_spatial,1"
  "libero_spatial,2"
  "libero_spatial,3"
  "libero_spatial,4"
  "libero_spatial,5"
  "libero_spatial,6"
  "libero_spatial,7"
  "libero_spatial,8"
  "libero_spatial,9"
  "libero_object,0"
  "libero_object,1"
  "libero_object,2"
  "libero_object,3"
  "libero_object,4"
  "libero_object,5"
  "libero_object,6"
  "libero_object,7"
  "libero_object,8"
  "libero_object,9"
)

for task in "${TASKS[@]}"; do
  suite="${task%,*}"
  task_id="${task#*,}"
  if ls "${OUT}/${suite}"/gpu*_task"${task_id}"_results.json >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] Skip existing ${suite},${task_id}"
    continue
  fi

  echo "[$(date '+%F %T')] Run ${suite},${task_id}"
  log_file="${LOG_DIR}/${suite}_task${task_id}_gpu0.log"
  set +e
  "${PYTHON_BIN}" experiments/libero/eval_libero_single.py \
    task=libero_unified_shared_2cam224_1e-4 \
    ckpt="${CKPT}" \
    EVALUATION.task_suite_name="${suite}" \
    EVALUATION.task_id="${task_id}" \
    gpu_id=0 \
    EVALUATION.num_trials=50 \
    EVALUATION.output_dir="${OUT}" \
    model.redirect_common_files=false \
    EVALUATION.dataset_stats_path="${STATS}" \
    EVALUATION.inference_mode=prefix \
    EVALUATION.video_prefix_steps=9 \
    EVALUATION.num_inference_steps=10 \
    EVALUATION.save_videos=false \
    EVALUATION.retry_invalid_episodes=true \
    EVALUATION.max_invalid_episode_retries=100 \
    EVALUATION.black_screen_filter=true \
    > "${log_file}" 2>&1
  rc=$?
  set -e

  if [ "${rc}" -ne 0 ]; then
    echo "[$(date '+%F %T')] Failed ${suite},${task_id} rc=${rc} log=${log_file}" | tee -a "${OUT}/resume_failed_tasks.txt"
    exit "${rc}"
  fi
done

echo "[$(date '+%F %T')] Generate summary"
"${PYTHON_BIN}" experiments/libero/summarize_results.py --output_dir="${OUT}"
echo "[$(date '+%F %T')] Resume complete"
