#!/usr/bin/env bash
set -euo pipefail

# RoboTwin 2.0 UniShare n-step evaluation.
# This is an independent prefix sweep and does not alter the existing
# eval_robotwin_incremental_8xh100.sh (wo/w) chain.

REPO_DIR="${REPO_DIR:-/root/feihong/FAST_WAM_github}"
FASTWAM_ENV="${FASTWAM_ENV:-/root/.venvs/fastwam}"
PYTHON_BIN="${PYTHON_BIN:-${FASTWAM_ENV}/bin/python}"
FASTWAM_STORAGE_ROOT="${FASTWAM_STORAGE_ROOT:-/root/feihong}"
RUNS_ROOT="${RUNS_ROOT:-${FASTWAM_STORAGE_ROOT}/FastWAM/formal_runs/FAST_WAM_github}"
TASK_NAME="${TASK_NAME:-robotwin_unified_shared_3cam_384_1e-4}"
RUN_DIR="${RUN_DIR:-${RUNS_ROOT}/${TASK_NAME}/2026-07-01_00-51-30}"
CKPT="${CKPT:-${RUN_DIR}/checkpoints/weights/latest.pt}"
DATASET_STATS="${DATASET_STATS:-${RUN_DIR}/dataset_stats.json}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FASTWAM_STORAGE_ROOT}/FastWAM/evaluate_results/robotwin_prefix_shared_T10_8xh100/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${FASTWAM_STORAGE_ROOT}/FastWAM/evaluate_logs/robotwin_prefix_shared_T10_8xh100/${RUN_ID}}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-1}"
EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-100}"
REPLAN_STEPS="${REPLAN_STEPS:-24}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
N_START="${N_START:-0}"
N_END="${N_END:-10}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
cd "${REPO_DIR}"
export PATH="${FASTWAM_ENV}/bin:${PATH}"
export PYTHONUNBUFFERED=1
# SAPIEN/Vulkan is initialized once per worker; stagger startup to avoid
# concurrent renderer initialization crashes on multi-GPU JiHe nodes.
export ROBOTWIN_EVAL_LAUNCH_DELAY_SECONDS="${ROBOTWIN_EVAL_LAUNCH_DELAY_SECONDS:-30}"
export HYDRA_FULL_ERROR=1
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${FASTWAM_STORAGE_ROOT}/FastWAM/checkpoints}"
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/src:${REPO_DIR}/third_party/RoboTwin:${PYTHONPATH:-}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Python env not found: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${CKPT}" ]] || { echo "Checkpoint not found: ${CKPT}" >&2; exit 1; }
[[ -f "${DATASET_STATS}" ]] || { echo "Dataset stats not found: ${DATASET_STATS}" >&2; exit 1; }
[[ "${N_START}" =~ ^[0-9]+$ && "${N_END}" =~ ^[0-9]+$ ]] || { echo "N_START/N_END must be integers" >&2; exit 1; }
(( N_START <= N_END && N_END <= NUM_INFERENCE_STEPS )) || { echo "Require 0 <= N_START <= N_END <= NUM_INFERENCE_STEPS" >&2; exit 1; }

cat <<EOF_CFG
[robotwin_prefix_eval]
  task=${TASK_NAME}
  ckpt=${CKPT}
  dataset_stats=${DATASET_STATS}
  gpu_ids=${GPU_IDS}
  num_gpus=${NUM_GPUS}
  max_tasks_per_gpu=${MAX_TASKS_PER_GPU}
  launch_delay_seconds=${ROBOTWIN_EVAL_LAUNCH_DELAY_SECONDS}
  episodes_per_phase=${EVAL_NUM_EPISODES}
  replan_steps=${REPLAN_STEPS}
  num_inference_steps=${NUM_INFERENCE_STEPS}
  prefix_range=${N_START}..${N_END}
  output_root=${OUTPUT_ROOT}
EOF_CFG

IFS=',' read -r -a gpu_array <<< "${GPU_IDS}"
(( ${#gpu_array[@]} == NUM_GPUS )) || { echo "GPU_IDS count does not equal NUM_GPUS" >&2; exit 1; }

for (( n=N_START; n<=N_END; n++ )); do
  n_out="${OUTPUT_ROOT}/n${n}"
  n_log="${LOG_ROOT}/n${n}.log"
  mkdir -p "${n_out}"
  echo "[$(date '+%F %T')] starting prefix n=${n}" | tee -a "${n_log}"
  set +e
  ROBOTWIN_EVAL_GPU_IDS="${GPU_IDS}" \
  "${PYTHON_BIN}" experiments/robotwin/run_robotwin_manager.py \
    "task=${TASK_NAME}" \
    "ckpt=${CKPT}" \
    "EVALUATION.dataset_stats_path=${DATASET_STATS}" \
    "EVALUATION.output_dir=${n_out}" \
    "EVALUATION.inference_mode=prefix" \
    "EVALUATION.video_prefix_steps=${n}" \
    "EVALUATION.eval_num_episodes=${EVAL_NUM_EPISODES}" \
    "EVALUATION.replan_steps=${REPLAN_STEPS}" \
    "EVALUATION.num_inference_steps=${NUM_INFERENCE_STEPS}" \
    "MULTIRUN.num_gpus=${NUM_GPUS}" \
    "MULTIRUN.max_tasks_per_gpu=${MAX_TASKS_PER_GPU}" \
    > >(tee -a "${n_log}") 2>&1
  status=$?
  set -e
  if (( status != 0 )); then
    echo "[$(date '+%F %T')] prefix n=${n} failed (rc=${status})" | tee -a "${n_log}"
    exit "${status}"
  fi
  echo "[$(date '+%F %T')] prefix n=${n} completed; summary=${REPO_DIR}/evaluate_results/robotwin" | tee -a "${n_log}"
done

echo "[done] RoboTwin UniShare prefix sweep completed: ${OUTPUT_ROOT}"
