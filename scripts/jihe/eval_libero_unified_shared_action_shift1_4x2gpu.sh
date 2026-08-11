#!/usr/bin/env bash
set -euo pipefail

# Standalone LIBERO evaluation for the Unified Shared checkpoint trained with
# action scheduler train/infer shift=1.0. Runs wo and w in parallel on 2x2 GPUs.

REPO_DIR="${REPO_DIR:-/root/feihong/FAST_WAM_github}"
FASTWAM_ENV="${FASTWAM_ENV:-/root/.venvs/fastwam}"
PYTHON_BIN="${PYTHON_BIN:-${FASTWAM_ENV}/bin/python}"
TASK_NAME="${TASK_NAME:-libero_unified_shared_2cam224_1e-4}"
TRAIN_RUN_GROUP="${TRAIN_RUN_GROUP:-libero_unified_shared_action_shift1_2cam224_1e-4}"
RUN_ID="${RUN_ID:-action_shift1_$(date +%Y-%m-%d_%H-%M-%S)}"
RUNS_ROOT="${RUNS_ROOT:-/root/feihong/FastWAM/formal_runs/FAST_WAM_github}"
FASTWAM_LIBERO_ROOT="${FASTWAM_LIBERO_ROOT:-/root/feihong/FastWAM/third_party/LIBERO}"
FASTWAM_LIBERO_DATASETS="${FASTWAM_LIBERO_DATASETS:-/root/feihong/FastWAM/datasets/libero_mujoco3.3.2}"
LIBERO_CONFIG_DIR="${LIBERO_CONFIG_DIR:-/root/feihong/FastWAM/libero_config_fastwam}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-/root/feihong/FastWAM/evaluate_results/libero_unified_shared_action_shift1_4x2gpu/${RUN_ID}}"
EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-/root/feihong/FastWAM/evaluate_logs/libero_unified_shared_action_shift1_4x2gpu/${RUN_ID}}"
NUM_TRIALS="${NUM_TRIALS:-50}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-1}"
MAX_EVAL_RESTARTS="${MAX_EVAL_RESTARTS:-3}"
MAX_INVALID_EPISODE_RETRIES="${MAX_INVALID_EPISODE_RETRIES:-100}"
FORCE_RERUN="${FORCE_RERUN:-1}"
REDIRECT_COMMON_FILES="${REDIRECT_COMMON_FILES:-false}"

mkdir -p "${EVAL_OUTPUT_ROOT}" "${EVAL_LOG_ROOT}"
cd "${REPO_DIR}"

export PATH="${FASTWAM_ENV}/bin:${PATH}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/root/feihong/FastWAM/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export HF_HOME="${HF_HOME:-/root/feihong/FastWAM/.cache/huggingface}"

fail() { echo "[error] $*" >&2; exit 1; }
require_path() { [[ -e "$1" ]] || fail "Missing required path: $1"; }

setup_fastwam_libero() {
  local benchmark_root="${FASTWAM_LIBERO_ROOT}/libero/libero"
  require_path "${benchmark_root}"
  require_path "${benchmark_root}/assets"
  require_path "${benchmark_root}/bddl_files"
  require_path "${benchmark_root}/init_files"
  require_path "${FASTWAM_LIBERO_DATASETS}"
  mkdir -p "${LIBERO_CONFIG_DIR}"
  cat > "${LIBERO_CONFIG_DIR}/config.yaml" <<EOF
assets: ${benchmark_root}/assets
bddl_files: ${benchmark_root}/bddl_files
benchmark_root: ${benchmark_root}
datasets: ${FASTWAM_LIBERO_DATASETS}
init_states: ${benchmark_root}/init_files
EOF
  export FASTWAM_LIBERO_ROOT
  export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_DIR}"
  export PYTHONPATH="${FASTWAM_LIBERO_ROOT}:${REPO_DIR}:${REPO_DIR}/src:${PYTHONPATH:-}"
}

latest_complete_run_dir() {
  local task_root="${RUNS_ROOT}/${TRAIN_RUN_GROUP}"
  local run_dir=""
  local latest=""
  [[ -d "${task_root}" ]] || fail "Missing run root: ${task_root}"
  while IFS= read -r run_dir; do
    if [[ -s "${run_dir}/checkpoints/weights/latest.pt" && -s "${run_dir}/dataset_stats.json" ]]; then
      latest="${run_dir}"
    fi
  done < <(find "${task_root}" -mindepth 1 -maxdepth 1 -type d -print | sort)
  [[ -n "${latest}" ]] || fail \
    "No complete run found under ${task_root} (requires checkpoints/weights/latest.pt and dataset_stats.json)"
  printf "%s\n" "${latest}"
}

RUN_DIR="${RUN_DIR:-$(latest_complete_run_dir)}"

setup_fastwam_libero

CKPT="${CKPT:-${RUN_DIR}/checkpoints/weights/latest.pt}"
DATASET_STATS="${DATASET_STATS:-${RUN_DIR}/dataset_stats.json}"
require_path "${CKPT}"
require_path "${DATASET_STATS}"

summary_is_complete() {
  local out_dir="$1"
  [[ -s "${out_dir}/summary.json" ]] || return 1
  [[ "$(find "${out_dir}" -type f -name 'gpu*_task*_results.json' | wc -l | tr -d ' ')" == "40" ]]
}

run_eval_job() {
  local label="$1"
  local mode="$2"
  local gpu_pair="$3"
  local out_dir="${EVAL_OUTPUT_ROOT}/${label}"
  local log_file="${EVAL_LOG_ROOT}/${label}.log"
  mkdir -p "${out_dir}" "${EVAL_LOG_ROOT}"

  if [[ "${FORCE_RERUN}" != "1" ]] && summary_is_complete "${out_dir}"; then
    echo "[${label}] complete result exists, skip: ${out_dir}" | tee -a "${log_file}"
    return 0
  fi

  local attempt=1
  while (( attempt <= MAX_EVAL_RESTARTS )); do
    echo "[${label}] attempt=${attempt}/${MAX_EVAL_RESTARTS} mode=${mode} gpus=${gpu_pair}" | tee -a "${log_file}"
    rm -rf "${out_dir}"
    mkdir -p "${out_dir}"

    set +e
    CUDA_VISIBLE_DEVICES="${gpu_pair}" \
      "${PYTHON_BIN}" experiments/libero/run_libero_manager.py \
      "task=${TASK_NAME}" \
      "ckpt=${CKPT}" \
      "EVALUATION.dataset_stats_path=${DATASET_STATS}" \
      "EVALUATION.inference_mode=${mode}" \
      "EVALUATION.output_dir=${out_dir}" \
      "EVALUATION.num_trials=${NUM_TRIALS}" \
      "EVALUATION.save_videos=false" \
      "EVALUATION.retry_invalid_episodes=true" \
      "EVALUATION.max_invalid_episode_retries=${MAX_INVALID_EPISODE_RETRIES}" \
      "EVALUATION.black_screen_filter=true" \
      "MULTIRUN.num_gpus=2" \
      "MULTIRUN.max_tasks_per_gpu=${MAX_TASKS_PER_GPU}" \
      "model.redirect_common_files=${REDIRECT_COMMON_FILES}" \
      model.action_scheduler.train_shift=1.0 \
      model.action_scheduler.infer_shift=1.0 \
      "$@" >> "${log_file}" 2>&1
    local status="$?"
    set -e

    if [[ "${status}" == "0" ]] && summary_is_complete "${out_dir}"; then
      echo "[${label}] done: ${out_dir}" | tee -a "${log_file}"
      return 0
    fi
    echo "[${label}] incomplete/failed status=${status}; retrying" | tee -a "${log_file}"
    attempt=$((attempt + 1))
  done

  return 1
}

echo "[libero_unified_shared_action_shift1]"
echo "  task=${TASK_NAME}"
echo "  train_run_group=${TRAIN_RUN_GROUP}"
echo "  run_dir=${RUN_DIR}"
echo "  ckpt=${CKPT}"
echo "  dataset_stats=${DATASET_STATS}"
echo "  output_root=${EVAL_OUTPUT_ROOT}"
echo "  action_train_shift=1.0 action_infer_shift=1.0"

failed=0
run_eval_job "unified_shared_wo" "wo" "0,1" "$@" & pid_wo=$!
run_eval_job "unified_shared_w" "w" "2,3" "$@" & pid_w=$!

wait "${pid_wo}" || failed=1
wait "${pid_w}" || failed=1

if [[ "${failed}" != "0" ]]; then
  fail "At least one eval job failed. Logs: ${EVAL_LOG_ROOT}"
fi

echo "[done] Unified Shared LIBERO action-shift=1.0 evaluation completed: ${EVAL_OUTPUT_ROOT}"
