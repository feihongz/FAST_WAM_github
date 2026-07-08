#!/usr/bin/env bash
set -euo pipefail

# Run the four LIBERO incremental eval modes sequentially on one local H100.
# Inside each mode, run multiple task workers on the same GPU via MAX_TASKS_PER_GPU.

REPO_DIR="${REPO_DIR:-/root/code/feihong/FAST_WAM_github}"
FASTWAM_ENV="${FASTWAM_ENV:-/root/.venvs/fastwam-libero}"
PYTHON_BIN="${PYTHON_BIN:-${FASTWAM_ENV}/bin/python}"
RUN_ID="${RUN_ID:-local_1gpu_$(date +%Y-%m-%d_%H-%M-%S)}"
RUNS_ROOT="${RUNS_ROOT:-/root/feihong/FastWAM/formal_runs/FAST_WAM_github}"
FASTWAM_LIBERO_ROOT="${FASTWAM_LIBERO_ROOT:-/root/nas/temp_nas/FastWAM/third_party/LIBERO}"
FASTWAM_LIBERO_DATASETS="${FASTWAM_LIBERO_DATASETS:-/root/nas/temp_nas/FastWAM/datasets/libero_mujoco3.3.2}"
LIBERO_CONFIG_DIR="${LIBERO_CONFIG_DIR:-/root/feihong/FastWAM/libero_config_fastwam}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-/root/feihong/FastWAM/evaluate_results/libero_incremental_1gpu_local/${RUN_ID}}"
EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-/root/feihong/FastWAM/evaluate_logs/libero_incremental_1gpu_local/${RUN_ID}}"
GPU_ID="${GPU_ID:-0}"
NUM_TRIALS="${NUM_TRIALS:-50}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-2}"
MAX_EVAL_RESTARTS="${MAX_EVAL_RESTARTS:-3}"
MAX_INVALID_EPISODE_RETRIES="${MAX_INVALID_EPISODE_RETRIES:-100}"
BLACK_SCREEN_MEAN_THRESHOLD="${BLACK_SCREEN_MEAN_THRESHOLD:-5.0}"
BLACK_SCREEN_STD_THRESHOLD="${BLACK_SCREEN_STD_THRESHOLD:-2.0}"
BLACK_SCREEN_MIN_FRAME_FRACTION="${BLACK_SCREEN_MIN_FRAME_FRACTION:-0.8}"
FORCE_RERUN="${FORCE_RERUN:-1}"

mkdir -p "${EVAL_OUTPUT_ROOT}" "${EVAL_LOG_ROOT}"
cd "${REPO_DIR}"

export PATH="${FASTWAM_ENV}/bin:${PATH}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/root/nas/temp_nas/FastWAM/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export HF_HOME="${HF_HOME:-/root/nas/temp_nas/FastWAM/.cache/huggingface}"
export PYTHON_BIN
export USE_TMUX=0

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
  echo "[libero] root=${FASTWAM_LIBERO_ROOT}"
  echo "[libero] config=${LIBERO_CONFIG_PATH}/config.yaml"
}

setup_fastwam_libero

latest_run_dir() {
  local task="$1"
  local task_root="${RUNS_ROOT}/${task}"
  [[ -d "${task_root}" ]] || fail "Missing run root: ${task_root}"
  find "${task_root}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

summary_is_complete() {
  local out_dir="$1"
  [[ -s "${out_dir}/summary.json" ]] || return 1
  local result_count
  result_count="$(find "${out_dir}" -type f -name 'gpu*_task*_results.json' | wc -l | tr -d ' ')"
  [[ "${result_count}" == "40" ]]
}

run_eval_job() {
  local label="$1"
  local task="$2"
  local inference_mode="$3"
  shift 3

  local run_dir ckpt stats out_dir log_file session_name
  run_dir="$(latest_run_dir "${task}")"
  ckpt="${run_dir}/checkpoints/weights/latest.pt"
  stats="${run_dir}/dataset_stats.json"
  out_dir="${EVAL_OUTPUT_ROOT}/${label}"
  log_file="${EVAL_LOG_ROOT}/${label}.log"
  session_name="libero_eval_${label}_${RUN_ID}"

  require_path "${ckpt}"
  require_path "${stats}"

  if [[ "${FORCE_RERUN}" != "1" ]] && summary_is_complete "${out_dir}"; then
    echo "[${label}] complete result exists, skip: ${out_dir}" | tee -a "${log_file}"
    return 0
  fi

  local attempt=1
  while (( attempt <= MAX_EVAL_RESTARTS )); do
    echo "[${label}] attempt=${attempt}/${MAX_EVAL_RESTARTS} task=${task} mode=${inference_mode} gpu=${GPU_ID} workers_per_gpu=${MAX_TASKS_PER_GPU}" | tee -a "${log_file}"
    rm -rf "${out_dir}"
    mkdir -p "${out_dir}"

    set +e
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    SESSION_NAME="${session_name}" \
    EXP_NAME="${label}" \
    "${PYTHON_BIN}" experiments/libero/run_libero_manager.py \
      "task=${task}" \
      "ckpt=${ckpt}" \
      "EVALUATION.dataset_stats_path=${stats}" \
      "EVALUATION.inference_mode=${inference_mode}" \
      "EVALUATION.output_dir=${out_dir}" \
      "EVALUATION.num_trials=${NUM_TRIALS}" \
      "EVALUATION.save_videos=false" \
      "EVALUATION.retry_invalid_episodes=true" \
      "EVALUATION.max_invalid_episode_retries=${MAX_INVALID_EPISODE_RETRIES}" \
      "EVALUATION.black_screen_filter=true" \
      "EVALUATION.black_screen_mean_threshold=${BLACK_SCREEN_MEAN_THRESHOLD}" \
      "EVALUATION.black_screen_std_threshold=${BLACK_SCREEN_STD_THRESHOLD}" \
      "EVALUATION.black_screen_min_frame_fraction=${BLACK_SCREEN_MIN_FRAME_FRACTION}" \
      "MULTIRUN.num_gpus=1" \
      "MULTIRUN.max_tasks_per_gpu=${MAX_TASKS_PER_GPU}" \
      "$@" >> "${log_file}" 2>&1
    local status="$?"
    set -e

    if [[ "${status}" == "0" ]] && summary_is_complete "${out_dir}"; then
      echo "[${label}] done: ${out_dir}" | tee -a "${log_file}"
      return 0
    fi

    echo "[${label}] incomplete/failed status=${status}; restarting from scratch" | tee -a "${log_file}"
    attempt=$((attempt + 1))
  done

  echo "[${label}] failed after ${MAX_EVAL_RESTARTS} attempts; see ${log_file}" | tee -a "${log_file}"
  return 1
}

write_combined_summary() {
  "${PYTHON_BIN}" - "${EVAL_OUTPUT_ROOT}" <<'PY_SUMMARY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
labels = [
    "unified_shared_wo",
    "unified_shared_w",
    "unified_two_action_wo",
    "unified_two_action_w",
]
rows = []
combined = {}
for label in labels:
    summary_path = root / label / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    task_results = summary.get("task_results", {})
    if len(task_results) != 40:
        raise SystemExit(f"{label}: expected 40 task results, got {len(task_results)}")
    task_rates = [float(v["success_rate"]) for v in task_results.values()]
    overall_task_avg = sum(task_rates) / len(task_rates)
    suite_stats = summary.get("suite_stats", {})
    row = {"label": label, "overall_40_task_avg_success_rate": f"{overall_task_avg:.2f}"}
    for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
        stats = suite_stats.get(suite, {})
        total_trials = float(stats.get("total_trials", 0))
        total_successes = float(stats.get("total_successes", 0))
        row[f"{suite}_success_rate"] = f"{(100.0 * total_successes / total_trials) if total_trials else 0.0:.2f}"
        row[f"{suite}_tasks"] = int(stats.get("total_tasks", 0))
    rows.append(row)
    combined[label] = {
        "summary_path": str(summary_path),
        "overall_40_task_avg_success_rate": overall_task_avg,
        "suite_stats": suite_stats,
        "task_results": task_results,
    }

(root / "combined_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
with (root / "combined_summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"[combined] wrote {root / 'combined_summary.json'}")
print(f"[combined] wrote {root / 'combined_summary.csv'}")
PY_SUMMARY
}

cat <<EOF
[libero_incremental_eval_1gpu_local]
  run_id=${RUN_ID}
  output_root=${EVAL_OUTPUT_ROOT}
  log_root=${EVAL_LOG_ROOT}
  gpu_id=${GPU_ID}
  num_trials=${NUM_TRIALS}
  max_tasks_per_gpu=${MAX_TASKS_PER_GPU}
  max_eval_restarts=${MAX_EVAL_RESTARTS}
  fastwam_libero_root=${FASTWAM_LIBERO_ROOT}
  libero_config_path=${LIBERO_CONFIG_PATH}
EOF

run_eval_job "unified_shared_wo" "libero_unified_shared_2cam224_1e-4" "wo" "$@"
run_eval_job "unified_shared_w" "libero_unified_shared_2cam224_1e-4" "w" "$@"
run_eval_job "unified_two_action_wo" "libero_unified_two_action_2cam224_1e-4" "wo" "$@"
run_eval_job "unified_two_action_w" "libero_unified_two_action_2cam224_1e-4" "w" "$@"

write_combined_summary
echo "[done] all four local one-GPU eval jobs completed. Results: ${EVAL_OUTPUT_ROOT}"
