#!/usr/bin/env bash
set -euo pipefail

# Run the four RoboTwin 2.0 incremental eval jobs concurrently on one 8xH100 node.
# Jobs:
#   0,1: unified_shared without video conditioning (wo)
#   2,3: unified_shared with video conditioning (w)
#   4,5: unified_two_action without video conditioning (wo)
#   6,7: unified_two_action with video conditioning (w)
#
# Each physical GPU runs MAX_TASKS_PER_GPU workers. The default is 2, so this
# script starts 16 RoboTwin workers in total.

REPO_DIR="${REPO_DIR:-/root/feihong/FAST_WAM_github}"
FASTWAM_ENV="${FASTWAM_ENV:-/root/.venvs/fastwam}"
PYTHON_BIN="${PYTHON_BIN:-${FASTWAM_ENV}/bin/python}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
FASTWAM_STORAGE_ROOT="${FASTWAM_STORAGE_ROOT:-/root/feihong}"
RUNS_ROOT="${RUNS_ROOT:-${FASTWAM_STORAGE_ROOT}/FastWAM/formal_runs/FAST_WAM_github}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${FASTWAM_STORAGE_ROOT}/FastWAM/evaluate_results/robotwin_incremental_8xh100/${RUN_ID}}"
EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-${FASTWAM_STORAGE_ROOT}/FastWAM/evaluate_logs/robotwin_incremental_8xh100/${RUN_ID}}"

SHARED_TASK="${SHARED_TASK:-robotwin_unified_shared_3cam_384_1e-4}"
TWO_ACTION_TASK="${TWO_ACTION_TASK:-robotwin_unified_two_action_3cam_384_1e-4}"
DEFAULT_SHARED_RUN_DIR="${DEFAULT_SHARED_RUN_DIR:-${RUNS_ROOT}/${SHARED_TASK}/2026-07-01_00-51-30}"
DEFAULT_TWO_ACTION_RUN_DIR="${DEFAULT_TWO_ACTION_RUN_DIR:-${RUNS_ROOT}/${TWO_ACTION_TASK}/2026-07-01_00-57-01}"
EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-100}"
REPLAN_STEPS="${REPLAN_STEPS:-24}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-2}"
MAX_EVAL_RESTARTS="${MAX_EVAL_RESTARTS:-1}"
FORCE_RERUN="${FORCE_RERUN:-1}"
DISABLE_EVAL_VIDEO_LOG="${DISABLE_EVAL_VIDEO_LOG:-1}"
EXPECTED_ROBOTWIN_TASKS="${EXPECTED_ROBOTWIN_TASKS:-50}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${EVAL_OUTPUT_ROOT}" "${EVAL_LOG_ROOT}"
cd "${REPO_DIR}"

export PATH="${FASTWAM_ENV}/bin:${PATH}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/src:${REPO_DIR}/third_party/RoboTwin:${PYTHONPATH:-}"
export PYTHON_BIN

fail() { echo "[error] $*" >&2; exit 1; }
require_path() { [[ -e "$1" ]] || fail "Missing required path: $1"; }

if [[ -z "${DIFFSYNTH_MODEL_BASE_PATH:-}" ]]; then
  for candidate in \
    /root/feihong/FastWAM/checkpoints \
    /root/feihong/FastWAM/checkpoints; do
    if [[ -e "${candidate}" ]]; then
      export DIFFSYNTH_MODEL_BASE_PATH="${candidate}"
      break
    fi
  done
fi
[[ -n "${DIFFSYNTH_MODEL_BASE_PATH:-}" ]] || fail "Set DIFFSYNTH_MODEL_BASE_PATH to the FastWAM checkpoint cache"

if [[ -z "${HF_HOME:-}" ]]; then
  for candidate in \
    /root/feihong/FastWAM/.cache/huggingface \
    /root/feihong/FastWAM/.cache/huggingface \
    /root/.cache/huggingface; do
    if [[ -e "${candidate}" ]]; then
      export HF_HOME="${candidate}"
      break
    fi
  done
fi

[[ -x "${PYTHON_BIN}" ]] || fail "Python env not found: ${PYTHON_BIN}"
require_path "${REPO_DIR}/experiments/robotwin/run_robotwin_manager.py"
require_path "${REPO_DIR}/experiments/robotwin/eval_robotwin_single.py"
require_path "${REPO_DIR}/third_party/RoboTwin/script/eval_policy.py"

if [[ "${DISABLE_EVAL_VIDEO_LOG}" == "1" ]]; then
  for cfg in \
    "${REPO_DIR}/third_party/RoboTwin/task_config/demo_clean.yml" \
    "${REPO_DIR}/third_party/RoboTwin/task_config/demo_randomized.yml"; do
    require_path "${cfg}"
    "${PYTHON_BIN}" - "${cfg}" <<'PY_CFG'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding='utf-8')
updated, count = re.subn(r'(?m)^(\s*eval_video_log\s*:\s*)\S+\s*$', r'\1false', text)
if count == 0:
    if not text.endswith('\n'):
        text += '\n'
    updated = text + 'eval_video_log: false\n'
path.write_text(updated, encoding='utf-8')
print(f"[robotwin] disabled video log in {path}")
PY_CFG
  done
fi

if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]]; then
  command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found"
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
  [[ "${gpu_count}" =~ ^[0-9]+$ ]] && (( gpu_count >= 8 )) || fail "Need 8 visible GPUs, found ${gpu_count}"
fi

latest_completed_run_dir() {
  local task="$1"
  local task_root="${RUNS_ROOT}/${task}"
  local run_dir=""
  [[ -d "${task_root}" ]] || fail "Missing run root: ${task_root}"
  while IFS= read -r candidate; do
    if [[ -f "${candidate}/checkpoints/weights/latest.pt" && -f "${candidate}/dataset_stats.json" ]]; then
      run_dir="${candidate}"
      break
    fi
  done < <(find "${task_root}" -mindepth 1 -maxdepth 1 -type d | sort -r)
  [[ -n "${run_dir}" ]] || fail "No completed run found under ${task_root} with checkpoints/weights/latest.pt and dataset_stats.json"
  printf '%s\n' "${run_dir}"
}

ckpt_tag() {
  "${PYTHON_BIN}" - "$1" <<'PY_TAG'
import sys
from pathlib import Path
p = Path(sys.argv[1]).resolve()
parts = p.parts
if "runs" in parts:
    idx = parts.index("runs")
    if idx + 2 >= len(parts):
        raise SystemExit(f"bad runs ckpt path: {p}")
    print(f"{parts[idx + 1]}_{parts[idx + 2]}")
else:
    print(p.stem)
PY_TAG
}

count_gpu_ids() {
  local csv="$1"
  local old_ifs="${IFS}"
  local ids
  IFS=',' read -r -a ids <<< "${csv}"
  IFS="${old_ifs}"
  printf '%s\n' "${#ids[@]}"
}

summary_is_complete() {
  local result_dir="$1"
  [[ -s "${result_dir}/summary.json" ]] || return 1
  "${PYTHON_BIN}" - "${result_dir}/summary.json" "${EXPECTED_ROBOTWIN_TASKS}" <<'PY_COMPLETE'
import json
import sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
expected = int(sys.argv[2])
per_task = summary.get('per_task', [])
if len(per_task) != expected:
    raise SystemExit(1)
for row in per_task:
    if row.get('clean_success_rate') is None or row.get('random_success_rate') is None:
        raise SystemExit(1)
PY_COMPLETE
}

SHARED_RUN_DIR="${SHARED_RUN_DIR:-${DEFAULT_SHARED_RUN_DIR}}"
TWO_ACTION_RUN_DIR="${TWO_ACTION_RUN_DIR:-${DEFAULT_TWO_ACTION_RUN_DIR}}"
if [[ ! -f "${SHARED_RUN_DIR}/checkpoints/weights/latest.pt" || ! -f "${SHARED_RUN_DIR}/dataset_stats.json" ]]; then
  SHARED_RUN_DIR="$(latest_completed_run_dir "${SHARED_TASK}")"
fi
if [[ ! -f "${TWO_ACTION_RUN_DIR}/checkpoints/weights/latest.pt" || ! -f "${TWO_ACTION_RUN_DIR}/dataset_stats.json" ]]; then
  TWO_ACTION_RUN_DIR="$(latest_completed_run_dir "${TWO_ACTION_TASK}")"
fi
SHARED_CKPT="${SHARED_CKPT:-${SHARED_RUN_DIR}/checkpoints/weights/latest.pt}"
TWO_ACTION_CKPT="${TWO_ACTION_CKPT:-${TWO_ACTION_RUN_DIR}/checkpoints/weights/latest.pt}"
SHARED_DATASET_STATS="${SHARED_DATASET_STATS:-${SHARED_RUN_DIR}/dataset_stats.json}"
TWO_ACTION_DATASET_STATS="${TWO_ACTION_DATASET_STATS:-${TWO_ACTION_RUN_DIR}/dataset_stats.json}"
require_path "${SHARED_CKPT}"
require_path "${TWO_ACTION_CKPT}"
require_path "${SHARED_DATASET_STATS}"
require_path "${TWO_ACTION_DATASET_STATS}"

RESULT_DIRS_DIR="${EVAL_OUTPUT_ROOT}/result_dirs"
mkdir -p "${RESULT_DIRS_DIR}"
rm -f "${RESULT_DIRS_DIR}"/*.tsv

run_eval_job() {
  local label="$1"
  local task="$2"
  local inference_mode="$3"
  local gpu_ids="$4"
  local ckpt="$5"
  local stats="$6"
  shift 6

  local gpu_count out_name tag result_dir log_file attempt status
  gpu_count="$(count_gpu_ids "${gpu_ids}")"
  out_name="${RUN_ID}_${label}"
  tag="$(ckpt_tag "${ckpt}")"
  result_dir="${REPO_DIR}/evaluate_results/robotwin/${tag}/${out_name}"
  log_file="${EVAL_LOG_ROOT}/${label}.log"

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${label}" "${task}" "${inference_mode}" "${gpu_ids}" "${ckpt}" "${result_dir}" > "${RESULT_DIRS_DIR}/${label}.tsv"

  if [[ "${FORCE_RERUN}" != "1" ]] && summary_is_complete "${result_dir}"; then
    echo "[${label}] complete result exists, skip: ${result_dir}" | tee -a "${log_file}"
    return 0
  fi

  attempt=1
  while (( attempt <= MAX_EVAL_RESTARTS )); do
    echo "[${label}] attempt=${attempt}/${MAX_EVAL_RESTARTS} task=${task} mode=${inference_mode} gpu_ids=${gpu_ids} workers=$((gpu_count * MAX_TASKS_PER_GPU)) ckpt=${ckpt}" | tee -a "${log_file}"
    if [[ "${FORCE_RERUN}" == "1" ]]; then
      rm -rf "${result_dir}"
    fi
    mkdir -p "${result_dir}"

    set +e
    ROBOTWIN_EVAL_GPU_IDS="${gpu_ids}" \
    "${PYTHON_BIN}" experiments/robotwin/run_robotwin_manager.py \
      "task=${task}" \
      "ckpt=${ckpt}" \
      "EVALUATION.dataset_stats_path=${stats}" \
      "EVALUATION.inference_mode=${inference_mode}" \
      "EVALUATION.output_dir=${EVAL_OUTPUT_ROOT}/${out_name}" \
      "EVALUATION.eval_num_episodes=${EVAL_NUM_EPISODES}" \
      "EVALUATION.replan_steps=${REPLAN_STEPS}" \
      "EVALUATION.num_inference_steps=${NUM_INFERENCE_STEPS}" \
      "MULTIRUN.num_gpus=${gpu_count}" \
      "MULTIRUN.max_tasks_per_gpu=${MAX_TASKS_PER_GPU}" \
      "$@" >> "${log_file}" 2>&1
    status="$?"
    set -e

    if [[ "${status}" == "0" ]] && summary_is_complete "${result_dir}"; then
      echo "[${label}] done: ${result_dir}" | tee -a "${log_file}"
      return 0
    fi

    echo "[${label}] incomplete/failed status=${status}; see ${log_file}" | tee -a "${log_file}"
    attempt=$((attempt + 1))
  done

  echo "[${label}] failed after ${MAX_EVAL_RESTARTS} attempt(s); see ${log_file}" | tee -a "${log_file}"
  return 1
}

write_combined_summary() {
  "${PYTHON_BIN}" - "${RESULT_DIRS_DIR}" "${EVAL_OUTPUT_ROOT}" "${EXPECTED_ROBOTWIN_TASKS}" <<'PY_SUMMARY'
import csv
import json
import sys
from pathlib import Path

result_dirs_dir = Path(sys.argv[1])
out_root = Path(sys.argv[2])
expected = int(sys.argv[3])
rows = []
combined = {}
for record_path in sorted(result_dirs_dir.glob('*.tsv')):
    with record_path.open('r', encoding='utf-8') as f:
        line = f.readline()
    if not line.strip():
        continue
    label, task, mode, gpu_ids, ckpt, result_dir = line.rstrip('\n').split('\t')
    summary_path = Path(result_dir) / 'summary.json'
    if not summary_path.exists():
        raise SystemExit(f"Missing summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    per_task = summary.get('per_task', [])
    if len(per_task) != expected:
        raise SystemExit(f"{label}: expected {expected} tasks, got {len(per_task)}")
    overall = summary.get('overall', {})
    row = {
        'label': label,
        'task': task,
        'inference_mode': mode,
        'gpu_ids': gpu_ids,
        'ckpt': ckpt,
        'result_dir': result_dir,
        'clean_mean_success_rate': overall.get('clean_mean_success_rate'),
        'random_mean_success_rate': overall.get('random_mean_success_rate'),
    }
    rows.append(row)
    combined[label] = {
        **row,
        'summary_path': str(summary_path),
        'per_task': per_task,
    }

if not rows:
    raise SystemExit('No result directories recorded')
(out_root / 'combined_summary.json').write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding='utf-8')
with (out_root / 'combined_summary.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"[combined] wrote {out_root / 'combined_summary.json'}")
print(f"[combined] wrote {out_root / 'combined_summary.csv'}")
PY_SUMMARY
}

cleanup_children() {
  local status="$?"
  if [[ "${status}" != "0" ]]; then
    jobs -pr | xargs -r kill 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup_children EXIT INT TERM

cat <<EOF
[robotwin_incremental_eval]
  run_id=${RUN_ID}
  shared_ckpt=${SHARED_CKPT}
  two_action_ckpt=${TWO_ACTION_CKPT}
  output_root=${EVAL_OUTPUT_ROOT}
  log_root=${EVAL_LOG_ROOT}
  canonical_results_root=${REPO_DIR}/evaluate_results/robotwin
  diffsynth_model_base_path=${DIFFSYNTH_MODEL_BASE_PATH}
  hf_home=${HF_HOME:-}
  eval_num_episodes=${EVAL_NUM_EPISODES}
  replan_steps=${REPLAN_STEPS}
  num_inference_steps=${NUM_INFERENCE_STEPS}
  max_tasks_per_gpu=${MAX_TASKS_PER_GPU}
  total_workers=$((8 * MAX_TASKS_PER_GPU))
  max_eval_restarts=${MAX_EVAL_RESTARTS}
  disable_eval_video_log=${DISABLE_EVAL_VIDEO_LOG}
  dry_run=${DRY_RUN}
EOF

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] command configuration validated; no eval jobs launched."
  exit 0
fi

run_eval_job "unified_shared_wo" "${SHARED_TASK}" "wo" "0,1" "${SHARED_CKPT}" "${SHARED_DATASET_STATS}" "$@" &
pids=("$!")
run_eval_job "unified_shared_w" "${SHARED_TASK}" "w" "2,3" "${SHARED_CKPT}" "${SHARED_DATASET_STATS}" "$@" &
pids+=("$!")
run_eval_job "unified_two_action_wo" "${TWO_ACTION_TASK}" "wo" "4,5" "${TWO_ACTION_CKPT}" "${TWO_ACTION_DATASET_STATS}" "$@" &
pids+=("$!")
run_eval_job "unified_two_action_w" "${TWO_ACTION_TASK}" "w" "6,7" "${TWO_ACTION_CKPT}" "${TWO_ACTION_DATASET_STATS}" "$@" &
pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [[ "${failed}" != "0" ]]; then
  fail "At least one eval job failed. Logs: ${EVAL_LOG_ROOT}"
fi

write_combined_summary
echo "[done] all four eval jobs completed. Combined results: ${EVAL_OUTPUT_ROOT}"
