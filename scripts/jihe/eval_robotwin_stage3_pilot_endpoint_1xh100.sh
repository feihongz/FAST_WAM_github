#!/usr/bin/env bash
set -euo pipefail

# One-H100 closed-loop connectivity smoke for the frozen RoboTwin 2.0 Stage 3
# 200-step pilot Adapter. This is not a formal success-rate evaluation.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
FASTWAM_REPO_DIR="${FASTWAM_REPO_DIR:-${DEFAULT_REPO_DIR}}"
FASTWAM_ENV="${FASTWAM_ENV:-/root/.venvs/fastwam}"
PYTHON_BIN="${PYTHON_BIN:-${FASTWAM_ENV}/bin/python}"
RUN_ID="${RUN_ID:-$(date -u +%Y-%m-%d_%H-%M-%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/feihong/FastWAM/evaluate_results/stage3_pilot_endpoint/robotwin/${RUN_ID}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/launch.log}"
FASTWAM_DRY_RUN="${FASTWAM_DRY_RUN:-0}"

BASE_CKPT="${BASE_CKPT:-/root/feihong/FastWAM/formal_runs/FAST_WAM_github/robotwin_unified_shared_3cam_384_1e-4/2026-07-01_00-51-30/checkpoints/weights/latest.pt}"
BASE_SHA256="${BASE_SHA256:-368a99ca9575a78d01f4cdcdee8820ec74d30c4528cf7aff07b83361a17cbbda}"
ADAPTER_PATH="${ADAPTER_PATH:-/root/feihong/FastWAM/formal_runs/pilots/stage3/robotwin_stage3_alignment_3cam384_1e-4/2026-08-29_17-00-07/checkpoints/exports/step_000200.pt}"
ADAPTER_SHA256="${ADAPTER_SHA256:-e5d984edb0bab0cb29c97b5bf484b882294d4430d7b04490d972180a0ecd2780}"
DATA_SHA256="${DATA_SHA256:-1190b75b1ef19a7abd949bdff5679da59afa7e51a043eeb43663cf2c4495173c}"
CONTRACT_SHA256="${CONTRACT_SHA256:-e9c18c334bf7863039c4a14e4f2db6c7d344688a9c1e047de75054399e1283e7}"
GLOBAL_STEP="${GLOBAL_STEP:-200}"
STATS_PATH="${STATS_PATH:-/root/feihong/FastWAM/datasets/robotwin2.0/dataset_stats.json}"
VAE_PATH="${VAE_PATH:-/root/feihong/FastWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${FASTWAM_REPO_DIR}/third_party/RoboTwin}"

TASK_NAME="${TASK_NAME:-click_alarmclock}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
NUM_EPISODES="${NUM_EPISODES:-1}"
REPLAN_STEPS="${REPLAN_STEPS:-32}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-2}"

fail() {
  echo "[error] $*" >&2
  exit 1
}

print_command() {
  printf "[launch]"
  printf " %q" "$@"
  printf "\n"
}

[[ "${OUTPUT_DIR}" == /* ]] || fail "OUTPUT_DIR must be absolute"
[[ "${NUM_EPISODES}" =~ ^[1-9][0-9]*$ ]] || fail "NUM_EPISODES must be positive"
[[ "${GLOBAL_STEP}" =~ ^[0-9]+$ ]] || fail "GLOBAL_STEP must be non-negative"
for override in "$@"; do
  case "${override}" in
    task=*|ckpt=*|gpu_id=*|EVALUATION.output_dir=*|EVALUATION.stage3_*=*|EVALUATION.inference_mode=*)
      fail "Locked pilot endpoint setting cannot be overridden: ${override}"
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATH="${FASTWAM_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${FASTWAM_REPO_DIR}:${FASTWAM_REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/root/feihong/FastWAM/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD="true"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export FASTWAM_ROBOTWIN_STAGE3_VAE="${VAE_PATH}"

COMMAND=(
  "${PYTHON_BIN}" experiments/robotwin/eval_robotwin_single.py
  "task=robotwin_stage3_alignment_3cam384_1e-4"
  "ckpt=${BASE_CKPT}"
  "gpu_id=0"
  "EVALUATION.robotwin_root=${ROBOTWIN_ROOT}"
  "EVALUATION.task_name=${TASK_NAME}"
  "EVALUATION.task_config=${TASK_CONFIG}"
  "EVALUATION.eval_num_episodes=${NUM_EPISODES}"
  "EVALUATION.output_dir=${OUTPUT_DIR}"
  "EVALUATION.dataset_stats_path=${STATS_PATH}"
  "EVALUATION.stage3_adapter_path=${ADAPTER_PATH}"
  "EVALUATION.stage3_adapter_sha256=${ADAPTER_SHA256}"
  "EVALUATION.stage3_base_sha256=${BASE_SHA256}"
  "EVALUATION.stage3_data_manifest_sha256=${DATA_SHA256}"
  "EVALUATION.stage3_training_contract_sha256=${CONTRACT_SHA256}"
  "EVALUATION.stage3_global_step=${GLOBAL_STEP}"
  "EVALUATION.inference_mode=w"
  "EVALUATION.replan_steps=${REPLAN_STEPS}"
  "EVALUATION.num_inference_steps=${NUM_INFERENCE_STEPS}"
  "EVALUATION.skip_get_obs_within_replan=true"
  "$@"
)

cat <<EOF
[stage3-pilot-endpoint-smoke]
  benchmark=RoboTwin-2.0
  gpu_count=1
  adapter_step=${GLOBAL_STEP}
  task=${TASK_NAME}
  task_config=${TASK_CONFIG}
  episodes=${NUM_EPISODES}
  inference_mode=w
  output_dir=${OUTPUT_DIR}
  note=connectivity-only; not a formal success-rate result
  preflight=base/Adapter/VAE/stats SHA checks can take several minutes
EOF
print_command "${COMMAND[@]}"

if [[ "${FASTWAM_DRY_RUN}" == "1" ]]; then
  exit 0
fi

cd "${FASTWAM_REPO_DIR}"
[[ -x "${PYTHON_BIN}" ]] || fail "Missing Python: ${PYTHON_BIN}"
[[ -f "${BASE_CKPT}" ]] || fail "Missing base checkpoint: ${BASE_CKPT}"
[[ -f "${ADAPTER_PATH}" ]] || fail "Missing Adapter: ${ADAPTER_PATH}"
[[ -f "${STATS_PATH}" ]] || fail "Missing normalization stats: ${STATS_PATH}"
[[ -f "${VAE_PATH}" ]] || fail "Missing VAE: ${VAE_PATH}"
[[ -d "${ROBOTWIN_ROOT}/script" ]] || fail "Missing RoboTwin root: ${ROBOTWIN_ROOT}"
[[ -d "${ROBOTWIN_ROOT}/assets" ]] || fail "Missing RoboTwin assets: ${ROBOTWIN_ROOT}/assets"
[[ -d "${ROBOTWIN_ROOT}/checkpoints" ]] || fail "Missing RoboTwin checkpoints: ${ROBOTWIN_ROOT}/checkpoints"
[[ -f "${ROBOTWIN_ROOT}/task_config/${TASK_CONFIG}.yml" ]] || fail "Missing RoboTwin task config: ${ROBOTWIN_ROOT}/task_config/${TASK_CONFIG}.yml"
[[ ! -e "${OUTPUT_DIR}" ]] || fail "OUTPUT_DIR already exists: ${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1
"${PYTHON_BIN}" - <<'PY'
import torch

names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print(f"[gpu] visible={names}")
if len(names) != 1 or "H100" not in names[0].upper():
    raise SystemExit(f"expected exactly one visible H100, found {names}")
PY
echo "[preflight] hashing and binding base + Adapter + VAE + stats before model load"
print_command "${COMMAND[@]}"
exec "${COMMAND[@]}"
