#!/usr/bin/env bash
set -euo pipefail

# One-H100 closed-loop connectivity smoke for the frozen LIBERO Stage 3
# 200-step pilot Adapter. This is not a formal success-rate evaluation.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
FASTWAM_REPO_DIR="${FASTWAM_REPO_DIR:-${DEFAULT_REPO_DIR}}"
FASTWAM_ENV="${FASTWAM_ENV:-/root/.venvs/fastwam}"
PYTHON_BIN="${PYTHON_BIN:-${FASTWAM_ENV}/bin/python}"
RUN_ID="${RUN_ID:-$(date -u +%Y-%m-%d_%H-%M-%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/feihong/FastWAM/evaluate_results/stage3_pilot_endpoint/libero/${RUN_ID}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/launch.log}"
FASTWAM_DRY_RUN="${FASTWAM_DRY_RUN:-0}"

BASE_CKPT="${BASE_CKPT:-/root/feihong/FastWAM/formal_runs/FAST_WAM_github/libero_unified_shared_2cam224_1e-4/2026-07-01_00-44-20/checkpoints/weights/latest.pt}"
BASE_SHA256="${BASE_SHA256:-17a5588cc2b8d162219c9daf818614f614ee4a7921933a4a26c5d678111330e9}"
ADAPTER_PATH="${ADAPTER_PATH:-/root/feihong/FastWAM/formal_runs/pilots/stage3/libero_stage3_alignment_2cam224_1e-4/2026-08-30_04-08-43/checkpoints/exports/step_000200.pt}"
ADAPTER_SHA256="${ADAPTER_SHA256:-d18341299afdd21474affc2358ec5cf1d8fe34cef6f0c7b7149e6d2f97645ac5}"
DATA_SHA256="${DATA_SHA256:-08da49109a57b55c67f3fa4ac31fbfa44e44dd541a194a5d3420838537d0d320}"
CONTRACT_SHA256="${CONTRACT_SHA256:-57600680a8cc33b2cdc1a372de622bc189e34fdd23fcbfe26031c8a74a82ac23}"
GLOBAL_STEP="${GLOBAL_STEP:-200}"
STATS_PATH="${STATS_PATH:-/root/feihong/FastWAM/formal_runs/FAST_WAM_github/libero_unified_shared_2cam224_1e-4/2026-07-01_00-44-20/dataset_stats.json}"
VAE_PATH="${VAE_PATH:-/root/feihong/FastWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"

FASTWAM_LIBERO_ROOT="${FASTWAM_LIBERO_ROOT:-/root/feihong/FastWAM/third_party/LIBERO}"
FASTWAM_LIBERO_DATASETS="${FASTWAM_LIBERO_DATASETS:-/root/feihong/FastWAM/datasets/libero_mujoco3.3.2}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${OUTPUT_DIR}/libero_config}"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
TASK_ID="${TASK_ID:-0}"
NUM_TRIALS="${NUM_TRIALS:-1}"
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
[[ "${NUM_TRIALS}" =~ ^[1-9][0-9]*$ ]] || fail "NUM_TRIALS must be positive"
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
export PYTHONPATH="${FASTWAM_LIBERO_ROOT}:${FASTWAM_REPO_DIR}:${FASTWAM_REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/root/feihong/FastWAM/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD="true"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export FASTWAM_STAGE3_VAE="${VAE_PATH}"
export FASTWAM_LIBERO_ROOT
export LIBERO_CONFIG_PATH

COMMAND=(
  "${PYTHON_BIN}" experiments/libero/eval_libero_single.py
  "task=libero_stage3_alignment_2cam224_1e-4"
  "ckpt=${BASE_CKPT}"
  "gpu_id=0"
  "EVALUATION.task_suite_name=${TASK_SUITE}"
  "EVALUATION.task_id=${TASK_ID}"
  "EVALUATION.num_trials=${NUM_TRIALS}"
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
  "EVALUATION.visualize_future_video=false"
  "EVALUATION.save_videos=false"
  "$@"
)

cat <<EOF
[stage3-pilot-endpoint-smoke]
  benchmark=LIBERO
  gpu_count=1
  adapter_step=${GLOBAL_STEP}
  task=${TASK_SUITE}/${TASK_ID}
  trials=${NUM_TRIALS}
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
[[ -d "${FASTWAM_LIBERO_ROOT}/libero/libero" ]] || fail "Missing LIBERO root: ${FASTWAM_LIBERO_ROOT}"
[[ -d "${FASTWAM_LIBERO_DATASETS}" ]] || fail "Missing LIBERO datasets: ${FASTWAM_LIBERO_DATASETS}"
[[ ! -e "${OUTPUT_DIR}" ]] || fail "OUTPUT_DIR already exists: ${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}" "${LIBERO_CONFIG_PATH}"
benchmark_root="${FASTWAM_LIBERO_ROOT}/libero/libero"
{
  echo "assets: ${benchmark_root}/assets"
  echo "bddl_files: ${benchmark_root}/bddl_files"
  echo "benchmark_root: ${benchmark_root}"
  echo "datasets: ${FASTWAM_LIBERO_DATASETS}"
  echo "init_states: ${benchmark_root}/init_files"
} > "${LIBERO_CONFIG_PATH}/config.yaml"

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
