#!/usr/bin/env bash
set -euo pipefail

# JiHe/HyperTrain: one node with exactly 8 H100s.
TASK_NAME="libero_stage3_alignment_2cam224_1e-4"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
FASTWAM_REPO_DIR="${FASTWAM_REPO_DIR:-${DEFAULT_REPO_DIR}}"
FASTWAM_ENV="${FASTWAM_ENV:-/root/.venvs/fastwam}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}}"
MASTER_PORT="${MASTER_PORT:-29531}"
RUN_ID="${RUN_ID:-$(date -u +%Y-%m-%d_%H-%M-%S)}"
FASTWAM_STORAGE_ROOT="${FASTWAM_STORAGE_ROOT:-/root/feihong}"
FASTWAM_OUTPUT_BASE="${FASTWAM_OUTPUT_BASE:-${FASTWAM_STORAGE_ROOT}/FastWAM/formal_runs/FAST_WAM_github}"
OUTPUT_DIR="${OUTPUT_DIR:-${FASTWAM_OUTPUT_BASE}/${TASK_NAME}/${RUN_ID}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/launch.log}"
RESUME_STATE="${RESUME_STATE:-}"
FASTWAM_DRY_RUN="${FASTWAM_DRY_RUN:-0}"

fail() {
  echo "[error] $*" >&2
  exit 1
}

print_command() {
  printf "[launch]"
  printf " %q" "$@"
  printf "\n"
}

[[ "${NPROC_PER_NODE}" == "8" ]] || fail "NPROC_PER_NODE must be exactly 8, got ${NPROC_PER_NODE}"
[[ "${NNODES:-1}" == "1" ]] || fail "This launcher supports one 8-GPU JiHe instance only"
[[ "${NODE_RANK:-0}" == "0" ]] || fail "NODE_RANK must be 0 for a single-node job"
[[ "${OUTPUT_DIR}" == /* ]] || fail "OUTPUT_DIR must be an absolute persistent-storage path"

for override in "$@"; do
  case "${override}" in
    task=*|output_dir=*|training.batch_size=*|training.num_workers=*|training.drop_last=*|training.gradient_accumulation_steps=*|training.mixed_precision=*|checkpoint.strict_resume=*|checkpoint.resume=*|runtime.repo_dir=*|runtime.require_clean_git=*)
      fail "Locked Stage3 setting cannot be overridden: ${override}"
      ;;
  esac
done

if [[ -n "${RESUME_STATE}" && "${RESUME_STATE}" == *.pt ]]; then
  fail "RESUME_STATE must be a complete state directory or states/LATEST, not an Adapter export"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PATH="${FASTWAM_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${FASTWAM_REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export DIFFSYNTH_MODEL_BASE_PATH="/root/feihong/FastWAM/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD="true"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

HYDRA_ARGS=(
  "task=${TASK_NAME}"
  "output_dir=${OUTPUT_DIR}"
  "training.batch_size=2"
  "training.num_workers=0"
  "training.drop_last=true"
  "training.gradient_accumulation_steps=3"
  "training.mixed_precision=bf16"
  "checkpoint.strict_resume=true"
  "runtime.repo_dir=${FASTWAM_REPO_DIR}"
  "runtime.require_clean_git=true"
)
if [[ -n "${RESUME_STATE}" ]]; then
  HYDRA_ARGS+=("checkpoint.resume=${RESUME_STATE}")
else
  HYDRA_ARGS+=("checkpoint.resume=null")
fi
HYDRA_ARGS+=("$@")

COMMAND=(
  "${FASTWAM_ENV}/bin/accelerate" launch
  --config_file scripts/accelerate_configs/accelerate_stage3_zero2.yaml
  --num_machines 1
  --machine_rank 0
  --num_processes 8
  --main_process_port "${MASTER_PORT}"
  scripts/train_stage3_alignment.py
  "${HYDRA_ARGS[@]}"
)

cat <<EOF
[stage3]
  benchmark=LIBERO
  task=${TASK_NAME}
  world_size=8
  per_rank_batch=2
  gradient_accumulation_steps=3
  global_batch=48
  zero_stage=2
  output_dir=${OUTPUT_DIR}
  resume_state=${RESUME_STATE:-null}
EOF
print_command "${COMMAND[@]}"

if [[ "${FASTWAM_DRY_RUN}" == "1" ]]; then
  exit 0
fi

cd "${FASTWAM_REPO_DIR}"
[[ -x "${FASTWAM_ENV}/bin/python" ]] || fail "Missing Python environment: ${FASTWAM_ENV}"
[[ -x "${FASTWAM_ENV}/bin/accelerate" ]] || fail "Missing accelerate: ${FASTWAM_ENV}/bin/accelerate"
[[ -f scripts/accelerate_configs/accelerate_stage3_zero2.yaml ]] || fail "Missing Stage3 Accelerate config"
[[ -f scripts/ds_configs/ds_stage3_zero2_config.json ]] || fail "Missing Stage3 ZeRO-2 config"
[[ -f scripts/jihe/ensure_torchcodec_runtime.sh ]] || fail "Missing TorchCodec runtime helper"
# shellcheck source=ensure_torchcodec_runtime.sh
source scripts/jihe/ensure_torchcodec_runtime.sh
[[ -f /root/feihong/FastWAM/formal_runs/FAST_WAM_github/libero_unified_shared_2cam224_1e-4/2026-07-01_00-44-20/checkpoints/weights/latest.pt ]] || fail "Missing locked LIBERO base checkpoint"
[[ -f /root/feihong/FastWAM/formal_runs/FAST_WAM_github/libero_unified_shared_2cam224_1e-4/2026-07-01_00-44-20/dataset_stats.json ]] || fail "Missing locked LIBERO normalization stats"
[[ -f /root/feihong/FastWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth ]] || fail "Missing locked Wan2.2 VAE"
[[ -f /root/feihong/FastWAM/formal_runs/contracts/stage3/libero_current_273465f_1693e/libero_stage3_data_manifest.json ]] || fail "Missing locked LIBERO data manifest"
if [[ -n "${RESUME_STATE}" ]]; then
  [[ -e "${RESUME_STATE}" ]] || fail "RESUME_STATE does not exist: ${RESUME_STATE}"
fi
git diff --quiet || fail "Tracked worktree is dirty"
git diff --cached --quiet || fail "Git index is dirty"
untracked_source="$(git ls-files --others --exclude-standard -- src configs scripts tests)"
[[ -z "${untracked_source}" ]] || fail "Untracked source/config/script/test files: ${untracked_source}"

"${FASTWAM_ENV}/bin/python" - <<'PY'
import torch

count = torch.cuda.device_count()
names = [torch.cuda.get_device_name(i) for i in range(count)]
print(f"[gpu] count={count} names={names}")
if count != 8:
    raise SystemExit(f"expected exactly 8 visible GPUs, found {count}")
if any("H100" not in name.upper() for name in names):
    raise SystemExit(f"expected 8 H100 GPUs, found {names}")
PY

mkdir -p "$(dirname -- "${OUTPUT_DIR}")"
mkdir "${OUTPUT_DIR}" ||
  fail "OUTPUT_DIR already exists; choose a new OUTPUT_DIR: ${OUTPUT_DIR}"
mkdir -p "$(dirname -- "${LOG_FILE}")"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[stage3] started_at=$(date -Is) repo=${FASTWAM_REPO_DIR} git=$(git rev-parse HEAD)"
nvidia-smi -L
print_command "${COMMAND[@]}"
exec "${COMMAND[@]}"
