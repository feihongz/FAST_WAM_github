#!/usr/bin/env bash
set -euo pipefail

# Self-contained RoboTwin 2.0 1/5 Uni-Share launcher.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FASTWAM_ENV="${FASTWAM_ENV:-/root/.venvs/fastwam}"
export FASTWAM_STORAGE_ROOT="${FASTWAM_STORAGE_ROOT:-/root/feihong}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/root/feihong/FastWAM/checkpoints}"
export HF_HOME="${HF_HOME:-/root/feihong/FastWAM/.cache/huggingface}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
if [[ "${NPROC_PER_NODE}" == "auto" || "${NPROC_PER_NODE}" == "Auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d " ")"
  else
    NPROC_PER_NODE=8
  fi
  export NPROC_PER_NODE
fi
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-fast-wam-formal}"
export WANDB_ENTITY="${WANDB_ENTITY:-smap}"
export NUM_EPOCHS="${NUM_EPOCHS:-5}"
export PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
STARTUP_LOG_DIR="${FASTWAM_STORAGE_ROOT}/FastWAM/formal_logs/FAST_WAM_github/robotwin_1of5"
mkdir -p "${STARTUP_LOG_DIR}"
STARTUP_LOG="${STARTUP_LOG_DIR}/startup_$(date +%Y-%m-%d_%H-%M-%S).log"
exec > >(tee -a "${STARTUP_LOG}") 2>&1
echo "[startup] env=${FASTWAM_ENV} nproc=${NPROC_PER_NODE} cuda_visible=${CUDA_VISIBLE_DEVICES}"
if [[ ! -x "${FASTWAM_ENV}/bin/python" ]]; then echo "[startup_error] Python env not found: ${FASTWAM_ENV}"; exit 1; fi
if [[ ! -d "${FASTWAM_STORAGE_ROOT}" ]]; then echo "[startup_error] storage root not mounted: ${FASTWAM_STORAGE_ROOT}"; exit 1; fi
echo "[startup] storage=${FASTWAM_STORAGE_ROOT} model_base=${DIFFSYNTH_MODEL_BASE_PATH} log=${STARTUP_LOG}"
exec env MODEL_VARIANT=unified_shared bash "${SCRIPT_DIR}/train_robotwin_1of5_8xh100.sh" "$@"
