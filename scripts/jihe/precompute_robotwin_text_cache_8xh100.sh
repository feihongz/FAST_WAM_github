#!/usr/bin/env bash
set -euo pipefail
cd /root/code/feihong/FAST_WAM_github
FASTWAM_ENV="${FASTWAM_ENV:-/root/.venvs/fastwam-robotwin}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}}"
MASTER_PORT="${MASTER_PORT:-29510}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
LOG_BASE="${FASTWAM_LOG_BASE:-/root/nas/temp_nas/FastWAM/formal_logs/FAST_WAM_github}"
LOG_DIR="${LOG_BASE}/precompute_robotwin_text_cache"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_ID}.log}"
mkdir -p "${LOG_DIR}"

if ! [[ "${NPROC_PER_NODE}" =~ ^[0-9]+$ ]] || (( NPROC_PER_NODE < 1 )); then
  detected_gpu_count=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    detected_gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
  fi
  if ! [[ "${detected_gpu_count}" =~ ^[0-9]+$ ]] || (( detected_gpu_count < 1 )); then
    detected_gpu_count="$(${FASTWAM_ENV}/bin/python - <<'PY_GPU_COUNT'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY_GPU_COUNT
)"
  fi
  [[ "${detected_gpu_count}" =~ ^[0-9]+$ ]] && (( detected_gpu_count > 0 )) || { echo "[error] Could not resolve NPROC_PER_NODE=${NPROC_PER_NODE} to a positive GPU count" >&2; exit 1; }
  echo "[info] NPROC_PER_NODE=${NPROC_PER_NODE}; detected ${detected_gpu_count} visible GPU(s)."
  NPROC_PER_NODE="${detected_gpu_count}"
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NPROC_PER_NODE - 1)))"
fi
export CUDA_VISIBLE_DEVICES
exec > >(tee -a "${LOG_FILE}") 2>&1
export PATH="${FASTWAM_ENV}/bin:${PATH}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/root/nas/temp_nas/FastWAM/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export HF_HOME="${HF_HOME:-/root/nas/temp_nas/FastWAM/.cache/huggingface}"

echo "[precompute] FAST-WAM RoboTwin 2.0 text cache, nproc=${NPROC_PER_NODE}, log=${LOG_FILE}"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
  scripts/precompute_text_embeds.py \
  task=robotwin_unified_shared_3cam_384_1e-4 \
  +overwrite=false \
  "$@"
