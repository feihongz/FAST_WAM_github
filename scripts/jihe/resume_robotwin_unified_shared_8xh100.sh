#!/usr/bin/env bash
set -euo pipefail

TASK_NAME="robotwin_unified_shared_3cam_384_1e-4"
RUN_ID="${RUN_ID:-2026-07-01_00-51-30}"
PROJECT_ROOT="${PROJECT_ROOT:-/root/feihong/FAST_WAM_github}"
FASTWAM_STORAGE_ROOT="${FASTWAM_STORAGE_ROOT:-/root/feihong}"
DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/root/feihong/FastWAM/checkpoints}"
FASTWAM_ENV="${FASTWAM_ENV:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

RESUME_STATE="${RESUME_STATE:-${FASTWAM_STORAGE_ROOT}/FastWAM/formal_runs/FAST_WAM_github/${TASK_NAME}/${RUN_ID}/checkpoints/state/latest}"
STARTUP_LOG_DIR="${FASTWAM_STORAGE_ROOT}/FastWAM/formal_logs/FAST_WAM_github/${TASK_NAME}"
mkdir -p "${STARTUP_LOG_DIR}"
STARTUP_LOG="${STARTUP_LOG_DIR}/${RUN_ID}.resume_startup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${STARTUP_LOG}") 2>&1

require_path() {
  [[ -e "$1" ]] || {
    echo "[resume_error] missing required path: $1" >&2
    exit 1
  }
}

echo "[resume_startup] task=${TASK_NAME}"
echo "[resume_startup] run_id=${RUN_ID}"
echo "[resume_startup] project_root=${PROJECT_ROOT}"
echo "[resume_startup] fastwam_env_requested=${FASTWAM_ENV:-<auto>}"
echo "[resume_startup] resume_state=${RESUME_STATE}"
echo "[resume_startup] diffsynth_model_base_path=${DIFFSYNTH_MODEL_BASE_PATH}"
echo "[resume_startup] startup_log=${STARTUP_LOG}"
echo "[resume_startup] nproc_per_node=${NPROC_PER_NODE}"
echo "[resume_startup] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
hostname || true
date
df -h "${FASTWAM_STORAGE_ROOT}" || true

require_path "${PROJECT_ROOT}"
require_path "${PROJECT_ROOT}/scripts/jihe/train_robotwin_unified_shared_8xh100.sh"
require_path "${RESUME_STATE}/trainer_state.json"
require_path "${RESUME_STATE}/scheduler.bin"
require_path "${RESUME_STATE}/pytorch_model"
require_path "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors"
require_path "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors"
require_path "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"
require_path "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"
require_path "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"

resolve_fastwam_env() {
  local candidate
  local candidates=()
  [[ -n "${FASTWAM_ENV:-}" ]] && candidates+=("${FASTWAM_ENV}")
  candidates+=(
    "/root/.venvs/fastwam-robotwin"
    "/root/.venvs/fastwam"
    "/root/feihong/FastWAM/uv_envs/fastwam-robotwin"
    "/root/feihong/FastWAM/uv_envs/fastwam"
    "${PROJECT_ROOT}/.venv"
  )

  for candidate in "${candidates[@]}"; do
    echo "[resume_startup] checking_fastwam_env=${candidate}"
    if [[ -x "${candidate}/bin/python" ]]; then
      FASTWAM_ENV="${candidate}"
      echo "[resume_startup] fastwam_env_resolved=${FASTWAM_ENV}"
      "${FASTWAM_ENV}/bin/python" -c 'import sys, torch, accelerate, deepspeed; print(f"[resume_startup] python={sys.executable} torch={torch.__version__} accelerate={accelerate.__version__} deepspeed={deepspeed.__version__}")'
      return 0
    fi
  done

  echo "[resume_error] could not find a usable FastWAM env with bin/python" >&2
  echo "[resume_error] set FASTWAM_ENV=/path/to/env and retry" >&2
  exit 1
}

resolve_fastwam_env

export RUN_ID FASTWAM_ENV NPROC_PER_NODE CUDA_VISIBLE_DEVICES FASTWAM_STORAGE_ROOT DIFFSYNTH_MODEL_BASE_PATH
export PATH="${FASTWAM_ENV}/bin:${PATH}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export HF_HOME="${HF_HOME:-${FASTWAM_STORAGE_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${FASTWAM_STORAGE_ROOT}/.cache/huggingface/datasets}"
export TMPDIR="${TMPDIR:-${FASTWAM_STORAGE_ROOT}/FastWAM/tmp}"
export WANDB_DIR="${WANDB_DIR:-${FASTWAM_STORAGE_ROOT}/FastWAM/formal_runs/FAST_WAM_github/robotwin_unified_shared_3cam_384_1e-4/${RUN_ID}/wandb}"

WANDB_ENV_FILE="${WANDB_ENV_FILE:-/root/feihong/FastWAM/secrets/wandb.env}"
USER_WANDB_API_KEY="${WANDB_API_KEY:-}"
USER_WANDB_MODE="${WANDB_MODE:-}"
USER_WANDB_PROJECT="${WANDB_PROJECT:-}"
USER_WANDB_ENTITY="${WANDB_ENTITY:-}"
USER_WANDB_GROUP="${WANDB_GROUP:-}"
USER_WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
if [[ -f "${WANDB_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${WANDB_ENV_FILE}"
fi
if [[ -n "${USER_WANDB_API_KEY}" ]]; then
  WANDB_API_KEY="${USER_WANDB_API_KEY}"
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
fi
WANDB_PROJECT="${USER_WANDB_PROJECT:-${FASTWAM_WANDB_PROJECT:-fast-wam-formal}}"
WANDB_MODE="${USER_WANDB_MODE:-${WANDB_MODE:-online}}"
WANDB_ENTITY="${USER_WANDB_ENTITY:-${FASTWAM_WANDB_ENTITY:-smap}}"
WANDB_GROUP="${USER_WANDB_GROUP:-${WANDB_GROUP:-robotwin_unified_shared_3cam_384_1e-4}}"
WANDB_RUN_NAME="${USER_WANDB_RUN_NAME:-${WANDB_RUN_NAME:-robotwin_unified_shared_3cam_384_1e-4_${RUN_ID}}}"
export WANDB_PROJECT WANDB_MODE WANDB_ENTITY WANDB_GROUP WANDB_NAME="${WANDB_RUN_NAME}" WANDB_DIR

cd "${PROJECT_ROOT}"
mkdir -p "${WANDB_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TMPDIR}"
echo "[resume_startup] hf_home=${HF_HOME}"
echo "[resume_startup] hf_datasets_cache=${HF_DATASETS_CACHE}"
echo "[resume_startup] tmpdir=${TMPDIR}"

HYDRA_ARGS=(
  "task=robotwin_unified_shared_3cam_384_1e-4"
  "output_dir=${FASTWAM_STORAGE_ROOT}/FastWAM/formal_runs/FAST_WAM_github/robotwin_unified_shared_3cam_384_1e-4/${RUN_ID}"
  "batch_size=4"
  "gradient_accumulation_steps=4"
  "num_epochs=5"
  "max_steps=null"
  "log_every=10"
  "save_every=1000"
  "checkpoint_keep_last=5"
  "eval_every=500"
  "save_final_checkpoint=true"
  "wandb.enabled=true"
  "wandb.mode=${WANDB_MODE}"
  "wandb.project=${WANDB_PROJECT}"
  "wandb.workspace=${WANDB_ENTITY}"
  "wandb.name=${WANDB_RUN_NAME}"
  "wandb.group=${WANDB_GROUP}"
  "model.redirect_common_files=false"
  "model.skip_dit_load_from_pretrain=true"
  "resume=${RESUME_STATE}"
)

cat <<EOF
[resume_train]
  task=robotwin_unified_shared_3cam_384_1e-4
  run_id=${RUN_ID}
  output_dir=${FASTWAM_STORAGE_ROOT}/FastWAM/formal_runs/FAST_WAM_github/robotwin_unified_shared_3cam_384_1e-4/${RUN_ID}
  resume=${RESUME_STATE}
  diffsynth_model_base_path=${DIFFSYNTH_MODEL_BASE_PATH}
  fastwam_env=${FASTWAM_ENV}
  nproc_per_node=${NPROC_PER_NODE}
  wandb_entity=${WANDB_ENTITY}
  wandb_project=${WANDB_PROJECT}
  wandb_name=${WANDB_RUN_NAME}
EOF

RUN_ID="${RUN_ID}" exec bash scripts/train_zero1.sh "${NPROC_PER_NODE}" "${HYDRA_ARGS[@]}" "$@"
