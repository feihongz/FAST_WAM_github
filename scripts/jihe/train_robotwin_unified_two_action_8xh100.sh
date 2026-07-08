#!/usr/bin/env bash
set -euo pipefail

# Formal 8xH100 launch script for exactly one training job.
TASK_NAME="robotwin_unified_two_action_3cam_384_1e-4"
DATASET_FAMILY="robotwin"
FASTWAM_ENV="${FASTWAM_ENV:-/root/.venvs/fastwam-robotwin}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
MASTER_PORT="${MASTER_PORT:-29500}"
FASTWAM_STORAGE_ROOT="${FASTWAM_STORAGE_ROOT:-/root/feihong}"
FASTWAM_OUTPUT_BASE="${FASTWAM_OUTPUT_BASE:-${FASTWAM_STORAGE_ROOT}/FastWAM/formal_runs/FAST_WAM_github}"
FASTWAM_LOG_BASE="${FASTWAM_LOG_BASE:-${FASTWAM_STORAGE_ROOT}/FastWAM/formal_logs/FAST_WAM_github}"
OUTPUT_DIR="${OUTPUT_DIR:-${FASTWAM_OUTPUT_BASE}/${TASK_NAME}/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${FASTWAM_LOG_BASE}/${TASK_NAME}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_ID}.log}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
CHECKPOINT_KEEP_LAST="${CHECKPOINT_KEEP_LAST:-5}"
# W&B env file follows the StarVLA stage2 launch style. It may provide WANDB_API_KEY,
# WANDB_MODE, WANDB_ENTITY, or WANDB_PROJECT. For FAST-WAM, shell overrides win,
# and the default entity/project stay smap/fast-wam-formal even if the shared env file has StarVLA values.
WANDB_ENV_FILE="${WANDB_ENV_FILE:-/root/nas/zian/.secrets/wandb.env}"
USER_WANDB_MODE="${WANDB_MODE:-}"
USER_WANDB_PROJECT="${WANDB_PROJECT:-}"
USER_WANDB_ENTITY="${WANDB_ENTITY:-}"
USER_WANDB_GROUP="${WANDB_GROUP:-}"
USER_WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
if [[ -f "${WANDB_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${WANDB_ENV_FILE}"
fi
WANDB_PROJECT="${USER_WANDB_PROJECT:-${FASTWAM_WANDB_PROJECT:-fast-wam-formal}}"
WANDB_MODE="${USER_WANDB_MODE:-${WANDB_MODE:-online}}"
WANDB_ENTITY="${USER_WANDB_ENTITY:-${FASTWAM_WANDB_ENTITY:-smap}}"
WANDB_RUN_NAME="${USER_WANDB_RUN_NAME:-${WANDB_RUN_NAME:-${TASK_NAME}_${RUN_ID}}}"
WANDB_GROUP="${USER_WANDB_GROUP:-${WANDB_GROUP:-${TASK_NAME}}}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/root/nas/temp_nas/FastWAM/secrets/wandb_api_key}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

if [[ ! -d "${FASTWAM_STORAGE_ROOT}" ]]; then
  echo "[error] FASTWAM_STORAGE_ROOT does not exist: ${FASTWAM_STORAGE_ROOT}" >&2
  echo "[error] Refusing to create it automatically because training outputs are too large for the system disk." >&2
  exit 1
fi
storage_total_kb="$(df -Pk "${FASTWAM_STORAGE_ROOT}" | awk 'NR == 2 {print $2}')"
if ! [[ "${storage_total_kb}" =~ ^[0-9]+$ ]] || (( storage_total_kb < 1000000000 )); then
  echo "[error] FASTWAM_STORAGE_ROOT=${FASTWAM_STORAGE_ROOT} does not look like the expected large disk: total_kb=${storage_total_kb}" >&2
  exit 1
fi

cd /root/code/feihong/FAST_WAM_github
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${OUTPUT_DIR}/wandb"
exec > >(tee -a "${LOG_FILE}") 2>&1

export PATH="${FASTWAM_ENV}/bin:${PATH}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/root/nas/temp_nas/FastWAM/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export HF_HOME="${HF_HOME:-/root/nas/temp_nas/FastWAM/.cache/huggingface}"
export WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}/wandb}"
export WANDB_MODE

fail() { echo "[error] $*" >&2; exit 1; }
require_path() { [[ -e "$1" ]] || fail "Missing required path: $1"; }

if [[ "${WANDB_MODE}" == "online" ]]; then
  if [[ -z "${WANDB_API_KEY:-}" && -f "${WANDB_API_KEY_FILE}" ]]; then
    WANDB_API_KEY="$(tr -d '[:space:]' < "${WANDB_API_KEY_FILE}")"
  fi
  [[ -n "${WANDB_API_KEY:-}" ]] || fail "WANDB_MODE=online requires WANDB_API_KEY, WANDB_ENV_FILE=${WANDB_ENV_FILE}, or readable WANDB_API_KEY_FILE=${WANDB_API_KEY_FILE}"
  export WANDB_API_KEY
fi
export WANDB_MODE WANDB_PROJECT WANDB_ENTITY WANDB_GROUP
[[ -n "${WANDB_RUN_NAME:-}" ]] && export WANDB_NAME="${WANDB_RUN_NAME}"
echo "[wandb] mode=${WANDB_MODE} entity=${WANDB_ENTITY} project=${WANDB_PROJECT} group=${WANDB_GROUP} run=${WANDB_RUN_NAME} env_file=${WANDB_ENV_FILE}"

[[ -x "${FASTWAM_ENV}/bin/python" ]] || fail "Python env not found: ${FASTWAM_ENV}"
command -v accelerate >/dev/null 2>&1 || fail "accelerate not found in ${FASTWAM_ENV}"

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
  [[ "${detected_gpu_count}" =~ ^[0-9]+$ ]] && (( detected_gpu_count > 0 )) || fail "Could not resolve NPROC_PER_NODE=${NPROC_PER_NODE} to a positive GPU count"
  echo "[info] NPROC_PER_NODE=${NPROC_PER_NODE}; detected ${detected_gpu_count} visible GPU(s)."
  NPROC_PER_NODE="${detected_gpu_count}"
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NPROC_PER_NODE - 1)))"
fi
export CUDA_VISIBLE_DEVICES

if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]]; then
  command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found"
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
  (( gpu_count >= NPROC_PER_NODE )) || fail "Need ${NPROC_PER_NODE} visible GPUs, found ${gpu_count}"
fi

require_path "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors"
require_path "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors"
require_path "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"
require_path "${DIFFSYNTH_MODEL_BASE_PATH}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors"
require_path "${DIFFSYNTH_MODEL_BASE_PATH}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
require_path "checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"

if [[ "${DATASET_FAMILY}" == "libero" ]]; then
  require_path "data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot/meta/tasks.jsonl"
  require_path "data/libero_mujoco3.3.2/libero_object_no_noops_lerobot/meta/tasks.jsonl"
  require_path "data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot/meta/tasks.jsonl"
  require_path "data/libero_mujoco3.3.2/libero_10_no_noops_lerobot/meta/tasks.jsonl"
  require_path "checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json"
  resolved_data="$(readlink -f data/libero_mujoco3.3.2)"
  echo "[data] FAST-WAM LIBERO path=${resolved_data}"
  cache_count="$(find -L data/text_embeds_cache/libero -type f -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')"
  if (( cache_count < 40 )); then
    fail "FAST-WAM LIBERO text cache incomplete: ${cache_count}/40"
  fi
else
  require_path "data/robotwin2.0/robotwin2.0/meta/info.json"
  require_path "data/robotwin2.0/robotwin2.0/meta/tasks.jsonl"
  require_path "data/robotwin2.0/dataset_stats.json"
  resolved_data="$(readlink -f data/robotwin2.0/robotwin2.0)"
  echo "[data] FAST-WAM RoboTwin 2.0 path=${resolved_data}"
  python - <<'PY_CHECK_ROBOTWIN'
import json
from pathlib import Path
p = Path('data/robotwin2.0/robotwin2.0/meta/info.json')
info = json.loads(p.read_text())
print(f"[data] RoboTwin episodes={info.get('total_episodes')} frames={info.get('total_frames')}")
if info.get('total_episodes') != 27500:
    raise SystemExit(f"Unexpected RoboTwin total_episodes: {info.get('total_episodes')}")
PY_CHECK_ROBOTWIN
  if [[ "${SKIP_TEXT_CACHE_CHECK:-0}" != "1" ]]; then
    expected="$(wc -l < data/robotwin2.0/robotwin2.0/meta/tasks.jsonl | tr -d ' ')"
    cache_count="$(find -L data/text_embeds_cache/robotwin -type f -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')"
    if (( cache_count < expected )); then
      fail "FAST-WAM RoboTwin text cache incomplete: ${cache_count}/${expected}. Run scripts/jihe/precompute_robotwin_text_cache_8xh100.sh first."
    fi
  fi
fi

"${FASTWAM_ENV}/bin/python" -c 'import torch, fastwam; print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} devices={torch.cuda.device_count()}")'

HYDRA_ARGS=(
  "task=${TASK_NAME}"
  "output_dir=${OUTPUT_DIR}"
  "batch_size=${PER_GPU_BATCH_SIZE}"
  "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
  "num_epochs=5"
  "max_steps=null"
  "log_every=10"
  "save_every=${SAVE_EVERY}"
  "checkpoint_keep_last=${CHECKPOINT_KEEP_LAST}"
  "eval_every=500"
  "save_final_checkpoint=true"
  "wandb.enabled=true"
  "wandb.mode=${WANDB_MODE}"
  "wandb.project=${WANDB_PROJECT}"
  "wandb.name=${WANDB_RUN_NAME}"
  "wandb.group=${WANDB_GROUP}"
)
HYDRA_ARGS+=("wandb.workspace=${WANDB_ENTITY}")
: # RoboTwin stats are already declared in configs/data/robotwin.yaml
HYDRA_ARGS+=("$@")

cat <<EOF
[formal_train]
  task=${TASK_NAME}
  dataset_family=${DATASET_FAMILY}
  data_path=${resolved_data}
  env=${FASTWAM_ENV}
  nproc_per_node=${NPROC_PER_NODE}
  per_gpu_batch=${PER_GPU_BATCH_SIZE}
  gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}
  save_every=${SAVE_EVERY}
  checkpoint_keep_last=${CHECKPOINT_KEEP_LAST}
  effective_global_batch=$((PER_GPU_BATCH_SIZE * NPROC_PER_NODE * GRADIENT_ACCUMULATION_STEPS))
  num_epochs=5
  output_dir=${OUTPUT_DIR}
  log_file=${LOG_FILE}
  wandb_entity=${WANDB_ENTITY}
  wandb_project=${WANDB_PROJECT}
  wandb_name=${WANDB_RUN_NAME}
  wandb_env_file=${WANDB_ENV_FILE}
EOF

RUN_ID="${RUN_ID}" bash scripts/train_zero1.sh "${NPROC_PER_NODE}" "${HYDRA_ARGS[@]}"
