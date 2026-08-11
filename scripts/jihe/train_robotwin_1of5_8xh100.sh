#!/usr/bin/env bash
set -euo pipefail

# Independent launcher for the read-only, task-balanced RoboTwin 1/5 subset.
# Select the model with MODEL_VARIANT=unified_shared or unified_two_action.
MODEL_VARIANT="${MODEL_VARIANT:-unified_shared}"
case "${MODEL_VARIANT}" in
  unified_shared)
    TASK_NAME="robotwin_unified_shared_3cam_384_1e-4_1of5"
    ;;
  unified_two_action)
    TASK_NAME="robotwin_unified_two_action_3cam_384_1e-4_1of5"
    ;;
  *)
    echo "[error] MODEL_VARIANT must be unified_shared or unified_two_action, got: ${MODEL_VARIANT}" >&2
    exit 1
    ;;
esac

FASTWAM_ENV="${FASTWAM_ENV:-/root/.venvs/fastwam-robotwin}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
MASTER_PORT="${MASTER_PORT:-29500}"
FASTWAM_STORAGE_ROOT="${FASTWAM_STORAGE_ROOT:-/root/feihong}"
FASTWAM_OUTPUT_BASE="${FASTWAM_OUTPUT_BASE:-${FASTWAM_STORAGE_ROOT}/FastWAM/formal_runs/FAST_WAM_github/robotwin_1of5}"
FASTWAM_LOG_BASE="${FASTWAM_LOG_BASE:-${FASTWAM_STORAGE_ROOT}/FastWAM/formal_logs/FAST_WAM_github/robotwin_1of5}"
OUTPUT_DIR="${OUTPUT_DIR:-${FASTWAM_OUTPUT_BASE}/${TASK_NAME}/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${FASTWAM_LOG_BASE}/${TASK_NAME}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_ID}.log}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
CHECKPOINT_KEEP_LAST="${CHECKPOINT_KEEP_LAST:-5}"
EXPECTED_TRAIN_FRAMES=1200269
EXPECTED_VAL_FRAMES=12626

WANDB_ENV_FILE="${WANDB_ENV_FILE:-/root/feihong/FastWAM/secrets/wandb.env}"
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
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/root/feihong/FastWAM/secrets/wandb_api_key}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

fail() { echo "[error] $*" >&2; exit 1; }
require_path() { [[ -e "$1" ]] || fail "Missing required path: $1"; }

if [[ ! -d "${FASTWAM_STORAGE_ROOT}" ]]; then
  fail "FASTWAM_STORAGE_ROOT does not exist: ${FASTWAM_STORAGE_ROOT}"
fi
storage_total_kb="$(df -Pk "${FASTWAM_STORAGE_ROOT}" | awk 'NR == 2 {print $2}')"
if ! [[ "${storage_total_kb}" =~ ^[0-9]+$ ]] || (( storage_total_kb < 1000000000 )); then
  fail "FASTWAM_STORAGE_ROOT=${FASTWAM_STORAGE_ROOT} does not look like the expected large disk: total_kb=${storage_total_kb}"
fi

cd /root/feihong/FAST_WAM_github
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${OUTPUT_DIR}/wandb"
exec > >(tee -a "${LOG_FILE}") 2>&1

export PATH="${FASTWAM_ENV}/bin:${PATH}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/root/feihong/FastWAM/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export HF_HOME="${HF_HOME:-/root/feihong/FastWAM/.cache/huggingface}"
export WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}/wandb}"
export MASTER_PORT

if [[ "${WANDB_MODE}" == "online" ]]; then
  if [[ -z "${WANDB_API_KEY:-}" && -f "${WANDB_API_KEY_FILE}" ]]; then
    WANDB_API_KEY="$(tr -d '[:space:]' < "${WANDB_API_KEY_FILE}")"
  fi
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_API_KEY
  elif [[ -f /root/.netrc ]]; then
    echo "[wandb] using credentials from /root/.netrc"
  else
    fail "WANDB_MODE=online requires a W&B API key or /root/.netrc"
  fi
  export WANDB_API_KEY
fi
export WANDB_MODE WANDB_PROJECT WANDB_ENTITY WANDB_GROUP
[[ -n "${WANDB_RUN_NAME:-}" ]] && export WANDB_NAME="${WANDB_RUN_NAME}"

[[ -x "${FASTWAM_ENV}/bin/python" ]] || fail "Python env not found: ${FASTWAM_ENV}"
command -v accelerate >/dev/null 2>&1 || fail "accelerate not found in ${FASTWAM_ENV}"
if ! [[ "${NPROC_PER_NODE}" =~ ^[0-9]+$ ]] || (( NPROC_PER_NODE < 1 )); then
  fail "NPROC_PER_NODE must be a positive integer, got: ${NPROC_PER_NODE}"
fi
for integer_name in PER_GPU_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS NUM_EPOCHS; do
  integer_value="${!integer_name}"
  if ! [[ "${integer_value}" =~ ^[0-9]+$ ]] || (( integer_value < 1 )); then
    fail "${integer_name} must be a positive integer, got: ${integer_value}"
  fi
done
GLOBAL_MICRO_BATCH=$((PER_GPU_BATCH_SIZE * NPROC_PER_NODE))
MICRO_STEPS_PER_EPOCH=$(((EXPECTED_TRAIN_FRAMES + GLOBAL_MICRO_BATCH - 1) / GLOBAL_MICRO_BATCH))
OPT_STEPS_PER_EPOCH=$(((MICRO_STEPS_PER_EPOCH + GRADIENT_ACCUMULATION_STEPS - 1) / GRADIENT_ACCUMULATION_STEPS))
EXPECTED_MAX_STEPS=$((OPT_STEPS_PER_EPOCH * NUM_EPOCHS))
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
require_path "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"
require_path "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
require_path "checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
require_path "data/robotwin2.0/robotwin2.0/meta/info.json"
require_path "data/robotwin2.0/robotwin2.0/meta/tasks.jsonl"
require_path "data/robotwin2.0/dataset_stats.json"

resolved_data="$(readlink -f data/robotwin2.0/robotwin2.0)"
"${FASTWAM_ENV}/bin/python" - "${OUTPUT_DIR}/subset_manifest.json" <<'PY_CHECK_SUBSET'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from fastwam.datasets.lerobot.episode_selector import GroupedStratifiedEpisodeSelector

info_path = Path("data/robotwin2.0/robotwin2.0/meta/info.json")
info = json.loads(info_path.read_text())
source_episodes = info.get("total_episodes")
source_frames = info.get("total_frames")
if source_frames != 6075103:
    raise SystemExit(f"Unexpected RoboTwin source frame count: {source_frames}")
selector = GroupedStratifiedEpisodeSelector(
    fraction=0.2,
    group_size=550,
    strata_sizes=[50, 500],
    seed=42,
    expected_total_episodes=27500,
    expected_selected_episodes=5500,
)
selected = selector(source_episodes)
shuffled = selected.copy()
np.random.default_rng(42).shuffle(shuffled)
split_index = int(len(shuffled) * 0.99)
train_episodes = shuffled[:split_index]
val_episodes = shuffled[split_index:]
if len(train_episodes) != 5445 or len(val_episodes) != 55:
    raise SystemExit(
        f"Unexpected subset split: train={len(train_episodes)} val={len(val_episodes)}"
    )
if set(train_episodes) & set(val_episodes):
    raise SystemExit("RoboTwin subset train/val episode overlap detected")

manifest_path = Path(sys.argv[1])
manifest = {
    "source_dataset": str(info_path.parent.parent),
    "source_total_episodes": source_episodes,
    "source_total_frames": source_frames,
    "selector": {
        "fraction": 0.2,
        "group_size": 550,
        "strata_sizes": [50, 500],
        "seed": 42,
    },
    "selected_episodes": selected,
    "train_episodes": train_episodes,
    "val_episodes": val_episodes,
}
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
print(
    "[data] RoboTwin source_episodes="
    f"{source_episodes} source_frames={source_frames} "
    f"selected_episodes={len(selected)} tasks=50 per_task=110 "
    "clean_per_task=10 randomized_per_task=100"
)
print(f"[data] subset_manifest={manifest_path} sha256={manifest_sha256}")
PY_CHECK_SUBSET

if [[ "${SKIP_TEXT_CACHE_CHECK:-0}" != "1" ]]; then
  expected="$(wc -l < data/robotwin2.0/robotwin2.0/meta/tasks.jsonl | tr -d ' ')"
  cache_count="$(find -L data/text_embeds_cache/robotwin -type f -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')"
  if (( cache_count < expected )); then
    fail "FAST-WAM RoboTwin text cache incomplete: ${cache_count}/${expected}"
  fi
fi

"${FASTWAM_ENV}/bin/python" -c 'import torch, fastwam; print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} devices={torch.cuda.device_count()}")'

HYDRA_ARGS=(
  "task=${TASK_NAME}"
  "output_dir=${OUTPUT_DIR}"
  "batch_size=${PER_GPU_BATCH_SIZE}"
  "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
  "num_epochs=${NUM_EPOCHS}"
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
  "wandb.workspace=${WANDB_ENTITY}"
)
HYDRA_ARGS+=("$@")

cat <<EOF
[robotwin_1of5_train]
  task=${TASK_NAME}
  model_variant=${MODEL_VARIANT}
  source_data_path=${resolved_data}
  selected_episodes=5500
  train_val_split=5445/55
  train_val_frames=${EXPECTED_TRAIN_FRAMES}/${EXPECTED_VAL_FRAMES}
  num_epochs=${NUM_EPOCHS}
  nproc_per_node=${NPROC_PER_NODE}
  per_gpu_batch=${PER_GPU_BATCH_SIZE}
  gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}
  effective_global_batch=$((PER_GPU_BATCH_SIZE * NPROC_PER_NODE * GRADIENT_ACCUMULATION_STEPS))
  expected_optimizer_steps_per_epoch=${OPT_STEPS_PER_EPOCH}
  expected_max_steps=${EXPECTED_MAX_STEPS}
  subset_manifest=${OUTPUT_DIR}/subset_manifest.json
  output_dir=${OUTPUT_DIR}
  log_file=${LOG_FILE}
  wandb_entity=${WANDB_ENTITY}
  wandb_project=${WANDB_PROJECT}
  wandb_name=${WANDB_RUN_NAME}
EOF

RUN_ID="${RUN_ID}" bash scripts/train_zero1.sh "${NPROC_PER_NODE}" "${HYDRA_ARGS[@]}"
