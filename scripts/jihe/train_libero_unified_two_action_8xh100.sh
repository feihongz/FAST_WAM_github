#!/usr/bin/env bash
set -euo pipefail

# Formal 8xH100 launch script for exactly one training job.
TASK_NAME="libero_unified_two_action_2cam224_1e-4"
DATASET_FAMILY="libero"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/jihe/resolve_fastwam_train_env.sh
source "${SCRIPT_DIR}/resolve_fastwam_train_env.sh"
resolve_fastwam_train_env
NPROC_PER_NODE="${NPROC_PER_NODE:-${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
MASTER_PORT="${MASTER_PORT:-29500}"
FASTWAM_STORAGE_ROOT="${FASTWAM_STORAGE_ROOT:-/root/feihong}"
FASTWAM_ASSET_ROOT="${FASTWAM_ASSET_ROOT:-${FASTWAM_STORAGE_ROOT}/FastWAM}"
# Use the feihong-local LIBERO asset copy; set LIBERO_DATA_ROOT to override it.
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-${FASTWAM_ASSET_ROOT}/datasets/libero_mujoco3.3.2}"
LIBERO_TEXT_CACHE_DIR="${LIBERO_TEXT_CACHE_DIR:-${FASTWAM_ASSET_ROOT}/data/text_embeds_cache/libero}"
ACTION_DIT_PRETRAINED_PATH="${ACTION_DIT_PRETRAINED_PATH:-${FASTWAM_ASSET_ROOT}/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
if [[ -z "${LIBERO_NORM_STATS_PATH:-}" ]]; then
  LIBERO_NORM_STATS_PRIMARY="${FASTWAM_ASSET_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json"
  LIBERO_NORM_STATS_FALLBACK="${FASTWAM_ASSET_ROOT}/formal_runs/FAST_WAM_github/libero_unified_shared_2cam224_1e-4/2026-07-01_00-44-20/dataset_stats.json"
  if [[ -e "${LIBERO_NORM_STATS_PRIMARY}" ]]; then
    LIBERO_NORM_STATS_PATH="${LIBERO_NORM_STATS_PRIMARY}"
  else
    LIBERO_NORM_STATS_PATH="${LIBERO_NORM_STATS_FALLBACK}"
  fi
fi
LIBERO_DATASET_DIRS=(
  "${LIBERO_DATA_ROOT}/libero_spatial_no_noops_lerobot"
  "${LIBERO_DATA_ROOT}/libero_object_no_noops_lerobot"
  "${LIBERO_DATA_ROOT}/libero_goal_no_noops_lerobot"
  "${LIBERO_DATA_ROOT}/libero_10_no_noops_lerobot"
)
LIBERO_DATASET_DIRS_HYDRA="${LIBERO_DATASET_DIRS[*]}"
LIBERO_DATASET_DIRS_HYDRA="${LIBERO_DATASET_DIRS_HYDRA// /,}"
FASTWAM_OUTPUT_BASE="${FASTWAM_OUTPUT_BASE:-${FASTWAM_STORAGE_ROOT}/FastWAM/formal_runs/FAST_WAM_github}"
FASTWAM_LOG_BASE="${FASTWAM_LOG_BASE:-${FASTWAM_STORAGE_ROOT}/FastWAM/formal_logs/FAST_WAM_github}"
OUTPUT_DIR="${OUTPUT_DIR:-${FASTWAM_OUTPUT_BASE}/${TASK_NAME}/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${FASTWAM_LOG_BASE}/${TASK_NAME}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_ID}.log}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
CHECKPOINT_KEEP_LAST="${CHECKPOINT_KEEP_LAST:-5}"
# W&B env file follows the StarVLA stage2 launch style. It may provide WANDB_API_KEY,
# WANDB_MODE, WANDB_ENTITY, or WANDB_PROJECT. For FAST-WAM, shell overrides win,
# and the default entity/project stay smap/fast-wam-formal even if the shared env file has StarVLA values.
WANDB_ENV_FILE="${WANDB_ENV_FILE:-${FASTWAM_ASSET_ROOT}/secrets/wandb.env}"
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
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-${FASTWAM_ASSET_ROOT}/secrets/wandb_api_key}"
WANDB_NETRC_PATH="${WANDB_NETRC_PATH:-/root/.netrc}"
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

cd /root/feihong/FAST_WAM_github
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${OUTPUT_DIR}/wandb" "${OUTPUT_DIR}/diagnostics"
exec > >(tee -a "${LOG_FILE}") 2>&1

export PATH="${FASTWAM_ENV}/bin:${PATH}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${FASTWAM_ASSET_ROOT}/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export HF_HOME="${HF_HOME:-${FASTWAM_ASSET_ROOT}/.cache/huggingface}"
export WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}/wandb}"
export WANDB_MODE

fail() { echo "[error] $*" >&2; exit 1; }
declare -a PREFLIGHT_ERRORS=()
preflight_require_path() {
  [[ -e "$1" ]] || PREFLIGHT_ERRORS+=("Missing required path: $1")
}
preflight_finish() {
  if (( ${#PREFLIGHT_ERRORS[@]} == 0 )); then
    echo "[preflight] all required assets and credentials are available"
    return 0
  fi
  echo "[error] Preflight found ${#PREFLIGHT_ERRORS[@]} missing requirement(s):" >&2
  for error in "${PREFLIGHT_ERRORS[@]}"; do
    echo "  - ${error}" >&2
  done
  exit 1
}

DIAG_FILE="${OUTPUT_DIR}/diagnostics/runtime_monitor.log"
DIAG_INTERVAL="${FASTWAM_DIAG_INTERVAL:-15}"
MONITOR_PID=""

diag_snapshot() {
  local label="${1:-snapshot}"
  {
    echo "[diag][${label}] time=$(date -Is) pid=$$ ppid=$PPID host=$(hostname)"
    echo "[diag][${label}] output_dir=${OUTPUT_DIR} log_file=${LOG_FILE}"
    if [[ -f /sys/fs/cgroup/memory.current ]]; then echo "[diag][${label}] memory.current=$(cat /sys/fs/cgroup/memory.current)"; fi
    if [[ -f /sys/fs/cgroup/memory.max ]]; then echo "[diag][${label}] memory.max=$(cat /sys/fs/cgroup/memory.max)"; fi
    if [[ -f /sys/fs/cgroup/memory.events ]]; then sed "s/^/[diag][${label}] memory.events /" /sys/fs/cgroup/memory.events; fi
    if [[ -f /proc/pressure/memory ]]; then sed "s/^/[diag][${label}] psi.memory /" /proc/pressure/memory; fi
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits 2>/dev/null | sed "s/^/[diag][${label}] gpu /" || true
      nvidia-smi 2>/dev/null | sed "s/^/[diag][${label}] nvidia-smi /" || true
    fi
    ps -eo pid,ppid,stat,rss,vsz,pcpu,pmem,cmd --sort=-rss | head -30 | sed "s/^/[diag][${label}] ps /" || true
    df -h "${OUTPUT_DIR}" | sed "s/^/[diag][${label}] df /" || true
    echo "[diag][${label}] ---"
  } >> "${DIAG_FILE}" 2>&1
}

start_diag_monitor() {
  [[ "${FASTWAM_DIAG_DISABLE:-0}" == "1" ]] && return 0
  diag_snapshot "start"
  (
    while true; do
      sleep "${DIAG_INTERVAL}" || exit 0
      diag_snapshot "heartbeat"
    done
  ) &
  MONITOR_PID="$!"
  echo "[diag] runtime monitor pid=${MONITOR_PID} file=${DIAG_FILE} interval=${DIAG_INTERVAL}s"
}

stop_diag_monitor() {
  if [[ -n "${MONITOR_PID:-}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
}

finish_diag() {
  local status="$?"
  diag_snapshot "exit_status_${status}"
  stop_diag_monitor
  echo "[diag] script exit_status=${status} at $(date -Is); monitor=${DIAG_FILE}"
}

on_term() {
  local sig="$1"
  echo "[diag] received ${sig} at $(date -Is)" | tee -a "${DIAG_FILE}" >&2
  diag_snapshot "signal_${sig}"
  exit 143
}

trap finish_diag EXIT
trap 'on_term SIGTERM' TERM
trap 'on_term SIGINT' INT

if [[ "${WANDB_MODE}" == "online" ]]; then
  if [[ -z "${WANDB_API_KEY:-}" && -f "${WANDB_API_KEY_FILE}" ]]; then
    WANDB_API_KEY="$(tr -d '[:space:]' < "${WANDB_API_KEY_FILE}")"
  fi
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_API_KEY
  elif [[ -r "${WANDB_NETRC_PATH}" ]] && grep -Eq "^[[:space:]]*machine[[:space:]]+api\.wandb\.ai([[:space:]]|$)" "${WANDB_NETRC_PATH}"; then
    echo "[wandb] using credentials from WANDB_NETRC_PATH=${WANDB_NETRC_PATH}"
  else
    PREFLIGHT_ERRORS+=("WANDB_MODE=online requires WANDB_API_KEY, WANDB_ENV_FILE=${WANDB_ENV_FILE}, WANDB_API_KEY_FILE=${WANDB_API_KEY_FILE}, or api.wandb.ai credentials in WANDB_NETRC_PATH=${WANDB_NETRC_PATH}")
  fi
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

preflight_require_path "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors"
preflight_require_path "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors"
preflight_require_path "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"
preflight_require_path "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"
preflight_require_path "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
preflight_require_path "${ACTION_DIT_PRETRAINED_PATH}"

if [[ "${DATASET_FAMILY}" == "libero" ]]; then
  preflight_require_path "${LIBERO_DATA_ROOT}/libero_spatial_no_noops_lerobot/meta/tasks.jsonl"
  preflight_require_path "${LIBERO_DATA_ROOT}/libero_object_no_noops_lerobot/meta/tasks.jsonl"
  preflight_require_path "${LIBERO_DATA_ROOT}/libero_goal_no_noops_lerobot/meta/tasks.jsonl"
  preflight_require_path "${LIBERO_DATA_ROOT}/libero_10_no_noops_lerobot/meta/tasks.jsonl"
  preflight_require_path "${LIBERO_NORM_STATS_PATH}"
  resolved_data="$(readlink -m "${LIBERO_DATA_ROOT}")"
  echo "[data] FAST-WAM LIBERO source=${resolved_data} (override with LIBERO_DATA_ROOT)"
  cache_count="$(find -L data/text_embeds_cache/libero -type f -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')"
  if (( cache_count < 40 )); then
    PREFLIGHT_ERRORS+=("FAST-WAM LIBERO text cache incomplete: ${cache_count}/40 at /root/feihong/FAST_WAM_github/data/text_embeds_cache/libero")
  fi
else
  preflight_require_path "data/robotwin2.0/robotwin2.0/meta/info.json"
  preflight_require_path "data/robotwin2.0/robotwin2.0/meta/tasks.jsonl"
  preflight_require_path "data/robotwin2.0/dataset_stats.json"
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

preflight_finish

"${FASTWAM_ENV}/bin/python" -c 'import torch, fastwam; print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} devices={torch.cuda.device_count()}")'

HYDRA_ARGS=(
  "task=${TASK_NAME}"
  "output_dir=${OUTPUT_DIR}"
  "batch_size=${PER_GPU_BATCH_SIZE}"
  "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
  "num_workers=${NUM_WORKERS}"
  "num_epochs=10"
  "max_steps=null"
  "log_every=10"
  "save_every=${SAVE_EVERY}"
  "checkpoint_keep_last=${CHECKPOINT_KEEP_LAST}"
  "eval_every=1000"
  "save_final_checkpoint=true"
  "wandb.enabled=true"
  "wandb.mode=${WANDB_MODE}"
  "wandb.project=${WANDB_PROJECT}"
  "wandb.name=${WANDB_RUN_NAME}"
  "wandb.group=${WANDB_GROUP}"
)
HYDRA_ARGS+=("wandb.workspace=${WANDB_ENTITY}")
HYDRA_ARGS+=("data.train.dataset_dirs=[${LIBERO_DATASET_DIRS_HYDRA}]")
HYDRA_ARGS+=("+data.train.pretrained_norm_stats=${LIBERO_NORM_STATS_PATH}")
HYDRA_ARGS+=("data.train.text_embedding_cache_dir=${LIBERO_TEXT_CACHE_DIR}")
HYDRA_ARGS+=("model.action_dit_pretrained_path=${ACTION_DIT_PRETRAINED_PATH}")
HYDRA_ARGS+=("model.redirect_common_files=false")
HYDRA_ARGS+=("$@")

cat <<EOF
[formal_train]
  task=${TASK_NAME}
  dataset_family=${DATASET_FAMILY}
  data_path=${resolved_data}
  diffsynth_model_base_path=${DIFFSYNTH_MODEL_BASE_PATH}
  action_dit_pretrained_path=${ACTION_DIT_PRETRAINED_PATH}
  norm_stats_path=${LIBERO_NORM_STATS_PATH}
  redirect_common_files=false
  env=${FASTWAM_ENV}
  nproc_per_node=${NPROC_PER_NODE}
  per_gpu_batch=${PER_GPU_BATCH_SIZE}
  gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}
  num_workers=${NUM_WORKERS}
  save_every=${SAVE_EVERY}
  checkpoint_keep_last=${CHECKPOINT_KEEP_LAST}
  effective_global_batch=$((PER_GPU_BATCH_SIZE * NPROC_PER_NODE * GRADIENT_ACCUMULATION_STEPS))
  num_epochs=10
  output_dir=${OUTPUT_DIR}
  log_file=${LOG_FILE}
  wandb_entity=${WANDB_ENTITY}
  wandb_project=${WANDB_PROJECT}
  wandb_name=${WANDB_RUN_NAME}
  wandb_env_file=${WANDB_ENV_FILE}
EOF

start_diag_monitor
set +e
RUN_ID="${RUN_ID}" bash scripts/train_zero1.sh "${NPROC_PER_NODE}" "${HYDRA_ARGS[@]}"
TRAIN_STATUS="$?"
set -e
echo "[diag] train_zero1 exit_status=${TRAIN_STATUS} at $(date -Is)"
exit "${TRAIN_STATUS}"
