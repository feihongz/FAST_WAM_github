#!/usr/bin/env bash
set -euo pipefail

# Unified Shared LIBERO training with action scheduler shift=1.0.
# All other settings are inherited from the formal 8xH100 launcher.
#
# This overrides both the training and inference action shifts:
#   model.action_scheduler.train_shift=1.0
#   model.action_scheduler.infer_shift=1.0
# Video scheduler shifts remain unchanged at their task-config defaults (5.0).
# Extra arguments are forwarded to the base launcher.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/train_libero_unified_shared_8xh100.sh"
EXPERIMENT_NAME="libero_unified_shared_action_shift1_2cam224_1e-4"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
FASTWAM_STORAGE_ROOT="${FASTWAM_STORAGE_ROOT:-/root/feihong}"
FASTWAM_OUTPUT_BASE="${FASTWAM_OUTPUT_BASE:-${FASTWAM_STORAGE_ROOT}/FastWAM/formal_runs/FAST_WAM_github}"
FASTWAM_LOG_BASE="${FASTWAM_LOG_BASE:-${FASTWAM_STORAGE_ROOT}/FastWAM/formal_logs/FAST_WAM_github}"
export RUN_ID
export OUTPUT_DIR="${OUTPUT_DIR:-${FASTWAM_OUTPUT_BASE}/${EXPERIMENT_NAME}/${RUN_ID}}"
export LOG_DIR="${LOG_DIR:-${FASTWAM_LOG_BASE}/${EXPERIMENT_NAME}}"
export LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_ID}.log}"
export WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_NAME}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXPERIMENT_NAME}_${RUN_ID}}"


if [[ ! -x "${BASE_SCRIPT}" ]]; then
  echo "[error] Base launcher is missing or not executable: ${BASE_SCRIPT}" >&2
  exit 1
fi

exec bash "${BASE_SCRIPT}" \
  model.action_scheduler.train_shift=1.0 \
  model.action_scheduler.infer_shift=1.0 \
  "$@"
