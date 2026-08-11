#!/usr/bin/env bash
set -euo pipefail

# Thin, isolated entry point for evaluating the two RoboTwin 1/5 training runs.
REPO_DIR="${REPO_DIR:-/root/feihong/FAST_WAM_github}"
FASTWAM_STORAGE_ROOT="${FASTWAM_STORAGE_ROOT:-/root/feihong}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"

export REPO_DIR FASTWAM_STORAGE_ROOT RUN_ID
export RUNS_ROOT="${RUNS_ROOT:-${FASTWAM_STORAGE_ROOT}/FastWAM/formal_runs/FAST_WAM_github/robotwin_1of5}"
export SHARED_TASK="${SHARED_TASK:-robotwin_unified_shared_3cam_384_1e-4_1of5}"
export TWO_ACTION_TASK="${TWO_ACTION_TASK:-robotwin_unified_two_action_3cam_384_1e-4_1of5}"
export EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${FASTWAM_STORAGE_ROOT}/FastWAM/evaluate_results/robotwin_incremental_1of5_8xh100/${RUN_ID}}"
export EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-${FASTWAM_STORAGE_ROOT}/FastWAM/evaluate_logs/robotwin_incremental_1of5_8xh100/${RUN_ID}}"

exec bash "${REPO_DIR}/scripts/jihe/eval_robotwin_incremental_8xh100.sh" "$@"
