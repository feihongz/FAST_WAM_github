#!/usr/bin/env bash
set -euo pipefail

# One-H100 load/execute health check for the frozen LIBERO Stage 3 final
# Adapter. This intentionally runs only one two-solver-step closed-loop trial;
# it is not the standalone always-w benchmark evaluation.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
RUN_ID="${RUN_ID:-$(date -u +%Y-%m-%d_%H-%M-%S)}"

export RUN_ID
export OUTPUT_DIR="${OUTPUT_DIR:-/root/feihong/FastWAM/evaluate_results/stage3_final_health/libero/${RUN_ID}}"
export BASE_CKPT="/root/feihong/FastWAM/formal_runs/FAST_WAM_github/libero_unified_shared_2cam224_1e-4/2026-07-01_00-44-20/checkpoints/weights/latest.pt"
export BASE_SHA256="17a5588cc2b8d162219c9daf818614f614ee4a7921933a4a26c5d678111330e9"
export ADAPTER_PATH="/root/feihong/FastWAM/formal_runs/stage3/full/libero_stage3_alignment_2cam224_1e-4/2026-08-30_10-29-08/checkpoints/exports/step_030000.pt"
export ADAPTER_SHA256="cbc593bc6ce99c0249a65e5c7cef754c9a1d7ea602f81fdae2b8cb158a25858c"
export DATA_SHA256="08da49109a57b55c67f3fa4ac31fbfa44e44dd541a194a5d3420838537d0d320"
export CONTRACT_SHA256="84ee86f32912ca96fa058b02ce7997362b8350e73f4e0f4377bc8728af3e6d98"
export GLOBAL_STEP="30000"
export STATS_PATH="/root/feihong/FastWAM/formal_runs/FAST_WAM_github/libero_unified_shared_2cam224_1e-4/2026-07-01_00-44-20/dataset_stats.json"
export VAE_PATH="/root/feihong/FastWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
export TASK_SUITE="libero_spatial"
export TASK_ID="0"
export NUM_TRIALS="1"
export REPLAN_STEPS="32"
export NUM_INFERENCE_STEPS="2"
export SMOKE_HEADER="stage3-final-health-smoke"
export SMOKE_NOTE="final Adapter load/execute health only; not a formal success-rate result"

exec "${SCRIPT_DIR}/eval_libero_stage3_pilot_endpoint_1xh100.sh" "$@"
