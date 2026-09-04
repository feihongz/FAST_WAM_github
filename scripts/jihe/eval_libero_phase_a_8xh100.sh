#!/usr/bin/env bash
set -euo pipefail

# Formal LIBERO Phase-A sweep on one JiHe 8xH100 node.
# Fresh:  bash scripts/jihe/eval_libero_phase_a_8xh100.sh
# Resume: bash scripts/jihe/eval_libero_phase_a_8xh100.sh --resume /absolute/run/root
# Inspect: FASTWAM_DRY_RUN=1 bash scripts/jihe/eval_libero_phase_a_8xh100.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
FASTWAM_REPO_DIR="${FASTWAM_REPO_DIR:-${DEFAULT_REPO_DIR}}"
FASTWAM_ENV="${FASTWAM_ENV:-/root/.venvs/fastwam}"
PYTHON_BIN="${PYTHON_BIN:-${FASTWAM_ENV}/bin/python}"
FASTWAM_DRY_RUN="${FASTWAM_DRY_RUN:-0}"
GPU_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

fail() {
  echo "[error] $*" >&2
  exit 1
}

[[ -d "${FASTWAM_REPO_DIR}" ]] || fail "Missing repository: ${FASTWAM_REPO_DIR}"
[[ -x "${PYTHON_BIN}" ]] || fail "Missing Python: ${PYTHON_BIN}"

cd "${FASTWAM_REPO_DIR}"
GIT_SHORT="$(git rev-parse --short=7 HEAD)"
RUN_ID="${RUN_ID:-${GIT_SHORT}_$(date -u +%Y-%m-%d_%H-%M-%S)}"
DEFAULT_RUN_ROOT="/root/feihong/FastWAM/evaluate_results/phase_a/libero/${RUN_ID}"

RESUME=0
if [[ "$#" -eq 0 ]]; then
  RUN_ROOT="${RUN_ROOT:-${DEFAULT_RUN_ROOT}}"
elif [[ "$#" -eq 2 && "$1" == "--resume" ]]; then
  RESUME=1
  RUN_ROOT="$2"
else
  fail "Usage: bash $0 [--resume /absolute/run/root]"
fi
[[ "${RUN_ROOT}" == /* ]] || fail "RUN_ROOT must be absolute"

export PATH="${FASTWAM_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="/root/feihong/FastWAM/third_party/LIBERO:${FASTWAM_REPO_DIR}:${FASTWAM_REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset MUJOCO_EGL_DEVICE_ID
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/root/feihong/FastWAM/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FASTWAM_STAGE3_VAE="/root/feihong/FastWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
export FASTWAM_STAGE3_DATA_MANIFEST="/root/feihong/FastWAM/formal_runs/contracts/stage3/libero_current_273465f_1693e/libero_stage3_data_manifest.json"
export FASTWAM_STAGE3_DATA_MANIFEST_SHA256="08da49109a57b55c67f3fa4ac31fbfa44e44dd541a194a5d3420838537d0d320"
export FASTWAM_LIBERO_STATS="/root/feihong/FastWAM/formal_runs/FAST_WAM_github/libero_unified_shared_2cam224_1e-4/2026-07-01_00-44-20/dataset_stats.json"
export FASTWAM_LIBERO_ROOT="/root/feihong/FastWAM/third_party/LIBERO"

COMMAND=(
  "${PYTHON_BIN}" scripts/run_libero_phase_a.py
  --run-root "${RUN_ROOT}"
  --python-bin "${PYTHON_BIN}"
  --gpu-devices "${GPU_DEVICES}"
)
if [[ "${RESUME}" == "1" ]]; then
  COMMAND+=(--resume)
fi
if [[ "${FASTWAM_DRY_RUN}" == "1" ]]; then
  COMMAND+=(--dry-run)
else
  GPU_DEVICES="${GPU_DEVICES}" "${PYTHON_BIN}" - <<'PY'
import os
import torch

requested = [part for part in os.environ["GPU_DEVICES"].split(",") if part]
names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
print(f"[gpu] requested={requested} visible={names}")
if len(requested) != 8 or len(names) != 8 or any("H100" not in name.upper() for name in names):
    raise SystemExit(f"expected exactly 8 visible H100 GPUs, found {names}")
PY
fi

cat <<EOF
[libero-phase-a]
  protocol=static_wo -> gate_r010 -> gate_r025 -> gate_r050 -> gate_r075 -> gate_r090 -> static_w
  tasks=40 trials_per_task=50 workers=8 max_per_gpu=1
  video_steps=10 replan_steps=32 action_horizon=32 seed=42
  mode=$([[ "${RESUME}" == "1" ]] && echo resume || echo fresh)
  run_root=${RUN_ROOT}
EOF

printf '[launch]'
printf ' %q' "${COMMAND[@]}"
printf '\n'
exec "${COMMAND[@]}"
