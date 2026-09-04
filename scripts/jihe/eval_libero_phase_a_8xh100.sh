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
FASTWAM_EVAL_CODE_SNAPSHOT_ROOT="${FASTWAM_EVAL_CODE_SNAPSHOT_ROOT:-/root/feihong/FastWAM/runtime/source_worktrees/libero_phase_a}"
FASTWAM_PHASE_A_SNAPSHOT_ACTIVE="${FASTWAM_PHASE_A_SNAPSHOT_ACTIVE:-0}"

fail() {
  echo "[error] $*" >&2
  exit 1
}

[[ -d "${FASTWAM_REPO_DIR}" ]] || fail "Missing repository: ${FASTWAM_REPO_DIR}"
[[ -x "${PYTHON_BIN}" ]] || fail "Missing Python: ${PYTHON_BIN}"

# Formal evaluation must never consume concurrent, uncommitted development.
# Bootstrap into a persistent detached worktree for the exact committed
# revision, then run the launcher from that committed snapshot.  The absolute
# snapshot path is deterministic because it is frozen into run_manifest.json
# and must remain identical across --resume invocations.
if [[ "${FASTWAM_PHASE_A_SNAPSHOT_ACTIVE}" != "1" ]]; then
  command -v git >/dev/null || fail "git is required"
  command -v flock >/dev/null || fail "flock is required"

  EVAL_COMMIT="$(git -C "${FASTWAM_REPO_DIR}" rev-parse HEAD)"
  RECORDED_REPO_DIR=""
  if [[ "$#" -eq 2 && "$1" == "--resume" ]]; then
    [[ "$2" == /* ]] || fail "RUN_ROOT must be absolute"
    RUN_MANIFEST="$2/run_manifest.json"
    [[ -f "${RUN_MANIFEST}" ]] || fail "Missing resume manifest: ${RUN_MANIFEST}"
    RESUME_METADATA="$("${PYTHON_BIN}" - "${RUN_MANIFEST}" <<'PY'
import json
import re
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as stream:
    manifest = json.load(stream)
spec = manifest.get("experiment_spec")
if not isinstance(spec, dict):
    raise SystemExit(f"invalid experiment_spec in {path}")
identity = spec.get("git_identity")
if not isinstance(identity, dict):
    raise SystemExit(f"invalid git_identity in {path}")
commit = identity.get("commit")
repo_root = spec.get("repo_root")
if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
    raise SystemExit(f"invalid git commit in {path}: {commit!r}")
if (
    not isinstance(repo_root, str)
    or not repo_root.startswith("/")
    or "\t" in repo_root
    or "\n" in repo_root
):
    raise SystemExit(f"invalid repo_root in {path}: {repo_root!r}")
print(f"{commit}\t{repo_root}")
PY
)"
    IFS=$'\t' read -r EVAL_COMMIT RECORDED_REPO_DIR <<<"${RESUME_METADATA}"
  elif [[ "$#" -ne 0 ]]; then
    fail "Usage: bash $0 [--resume /absolute/run/root]"
  fi

  [[ "${EVAL_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "Invalid evaluation commit: ${EVAL_COMMIT}"
  git -C "${FASTWAM_REPO_DIR}" cat-file -e "${EVAL_COMMIT}^{commit}" || \
    fail "Evaluation commit is unavailable: ${EVAL_COMMIT}"
  SNAPSHOT_REPO_DIR="${FASTWAM_EVAL_CODE_SNAPSHOT_ROOT}/${EVAL_COMMIT}"
  if [[ -n "${RECORDED_REPO_DIR}" && "${RECORDED_REPO_DIR}" != "${SNAPSHOT_REPO_DIR}" ]]; then
    fail "Resume source path mismatch: recorded=${RECORDED_REPO_DIR} expected=${SNAPSHOT_REPO_DIR}"
  fi

  mkdir -p "${FASTWAM_EVAL_CODE_SNAPSHOT_ROOT}"
  exec 9>"${FASTWAM_EVAL_CODE_SNAPSHOT_ROOT}/.prepare.lock"
  flock 9
  if [[ ! -e "${SNAPSHOT_REPO_DIR}/.git" ]]; then
    [[ ! -e "${SNAPSHOT_REPO_DIR}" ]] || \
      fail "Incomplete code snapshot already exists: ${SNAPSHOT_REPO_DIR}"
    git -C "${FASTWAM_REPO_DIR}" worktree add --detach \
      "${SNAPSHOT_REPO_DIR}" "${EVAL_COMMIT}"
  fi
  SNAPSHOT_COMMIT="$(git -C "${SNAPSHOT_REPO_DIR}" rev-parse HEAD)"
  [[ "${SNAPSHOT_COMMIT}" == "${EVAL_COMMIT}" ]] || \
    fail "Code snapshot commit mismatch: expected=${EVAL_COMMIT} actual=${SNAPSHOT_COMMIT}"
  SNAPSHOT_TRACKED_STATUS="$(git -C "${SNAPSHOT_REPO_DIR}" status --porcelain --untracked-files=no)"
  SNAPSHOT_UNTRACKED_SOURCE="$(git -C "${SNAPSHOT_REPO_DIR}" ls-files --others --exclude-standard -- src configs scripts experiments tests)"
  [[ -z "${SNAPSHOT_TRACKED_STATUS}" && -z "${SNAPSHOT_UNTRACKED_SOURCE}" ]] || \
    fail "Managed code snapshot is dirty: ${SNAPSHOT_REPO_DIR}"
  flock -u 9
  exec 9>&-

  echo "[source-snapshot] source=${FASTWAM_REPO_DIR} commit=${EVAL_COMMIT} repo=${SNAPSHOT_REPO_DIR}"
  exec env \
    FASTWAM_PHASE_A_SNAPSHOT_ACTIVE=1 \
    FASTWAM_REPO_DIR="${SNAPSHOT_REPO_DIR}" \
    bash "${SNAPSHOT_REPO_DIR}/scripts/jihe/eval_libero_phase_a_8xh100.sh" "$@"
fi

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
