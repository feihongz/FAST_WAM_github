#!/usr/bin/env bash
set -euo pipefail

# Full one-epoch acceptance smoke for the small LIBERO BinaryVideoGate.
# This is intentionally one ordinary Python process on one visible H100.

fail() {
  echo "[error] $*" >&2
  exit 1
}

print_command() {
  printf "[launch]"
  printf " %q" "$@"
  printf "\n"
}

[[ "$#" == "0" ]] ||
  fail "this one-click launcher takes no arguments; use environment variables only for output selection"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
FASTWAM_REPO_DIR="${FASTWAM_REPO_DIR:-${DEFAULT_REPO_DIR}}"
FASTWAM_ENV="${FASTWAM_ENV:-/root/.venvs/fastwam}"
FASTWAM_STORAGE_ROOT="${FASTWAM_STORAGE_ROOT:-/root/feihong}"
FASTWAM_DRY_RUN="${FASTWAM_DRY_RUN:-0}"
PYTHON_BIN="${FASTWAM_ENV}/bin/python"
RUN_ID="${RUN_ID:-$(date -u +%Y-%m-%d_%H-%M-%S)}"
VISIBLE_GPU="${FASTWAM_CUDA_VISIBLE_DEVICES:-0}"

readonly DATA_MANIFEST="/root/feihong/FastWAM/formal_runs/contracts/stage3/libero_current_273465f_1693e/libero_stage3_data_manifest.json"
readonly DATA_MANIFEST_SHA256="08da49109a57b55c67f3fa4ac31fbfa44e44dd541a194a5d3420838537d0d320"
readonly SELECTION_DIR="/root/feihong/FastWAM/formal_runs/contracts/stage2/libero_nested64_stratified_v2_426b635d"
readonly SELECTION_SHA256="426b635d637a0f3e5d31dd13612ff5ad786fd5cfe9ce27b0e8689854d9aa9e9b"
readonly COVERAGE_SHA256="d114ac25b61ab30f18185c9ea69a33d537b5196b145a8c5c3d6f6fd9d884708f"
readonly EPISODE_ASSIGNMENT_SHA256="a77efa24249dab8cfacbc228b1da341947240b36fa77d90182701c07bdcf7787"
readonly LABEL_JOB="/root/feihong/FastWAM/formal_runs/stage2/labels/libero_stage2_gate_labels_2cam224/selection_426b635d_d75c04a"
readonly LABEL_CONTRACT_SHA256="0e089ebc97b0532484a7dacf526cfc8e2c68894e637fe7d9e483fa566e46ff17"
readonly MERGED_MANIFEST="/root/feihong/FastWAM/formal_runs/stage2/merged/libero_stage2_gate_labels_2cam224/selection_426b635d_d75c04a_formal_d114ac25/manifest.json"
readonly MERGED_MANIFEST_SHA256="d6dc98a6a36c30150db30000c86d07c7a1e7d90b1dc5d1a5a60e02126c22b3e0"
readonly BASE_SHA256="17a5588cc2b8d162219c9daf818614f614ee4a7921933a4a26c5d678111330e9"
readonly ADAPTER_SHA256="cbc593bc6ce99c0249a65e5c7cef754c9a1d7ea602f81fdae2b8cb158a25858c"
readonly NORMALIZATION_STATS="/root/feihong/FastWAM/formal_runs/FAST_WAM_github/libero_unified_shared_2cam224_1e-4/2026-07-01_00-44-20/dataset_stats.json"
readonly NORMALIZATION_STATS_SHA256="30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638"
readonly FFMPEG_APT_VERSION="7:4.4.2-0ubuntu0.22.04.1"
readonly FFMPEG_RUNTIME_VERSION="4.4.2-0ubuntu0.22.04.1"
readonly EXPECTED_TRAIN_SAMPLES="48768"
readonly EXPECTED_VALIDATION_SAMPLES="5408"
readonly EXPECTED_UPDATES="762"
readonly EXPECTED_PARAMETER_COUNT="658977"

[[ "${FASTWAM_DRY_RUN}" == "0" || "${FASTWAM_DRY_RUN}" == "1" ]] ||
  fail "FASTWAM_DRY_RUN must be 0 or 1"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] ||
  fail "RUN_ID may contain only letters, digits, dot, underscore, and hyphen"
[[ "${VISIBLE_GPU}" != *","* && -n "${VISIBLE_GPU}" && "${VISIBLE_GPU}" != *[[:space:]]* ]] ||
  fail "FASTWAM_CUDA_VISIBLE_DEVICES must name exactly one device without whitespace"
[[ -d "${FASTWAM_REPO_DIR}" ]] ||
  fail "FASTWAM_REPO_DIR does not exist: ${FASTWAM_REPO_DIR}"
FASTWAM_REPO_DIR="$(cd -- "${FASTWAM_REPO_DIR}" && pwd -P)"
[[ "${FASTWAM_REPO_DIR}" == "${DEFAULT_REPO_DIR}" ]] ||
  fail "FASTWAM_REPO_DIR must resolve to the checkout containing this launcher"
[[ "${FASTWAM_STORAGE_ROOT}" == /* ]] ||
  fail "FASTWAM_STORAGE_ROOT must be absolute"

GIT_COMMIT="$(git -C "${FASTWAM_REPO_DIR}" rev-parse HEAD)"
GIT_SHORT="${GIT_COMMIT:0:7}"
PERSISTENT_ROOT="$(realpath -m -- "${FASTWAM_STORAGE_ROOT}/FastWAM")"
DEFAULT_SMOKE_ROOT="${PERSISTENT_ROOT}/formal_runs/smokes/stage2/gate/libero_1xh100_${GIT_SHORT}_${RUN_ID}"
RAW_SMOKE_ROOT="${FASTWAM_LIBERO_GATE_SMOKE_ROOT:-${DEFAULT_SMOKE_ROOT}}"
[[ "${RAW_SMOKE_ROOT}" == /* ]] ||
  fail "FASTWAM_LIBERO_GATE_SMOKE_ROOT must be absolute"
SMOKE_ROOT="$(realpath -m -- "${RAW_SMOKE_ROOT}")"
if [[ "${FASTWAM_DRY_RUN}" == "0" ]]; then
  [[ "${SMOKE_ROOT}" == "${PERSISTENT_ROOT}/"* ]] ||
    fail "smoke output must stay under ${PERSISTENT_ROOT}"
fi
GATE_RUN="${SMOKE_ROOT}/gate_run"
TRAIN_LOG="${SMOKE_ROOT}/train.log"
RECEIPT="${SMOKE_ROOT}/verification_receipt.json"
VERIFIER="${FASTWAM_REPO_DIR}/scripts/verify_libero_stage2_gate_smoke.py"

export FASTWAM_LIBERO_STAGE2_GATE_RUN="${GATE_RUN}"
export FASTWAM_LIBERO_STAGE3_DATA_MANIFEST="${DATA_MANIFEST}"
export FASTWAM_LIBERO_STAGE2_SELECTION_DIR="${SELECTION_DIR}"
export FASTWAM_LIBERO_STAGE2_LABEL_JOB="${LABEL_JOB}"
export FASTWAM_LIBERO_STAGE2_MERGED_MANIFEST="${MERGED_MANIFEST}"
export FASTWAM_LIBERO_STATS="${NORMALIZATION_STATS}"
export FASTWAM_FFMPEG_APT_VERSION="${FFMPEG_APT_VERSION}"
export FASTWAM_FFMPEG_RUNTIME_VERSION="${FFMPEG_RUNTIME_VERSION}"
export CUDA_VISIBLE_DEVICES="${VISIBLE_GPU}"
export PATH="${FASTWAM_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${FASTWAM_REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export HF_HUB_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

# JiHe may inject allocation-wide distributed variables. Gate training is
# deliberately single-process, so discard all of them rather than forwarding
# NPROC_PER_NODE=auto or a stale rank.
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK NODE_RANK
unset NPROC_PER_NODE NNODES MASTER_ADDR MASTER_PORT

COMMAND=(
  "${PYTHON_BIN}"
  "${FASTWAM_REPO_DIR}/scripts/train_video_gate.py"
  "task=libero_stage2_gate_2cam224"
  "output_dir=${GATE_RUN}"
  "data_manifest.path=${DATA_MANIFEST}"
  "data_manifest.expected_sha256=${DATA_MANIFEST_SHA256}"
  "label_selection.directory=${SELECTION_DIR}"
  "label_selection.expected_sha256=${SELECTION_SHA256}"
  "label_coverage.tier=formal"
  "label_coverage.expected_sha256=${COVERAGE_SHA256}"
  "episode_split.path=${SELECTION_DIR}/episode_split.json"
  "episode_split.expected_assignment_sha256=${EPISODE_ASSIGNMENT_SHA256}"
  "label_contract.path=${LABEL_JOB}/label_contract.json"
  "label_contract.expected_sha256=${LABEL_CONTRACT_SHA256}"
  "label_manifest.path=${MERGED_MANIFEST}"
  "label_manifest.expected_sha256=${MERGED_MANIFEST_SHA256}"
  "source_identities.base_checkpoint_sha256=${BASE_SHA256}"
  "source_identities.adapter_checkpoint_sha256=${ADAPTER_SHA256}"
  "assets.normalization_stats.path=${NORMALIZATION_STATS}"
  "assets.normalization_stats.expected_sha256=${NORMALIZATION_STATS_SHA256}"
  "data.train.pretrained_norm_stats=${NORMALIZATION_STATS}"
  "data.train.seed=42"
  "data.train.strict_data_mode=true"
  "data.train.video_backend=torchcodec"
  "data.train.save_stats_copy=false"
  "training.seed=42"
  "training.batch_size=64"
  "training.num_workers=0"
  "training.pin_memory=true"
  "training.shuffle=true"
  "training.learning_rate=1.0e-4"
  "training.weight_decay=1.0e-4"
  "training.max_grad_norm=1.0"
  "training.num_epochs=1"
  "training.early_stop_patience=3"
  "training.min_delta=0.0"
  "training.threshold=0.5"
  "training.num_calibration_bins=10"
  "checkpoint.strict_resume=true"
  "checkpoint.resume=null"
  "runtime.repo_dir=${FASTWAM_REPO_DIR}"
  "runtime.require_clean_git=true"
  "runtime.device=cuda:0"
  "runtime.require_cuda=true"
  "runtime.deterministic_algorithms=true"
)
VERIFY_COMMAND=(
  "${PYTHON_BIN}"
  "${VERIFIER}"
  --output-dir "${GATE_RUN}"
  --expected-git-commit "${GIT_COMMIT}"
  --receipt "${RECEIPT}"
)

cat <<EOF
[stage2-gate-smoke]
  benchmark=LIBERO
  topology=1x1
  process_mode=single_python_no_torchrun
  gate_parameters=${EXPECTED_PARAMETER_COUNT}
  train_samples=${EXPECTED_TRAIN_SAMPLES}
  validation_samples=${EXPECTED_VALIDATION_SAMPLES}
  batch_size=64
  expected_updates=${EXPECTED_UPDATES}
  epochs=1
  cublas_workspace_config=${CUBLAS_WORKSPACE_CONFIG}
  label_manifest_sha256=${MERGED_MANIFEST_SHA256}
  smoke_root=${SMOKE_ROOT}
  git_commit=${GIT_COMMIT}
EOF
print_command "${COMMAND[@]}"
print_command "${VERIFY_COMMAND[@]}"

if [[ "${FASTWAM_DRY_RUN}" == "1" ]]; then
  echo "[dry-run] no files, GPUs, packages, or output directories were touched"
  exit 0
fi

preflight() {
  [[ -x "${PYTHON_BIN}" ]] || fail "missing Python environment: ${PYTHON_BIN}"
  [[ -f "${VERIFIER}" ]] || fail "missing Gate smoke verifier: ${VERIFIER}"
  [[ -f "${SCRIPT_DIR}/ensure_torchcodec_runtime.sh" ]] ||
    fail "missing TorchCodec runtime helper"
  local asset
  for asset in     "${DATA_MANIFEST}"     "${SELECTION_DIR}/episode_split.json"     "${LABEL_JOB}/label_contract.json"     "${MERGED_MANIFEST}"     "${MERGED_MANIFEST%/*}/labels.jsonl"     "${NORMALIZATION_STATS}"; do
    [[ -f "${asset}" ]] || fail "locked Stage 2 artifact is missing: ${asset}"
  done
  git -C "${FASTWAM_REPO_DIR}" diff --quiet ||
    fail "tracked worktree is dirty"
  git -C "${FASTWAM_REPO_DIR}" diff --cached --quiet ||
    fail "Git index is dirty"
  local untracked_source
  untracked_source="$(
    git -C "${FASTWAM_REPO_DIR}" ls-files --others --exclude-standard --       src configs scripts tests
  )"
  [[ -z "${untracked_source}" ]] ||
    fail "untracked source/config/script/test files: ${untracked_source}"
  [[ ! -e "${SMOKE_ROOT}" && ! -L "${SMOKE_ROOT}" ]] ||
    fail "fresh smoke output already exists: ${SMOKE_ROOT}"
}

preflight
# shellcheck source=ensure_torchcodec_runtime.sh
source "${SCRIPT_DIR}/ensure_torchcodec_runtime.sh"
"${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(
        f"expected exactly one visible CUDA device, found {torch.cuda.device_count()}"
    )
name = torch.cuda.get_device_name(0)
if "H100" not in name:
    raise SystemExit(f"expected one H100, found {name}")
print(f"[gpu] count=1 name={name}")
PY

mkdir -p -- "${SMOKE_ROOT}"
HEARTBEAT_PID=""
ACTIVE_PID=""
cleanup_heartbeat() {
  if [[ -n "${HEARTBEAT_PID}" ]]; then
    kill "${HEARTBEAT_PID}" 2>/dev/null || true
    wait "${HEARTBEAT_PID}" 2>/dev/null || true
    HEARTBEAT_PID=""
  fi
}
terminate_on_signal() {
  local signal_name="$1"
  local exit_code="$2"
  trap - INT TERM HUP
  echo "[signal] received ${signal_name}; terminating active Gate process" >&2
  if [[ -n "${ACTIVE_PID}" ]] && kill -0 "${ACTIVE_PID}" 2>/dev/null; then
    kill -s "${signal_name}" "${ACTIVE_PID}" 2>/dev/null || true
    wait "${ACTIVE_PID}" 2>/dev/null || true
    ACTIVE_PID=""
  fi
  cleanup_heartbeat
  exit "${exit_code}"
}
trap cleanup_heartbeat EXIT
trap 'terminate_on_signal INT 130' INT
trap 'terminate_on_signal TERM 143' TERM
trap 'terminate_on_signal HUP 129' HUP
heartbeat() {
  while sleep 60; do
    echo "[heartbeat] Gate smoke is running; elapsed=${SECONDS}s (epoch logs publish only after a full pass)"
    nvidia-smi --query-gpu=utilization.gpu,memory.used       --format=csv,noheader,nounits 2>/dev/null |
      sed 's/^/[heartbeat] gpu_util_percent,memory_used_MiB=/'
  done
}
SECONDS=0
heartbeat &
HEARTBEAT_PID="$!"

set +e
"${COMMAND[@]}" > >(tee "${TRAIN_LOG}") 2>&1 &
ACTIVE_PID="$!"
wait "${ACTIVE_PID}"
TRAIN_STATUS="$?"
ACTIVE_PID=""
set -e
cleanup_heartbeat
[[ "${TRAIN_STATUS}" == "0" ]] ||
  fail "LIBERO Gate smoke training failed with exit code ${TRAIN_STATUS}"

set +e
"${VERIFY_COMMAND[@]}" &
ACTIVE_PID="$!"
wait "${ACTIVE_PID}"
VERIFY_STATUS="$?"
ACTIVE_PID=""
set -e
[[ "${VERIFY_STATUS}" == "0" ]] ||
  fail "LIBERO Gate smoke verification failed with exit code ${VERIFY_STATUS}"
trap - EXIT INT TERM HUP

echo "[ok] LIBERO Stage 2 Gate full one-epoch smoke passed"
echo "[ok] gate_run=${GATE_RUN}"
echo "[ok] verification_receipt=${RECEIPT}"
echo "[next] inspect smoke metrics, then start a separate fresh formal Gate run"
