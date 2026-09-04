#!/usr/bin/env bash
set -euo pipefail

# One-click, one-H100 replay of the immutable LIBERO Gate validation split.
# It exports probabilities once and derives five compute-target thresholds.

fail() {
  echo "[error] $*" >&2
  exit 1
}

print_command() {
  printf "[launch]"
  printf " %q" "$@"
  printf "\n"
}

[[ "$#" == "0" ]] || fail "this one-click launcher takes no arguments"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
FASTWAM_REPO_DIR="${FASTWAM_REPO_DIR:-${DEFAULT_REPO_DIR}}"
FASTWAM_ENV="${FASTWAM_ENV:-/root/.venvs/fastwam}"
FASTWAM_STORAGE_ROOT="${FASTWAM_STORAGE_ROOT:-/root/feihong}"
FASTWAM_DRY_RUN="${FASTWAM_DRY_RUN:-0}"
PYTHON_BIN="${FASTWAM_ENV}/bin/python"
RUN_ID="${RUN_ID:-$(date -u +%Y-%m-%d_%H-%M-%S)}"
VISIBLE_GPU="${FASTWAM_CUDA_VISIBLE_DEVICES:-0}"

readonly GATE_RUN="/root/feihong/FastWAM/formal_runs/stage2/gate/libero_stage2_gate_2cam224_20ep/22a8d65_2026-09-02_03-29-13/gate_run"
readonly GATE_CHECKPOINT="${GATE_RUN}/gate_best.pt"
readonly GATE_CHECKPOINT_SHA256="67db6f46abe67f5c6a4417b60864f0ad0535edf8f911d9e4d11eaed137b9b722"
readonly GATE_RUN_IDENTITY="${GATE_RUN}/run_identity.json"
readonly GATE_RUN_IDENTITY_SHA256="c7562b913a54c435b4866aaec02166fbd72d50c0e743e4c0ea3e2d1ebac15247"
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
readonly EXPECTED_VALIDATION_SAMPLES="5408"
readonly EXPECTED_VALIDATION_BATCHES="85"
readonly TARGET_WITH_RATES="[0.10,0.25,0.50,0.75,0.90]"
readonly CONFIGURED_VIDEO_STEPS="10"

[[ "${FASTWAM_DRY_RUN}" == "0" || "${FASTWAM_DRY_RUN}" == "1" ]] ||
  fail "FASTWAM_DRY_RUN must be 0 or 1"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] ||
  fail "RUN_ID may contain only letters, digits, dot, underscore, and hyphen"
[[ "${VISIBLE_GPU}" != *","* && -n "${VISIBLE_GPU}" && "${VISIBLE_GPU}" != *[[:space:]]* ]] ||
  fail "FASTWAM_CUDA_VISIBLE_DEVICES must name exactly one device"
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
DEFAULT_RUN_ROOT="${PERSISTENT_ROOT}/formal_runs/stage2/calibration/libero_stage2_gate_2cam224/${GIT_SHORT}_${RUN_ID}"
RAW_RUN_ROOT="${FASTWAM_LIBERO_GATE_CALIBRATION_ROOT:-${DEFAULT_RUN_ROOT}}"
[[ "${RAW_RUN_ROOT}" == /* ]] ||
  fail "FASTWAM_LIBERO_GATE_CALIBRATION_ROOT must be absolute"
RUN_ROOT="$(realpath -m -- "${RAW_RUN_ROOT}")"
if [[ "${FASTWAM_DRY_RUN}" == "0" ]]; then
  [[ "${RUN_ROOT}" == "${PERSISTENT_ROOT}/"* ]] ||
    fail "calibration output must stay under ${PERSISTENT_ROOT}"
fi
CALIBRATION_DIR="${RUN_ROOT}/calibration"
LOG_FILE="${RUN_ROOT}/calibration.log"

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
export PYTHONPATH="${FASTWAM_REPO_DIR}:${FASTWAM_REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export HF_HUB_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

# JiHe can inject allocation-wide distributed variables.  Calibration is one
# deterministic process on one H100.
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK NODE_RANK
unset MACHINE_RANK NNODES NPROC_PER_NODE MASTER_ADDR MASTER_PORT

COMMAND=(
  "${PYTHON_BIN}"
  "${FASTWAM_REPO_DIR}/scripts/calibrate_video_gate.py"
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
  "training.num_epochs=20"
  "training.early_stop_patience=3"
  "training.min_delta=1.0e-4"
  "training.threshold=0.5"
  "training.num_calibration_bins=10"
  "checkpoint.strict_resume=true"
  "checkpoint.resume=null"
  "runtime.repo_dir=${FASTWAM_REPO_DIR}"
  "runtime.require_clean_git=true"
  "runtime.device=cuda:0"
  "runtime.require_cuda=true"
  "runtime.deterministic_algorithms=true"
  "calibration.gate_checkpoint=${GATE_CHECKPOINT}"
  "calibration.gate_checkpoint_sha256=${GATE_CHECKPOINT_SHA256}"
  "calibration.gate_run_identity=${GATE_RUN_IDENTITY}"
  "calibration.gate_run_identity_sha256=${GATE_RUN_IDENTITY_SHA256}"
  "calibration.output_dir=${CALIBRATION_DIR}"
  "calibration.source_split=validation"
  "calibration.target_with_rates=${TARGET_WITH_RATES}"
  "calibration.configured_video_steps=${CONFIGURED_VIDEO_STEPS}"
  "calibration.expected_validation_samples=${EXPECTED_VALIDATION_SAMPLES}"
  "calibration.validation_batch_size=64"
  "calibration.validation_num_workers=0"
  "calibration.validation_pin_memory=true"
  "calibration.metric_abs_tolerance=1.0e-6"
  "calibration.require_exact_training_numerical_runtime=false"
  "calibration.progress_every_batches=5"
)

cat <<EOF
[libero-stage2-gate-calibration]
  topology=1xH100 single_process
  source_split=validation
  samples=${EXPECTED_VALIDATION_SAMPLES}
  batches=${EXPECTED_VALIDATION_BATCHES}
  target_with_rates=${TARGET_WITH_RATES}
  video_steps_when_selected=${CONFIGURED_VIDEO_STEPS}
  numerical_runtime_policy=metric_reproduction_guarded
  gate_sha256=${GATE_CHECKPOINT_SHA256}
  run_root=${RUN_ROOT}
  git_commit=${GIT_COMMIT}
EOF
print_command "${COMMAND[@]}"

if [[ "${FASTWAM_DRY_RUN}" == "1" ]]; then
  echo "[dry-run] no files, GPUs, packages, or output directories were touched"
  exit 0
fi

verify_repository_immutability() {
  local phase="$1"
  [[ "$(git -C "${FASTWAM_REPO_DIR}" rev-parse HEAD)" == "${GIT_COMMIT}" ]] ||
    fail "Git HEAD changed after ${phase}"
  git -C "${FASTWAM_REPO_DIR}" diff --quiet ||
    fail "tracked worktree is dirty at ${phase}"
  git -C "${FASTWAM_REPO_DIR}" diff --cached --quiet ||
    fail "Git index is dirty at ${phase}"
  local untracked_source
  untracked_source="$(
    git -C "${FASTWAM_REPO_DIR}" ls-files --others --exclude-standard -- src configs scripts tests
  )"
  [[ -z "${untracked_source}" ]] ||
    fail "untracked source/config/script/test files at ${phase}: ${untracked_source}"
}

preflight() {
  [[ -x "${PYTHON_BIN}" ]] || fail "missing Python: ${PYTHON_BIN}"
  [[ -f "${FASTWAM_REPO_DIR}/scripts/calibrate_video_gate.py" ]] ||
    fail "missing Gate calibration entrypoint"
  [[ -f "${SCRIPT_DIR}/ensure_torchcodec_runtime.sh" ]] ||
    fail "missing TorchCodec runtime helper"
  local asset
  for asset in \
    "${GATE_CHECKPOINT}" \
    "${GATE_RUN_IDENTITY}" \
    "${DATA_MANIFEST}" \
    "${SELECTION_DIR}/episode_split.json" \
    "${LABEL_JOB}/label_contract.json" \
    "${MERGED_MANIFEST}" \
    "${MERGED_MANIFEST%/*}/labels.jsonl" \
    "${NORMALIZATION_STATS}"; do
    [[ -f "${asset}" ]] || fail "locked calibration artifact is missing: ${asset}"
  done
  verify_repository_immutability "preflight"
  [[ ! -e "${RUN_ROOT}" && ! -L "${RUN_ROOT}" ]] ||
    fail "fresh calibration output already exists: ${RUN_ROOT}"
}

preflight
# shellcheck source=ensure_torchcodec_runtime.sh
source "${SCRIPT_DIR}/ensure_torchcodec_runtime.sh"
"${PYTHON_BIN}" - <<'PY'
import torch

names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
if len(names) != 1 or "H100" not in names[0].upper():
    raise SystemExit(f"expected exactly one visible H100, found {names}")
print(f"[gpu] count=1 name={names[0]}")
PY

mkdir -p -- "${RUN_ROOT%/*}"
mkdir -- "${RUN_ROOT}" || fail "failed to claim calibration output: ${RUN_ROOT}"

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
  if [[ -n "${ACTIVE_PID}" ]] && kill -0 "${ACTIVE_PID}" 2>/dev/null; then
    kill -s "${signal_name}" "${ACTIVE_PID}" 2>/dev/null || true
    wait "${ACTIVE_PID}" 2>/dev/null || true
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
    echo "[heartbeat] calibration running; elapsed=${SECONDS}s"
    nvidia-smi --query-gpu=utilization.gpu,memory.used \
      --format=csv,noheader,nounits 2>/dev/null |
      sed 's/^/[heartbeat] gpu_util_percent,memory_used_MiB=/'
  done
}

SECONDS=0
heartbeat &
HEARTBEAT_PID="$!"
set +e
"${COMMAND[@]}" > >(tee "${LOG_FILE}") 2>&1 &
ACTIVE_PID="$!"
wait "${ACTIVE_PID}"
STATUS="$?"
ACTIVE_PID=""
set -e
cleanup_heartbeat
[[ "${STATUS}" == "0" ]] ||
  fail "LIBERO Gate calibration failed with exit code ${STATUS}"
verify_repository_immutability "calibration"
[[ -f "${CALIBRATION_DIR}/COMPLETE" ]] ||
  fail "calibration returned without COMPLETE"
[[ -f "${CALIBRATION_DIR}/thresholds.json" ]] ||
  fail "calibration returned without thresholds.json"
trap - EXIT INT TERM HUP

echo "[ok] LIBERO Gate validation calibration complete"
echo "[ok] thresholds=${CALIBRATION_DIR}/thresholds.json"
echo "[ok] manifest=${CALIBRATION_DIR}/calibration_manifest.json"
echo "[next] run static endpoints and one independent closed-loop run per calibrated threshold"
