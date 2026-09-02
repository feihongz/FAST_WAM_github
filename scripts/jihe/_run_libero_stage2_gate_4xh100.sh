#!/usr/bin/env bash
set -euo pipefail

# Strict one-node/four-H100 engine for the small LIBERO BinaryVideoGate.

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
TORCHRUN_BIN="${FASTWAM_ENV}/bin/torchrun"
RUN_ID="${RUN_ID:-$(date -u +%Y-%m-%d_%H-%M-%S)}"
VISIBLE_GPUS="${FASTWAM_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
PROFILE="${FASTWAM_GATE_PROFILE:-formal}"
RESUME_MODE="${FASTWAM_GATE_RESUME:-0}"
REQUESTED_NPROC="${NPROC_PER_NODE:-${SENSECORE_ACCELERATE_DEVICE_COUNT:-4}}"
REQUESTED_NNODES="${NNODES:-1}"

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
readonly EXPECTED_PARAMETER_COUNT="658977"
readonly EXPECTED_WORLD_SIZE="4"
readonly PER_RANK_BATCH="16"
readonly GLOBAL_BATCH="64"
readonly UPDATES_PER_EPOCH="762"
readonly NUM_EPOCHS="20"
readonly MAXIMUM_UPDATES="15240"
readonly MIN_DELTA="1.0e-4"

[[ "${FASTWAM_DRY_RUN}" == "0" || "${FASTWAM_DRY_RUN}" == "1" ]] ||
  fail "FASTWAM_DRY_RUN must be 0 or 1"
[[ "${RESUME_MODE}" == "0" || "${RESUME_MODE}" == "1" ]] ||
  fail "FASTWAM_GATE_RESUME must be 0 or 1"
[[ "${PROFILE}" == "formal" ]] ||
  fail "FASTWAM_GATE_PROFILE must be formal"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] ||
  fail "RUN_ID may contain only letters, digits, dot, underscore, and hyphen"
if [[ "${REQUESTED_NPROC,,}" == "auto" ]]; then
  # JiHe uses the literal `auto`; this launcher owns an immutable 4-rank job.
  REQUESTED_NPROC="4"
fi
[[ "${REQUESTED_NPROC}" == "4" ]] ||
  fail "NPROC_PER_NODE must resolve to exactly 4, got ${REQUESTED_NPROC}"
[[ "${REQUESTED_NNODES}" == "1" ]] ||
  fail "NNODES must be exactly 1, got ${REQUESTED_NNODES}"
for rank_name in NODE_RANK MACHINE_RANK GROUP_RANK; do
  rank_value="${!rank_name:-0}"
  [[ "${rank_value}" == "0" ]] ||
    fail "NODE_RANK must be 0 for one-node Gate DDP (got ${rank_name}=${rank_value})"
done
[[ "${VISIBLE_GPUS}" =~ ^[^,[:space:]]+,[^,[:space:]]+,[^,[:space:]]+,[^,[:space:]]+$ ]] ||
  fail "FASTWAM_CUDA_VISIBLE_DEVICES must name exactly four devices without whitespace"
IFS=',' read -r -a GPU_TOKENS <<< "${VISIBLE_GPUS}"
[[ "${#GPU_TOKENS[@]}" == "4" ]] ||
  fail "FASTWAM_CUDA_VISIBLE_DEVICES must name exactly four devices"
for left in 0 1 2 3; do
  for right in 0 1 2 3; do
    if (( left < right )) && [[ "${GPU_TOKENS[${left}]}" == "${GPU_TOKENS[${right}]}" ]]; then
      fail "FASTWAM_CUDA_VISIBLE_DEVICES must name four unique devices"
    fi
  done
done
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
DEFAULT_RUN_ROOT="${PERSISTENT_ROOT}/formal_runs/stage2/gate/libero_stage2_gate_2cam224_20ep_4xh100/${GIT_SHORT}_${RUN_ID}"
if [[ "${RESUME_MODE}" == "1" && -z "${FASTWAM_LIBERO_GATE_FORMAL_4XH100_ROOT:-}" ]]; then
  fail "FASTWAM_GATE_RESUME=1 requires FASTWAM_LIBERO_GATE_FORMAL_4XH100_ROOT"
fi
RAW_RUN_ROOT="${FASTWAM_LIBERO_GATE_FORMAL_4XH100_ROOT:-${DEFAULT_RUN_ROOT}}"
[[ "${RAW_RUN_ROOT}" == /* ]] ||
  fail "FASTWAM_LIBERO_GATE_FORMAL_4XH100_ROOT must be absolute"
RUN_ROOT="$(realpath -m -- "${RAW_RUN_ROOT}")"
if [[ "${FASTWAM_DRY_RUN}" == "0" ]]; then
  [[ "${RUN_ROOT}" == "${PERSISTENT_ROOT}/"* ]] ||
    fail "formal output must stay under ${PERSISTENT_ROOT}"
fi
GATE_RUN="${RUN_ROOT}/gate_run"
TRAIN_LOG="${RUN_ROOT}/train.log"
RECEIPT="${RUN_ROOT}/verification_receipt.json"
VERIFIER="${FASTWAM_REPO_DIR}/scripts/verify_libero_stage2_gate_formal_4xh100.py"
if [[ "${RESUME_MODE}" == "1" ]]; then
  RESUME_STATE="${GATE_RUN}/training_state.pt"
else
  RESUME_STATE="null"
fi

export FASTWAM_LIBERO_STAGE2_GATE_RUN="${GATE_RUN}"
export FASTWAM_LIBERO_STAGE3_DATA_MANIFEST="${DATA_MANIFEST}"
export FASTWAM_LIBERO_STAGE2_SELECTION_DIR="${SELECTION_DIR}"
export FASTWAM_LIBERO_STAGE2_LABEL_JOB="${LABEL_JOB}"
export FASTWAM_LIBERO_STAGE2_MERGED_MANIFEST="${MERGED_MANIFEST}"
export FASTWAM_LIBERO_STATS="${NORMALIZATION_STATS}"
export FASTWAM_FFMPEG_APT_VERSION="${FFMPEG_APT_VERSION}"
export FASTWAM_FFMPEG_RUNTIME_VERSION="${FFMPEG_RUNTIME_VERSION}"
export CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}"
export PATH="${FASTWAM_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${FASTWAM_REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export HF_HUB_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export NCCL_ASYNC_ERROR_HANDLING="1"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="1"

# JiHe may inject allocation-wide metadata. The nested one-node torchrun owns it.
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE ROLE_RANK NODE_RANK
unset MACHINE_RANK GROUP_RANK NNODES NPROC_PER_NODE MASTER_ADDR MASTER_PORT

COMMAND=(
  "${TORCHRUN_BIN}"
  --standalone
  --nnodes=1
  --nproc_per_node=4
  --max_restarts=0
  "${FASTWAM_REPO_DIR}/scripts/train_video_gate_distributed.py"
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
  "training.batch_size=${GLOBAL_BATCH}"
  "training.num_workers=0"
  "training.pin_memory=true"
  "training.shuffle=true"
  "training.learning_rate=1.0e-4"
  "training.weight_decay=1.0e-4"
  "training.max_grad_norm=1.0"
  "training.num_epochs=${NUM_EPOCHS}"
  "training.early_stop_patience=3"
  "training.min_delta=${MIN_DELTA}"
  "training.threshold=0.5"
  "training.num_calibration_bins=10"
  "checkpoint.strict_resume=true"
  "checkpoint.resume=${RESUME_STATE}"
  "runtime.repo_dir=${FASTWAM_REPO_DIR}"
  "runtime.require_clean_git=true"
  "runtime.device=cuda:0"
  "runtime.require_cuda=true"
  "runtime.deterministic_algorithms=true"
)
VERIFY_COMMAND=(
  env
  "CUDA_VISIBLE_DEVICES=${GPU_TOKENS[0]}"
  "${PYTHON_BIN}"
  "${VERIFIER}"
  --output-dir "${GATE_RUN}"
  --expected-git-commit "${GIT_COMMIT}"
  --receipt "${RECEIPT}"
)

cat <<EOF
[stage2-gate-${PROFILE}]
  profile=${PROFILE}
  benchmark=LIBERO
  topology=1x4
  world_size=${EXPECTED_WORLD_SIZE}
  process_mode=torchrun_native_ddp
  gate_parameters=${EXPECTED_PARAMETER_COUNT}
  train_samples=${EXPECTED_TRAIN_SAMPLES}
  validation_samples=${EXPECTED_VALIDATION_SAMPLES}
  per_rank_batch=${PER_RANK_BATCH}
  global_batch=${GLOBAL_BATCH}
  updates_per_epoch=${UPDATES_PER_EPOCH}
  maximum_updates=${MAXIMUM_UPDATES}
  epochs=${NUM_EPOCHS}
  early_stop_patience=3
  min_delta=${MIN_DELTA}
  cublas_workspace_config=${CUBLAS_WORKSPACE_CONFIG}
  label_manifest_sha256=${MERGED_MANIFEST_SHA256}
  run_root=${RUN_ROOT}
  resume_mode=${RESUME_MODE}
  resume_state=${RESUME_STATE}
  git_commit=${GIT_COMMIT}
EOF
print_command "${COMMAND[@]}"
print_command "${VERIFY_COMMAND[@]}"

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
  [[ -x "${PYTHON_BIN}" ]] || fail "missing Python environment: ${PYTHON_BIN}"
  [[ -x "${TORCHRUN_BIN}" ]] || fail "missing torchrun: ${TORCHRUN_BIN}"
  [[ -f "${FASTWAM_REPO_DIR}/scripts/train_video_gate_distributed.py" ]] ||
    fail "missing distributed Gate entrypoint"
  [[ -f "${VERIFIER}" ]] || fail "missing four-rank Gate verifier: ${VERIFIER}"
  [[ -f "${SCRIPT_DIR}/ensure_torchcodec_runtime.sh" ]] ||
    fail "missing TorchCodec runtime helper"
  local asset
  for asset in \
    "${DATA_MANIFEST}" \
    "${SELECTION_DIR}/episode_split.json" \
    "${LABEL_JOB}/label_contract.json" \
    "${MERGED_MANIFEST}" \
    "${MERGED_MANIFEST%/*}/labels.jsonl" \
    "${NORMALIZATION_STATS}"; do
    [[ -f "${asset}" ]] || fail "locked Stage 2 artifact is missing: ${asset}"
  done
  verify_repository_immutability "preflight"
  if [[ "${RESUME_MODE}" == "1" ]]; then
    [[ -d "${RAW_RUN_ROOT}" && ! -L "${RAW_RUN_ROOT}" ]] ||
      fail "resume formal output must be an existing non-symlink directory: ${RUN_ROOT}"
    [[ -f "${RESUME_STATE}" && ! -L "${RESUME_STATE}" ]] ||
      fail "resume training state must be an existing non-symlink file: ${RESUME_STATE}"
  else
    [[ ! -e "${RUN_ROOT}" && ! -L "${RUN_ROOT}" ]] ||
      fail "fresh formal output already exists: ${RUN_ROOT}"
  fi
}

preflight
# shellcheck source=ensure_torchcodec_runtime.sh
source "${SCRIPT_DIR}/ensure_torchcodec_runtime.sh"
"${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
    raise SystemExit(
        "expected exactly four visible CUDA devices, "
        f"found {torch.cuda.device_count()}"
    )
names = [torch.cuda.get_device_name(index) for index in range(4)]
if any("H100" not in name for name in names):
    raise SystemExit(f"expected four H100 GPUs, found {names}")
print(f"[gpu] count=4 names={names}")
PY

if [[ "${RESUME_MODE}" == "0" ]]; then
  mkdir -p -- "${RUN_ROOT%/*}"
  mkdir -- "${RUN_ROOT}" ||
    fail "failed to atomically claim fresh formal output: ${RUN_ROOT}"
fi
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
  echo "[signal] received ${signal_name}; terminating active Gate process group" >&2
  if [[ -n "${ACTIVE_PID}" ]] && kill -0 "${ACTIVE_PID}" 2>/dev/null; then
    kill -s "${signal_name}" -- "-${ACTIVE_PID}" 2>/dev/null || true
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
    echo "[heartbeat] Gate formal DDP is running; elapsed=${SECONDS}s (epoch logs publish after a full pass)"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
      --format=csv,noheader,nounits 2>/dev/null |
      sed 's/^/[heartbeat] gpu_index,util_percent,memory_used_MiB=/'
  done
}
SECONDS=0
heartbeat &
HEARTBEAT_PID="$!"

set +e
setsid "${COMMAND[@]}" > >(tee -a "${TRAIN_LOG}") 2>&1 &
ACTIVE_PID="$!"
wait "${ACTIVE_PID}"
TRAIN_STATUS="$?"
ACTIVE_PID=""
set -e
cleanup_heartbeat
[[ "${TRAIN_STATUS}" == "0" ]] ||
  fail "LIBERO Gate formal DDP training failed with exit code ${TRAIN_STATUS}"
verify_repository_immutability "training"

set +e
setsid "${VERIFY_COMMAND[@]}" &
ACTIVE_PID="$!"
wait "${ACTIVE_PID}"
VERIFY_STATUS="$?"
ACTIVE_PID=""
set -e
[[ "${VERIFY_STATUS}" == "0" ]] ||
  fail "LIBERO Gate formal DDP verification failed with exit code ${VERIFY_STATUS}"
verify_repository_immutability "verification"
trap - EXIT INT TERM HUP

echo "[ok] LIBERO Stage 2 Gate formal 4xH100 run passed"
echo "[ok] gate_run=${GATE_RUN}"
echo "[ok] verification_receipt=${RECEIPT}"
echo "[next] grade offline Gate quality, sweep the routing threshold, then run held-out policy eval"
