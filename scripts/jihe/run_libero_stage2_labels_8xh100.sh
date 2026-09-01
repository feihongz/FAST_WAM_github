#!/usr/bin/env bash
set -euo pipefail
set +m

# Formal, resumable LIBERO Stage 2 E0/E10 label generation on one 8xH100
# JiHe node. The formal tier labels a stratified nested subset; cohort chunks
# remain reusable when the same selection campaign expands to the cap tier.

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
  fail "this one-click launcher takes no arguments; its formal contract is immutable"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
FASTWAM_REPO_DIR="${FASTWAM_REPO_DIR:-${DEFAULT_REPO_DIR}}"
FASTWAM_ENV="${FASTWAM_ENV:-/root/.venvs/fastwam}"
FASTWAM_STORAGE_ROOT="${FASTWAM_STORAGE_ROOT:-/root/feihong}"
FASTWAM_DRY_RUN="${FASTWAM_DRY_RUN:-0}"
PYTHON_BIN="${FASTWAM_ENV}/bin/python"
TORCHRUN_BIN="${FASTWAM_ENV}/bin/torchrun"

# Keep the topology immutable even when JiHe injects `auto` or an unrelated
# allocation-wide value. Physical/logical GPU remapping is available only
# through the purpose-specific variable below.
REQUESTED_NPROC="${NPROC_PER_NODE:-${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}}"
if [[ "${REQUESTED_NPROC,,}" == "auto" ]]; then
  REQUESTED_NPROC="8"
fi
[[ "${REQUESTED_NPROC}" == "8" ]] ||
  fail "formal LIBERO Stage 2 labels require exactly 8 H100s, got NPROC_PER_NODE=${REQUESTED_NPROC}"
readonly NPROC_PER_NODE="8"
readonly NNODES="1"

VISIBLE_GPUS="${FASTWAM_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
[[ "${VISIBLE_GPUS}" != ,* && "${VISIBLE_GPUS}" != *, && "${VISIBLE_GPUS}" != *,,* ]] ||
  fail "FASTWAM_CUDA_VISIBLE_DEVICES must contain exactly eight non-empty comma-separated devices"
IFS=',' read -r -a VISIBLE_GPU_TOKENS <<< "${VISIBLE_GPUS}"
[[ "${#VISIBLE_GPU_TOKENS[@]}" == "8" ]] ||
  fail "FASTWAM_CUDA_VISIBLE_DEVICES must contain exactly eight comma-separated devices"
declare -A SEEN_VISIBLE_GPUS=()
for token in "${VISIBLE_GPU_TOKENS[@]}"; do
  [[ -n "${token}" && "${token}" != *[[:space:]]* ]] ||
    fail "FASTWAM_CUDA_VISIBLE_DEVICES contains an empty or whitespace-bearing device"
  [[ -z "${SEEN_VISIBLE_GPUS[${token}]:-}" ]] ||
    fail "FASTWAM_CUDA_VISIBLE_DEVICES contains a duplicate device: ${token}"
  SEEN_VISIBLE_GPUS["${token}"]=1
done
unset SEEN_VISIBLE_GPUS VISIBLE_GPU_TOKENS

# Frozen LIBERO Stage 3 final identities. A new path or digest requires a new,
# reviewed launcher and label contract.
readonly BASE_CHECKPOINT="/root/feihong/FastWAM/formal_runs/FAST_WAM_github/libero_unified_shared_2cam224_1e-4/2026-07-01_00-44-20/checkpoints/weights/latest.pt"
readonly BASE_SHA256="17a5588cc2b8d162219c9daf818614f614ee4a7921933a4a26c5d678111330e9"
readonly ADAPTER_CHECKPOINT="/root/feihong/FastWAM/formal_runs/stage3/full/libero_stage3_alignment_2cam224_1e-4/2026-08-30_10-29-08/checkpoints/exports/step_030000.pt"
readonly ADAPTER_SHA256="cbc593bc6ce99c0249a65e5c7cef754c9a1d7ea602f81fdae2b8cb158a25858c"
readonly DATA_MANIFEST="/root/feihong/FastWAM/formal_runs/contracts/stage3/libero_current_273465f_1693e/libero_stage3_data_manifest.json"
readonly DATA_MANIFEST_SHA256="08da49109a57b55c67f3fa4ac31fbfa44e44dd541a194a5d3420838537d0d320"
readonly NORMALIZATION_STATS="/root/feihong/FastWAM/formal_runs/FAST_WAM_github/libero_unified_shared_2cam224_1e-4/2026-07-01_00-44-20/dataset_stats.json"
readonly NORMALIZATION_STATS_SHA256="30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638"
readonly VAE_CHECKPOINT="/root/feihong/FastWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
readonly VAE_SHA256="20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"
readonly SELECTION_DIR="${FASTWAM_LIBERO_STAGE2_SELECTION_DIR:-/root/feihong/FastWAM/formal_runs/contracts/stage2/libero_nested64_stratified_v2_426b635d}"
readonly SELECTION_SHA256="426b635d637a0f3e5d31dd13612ff5ad786fd5cfe9ce27b0e8689854d9aa9e9b"
readonly COVERAGE_TIER="formal"
readonly COVERAGE_SHA256="d114ac25b61ab30f18185c9ea69a33d537b5196b145a8c5c3d6f6fd9d884708f"
readonly EXPECTED_SAMPLE_IDS_SHA256="e1122e20a0f48fd988baad3b70eea5258f4091918460bb78a7b50c4b30924aac"
readonly SPLIT_ASSIGNMENT_SHA256="a77efa24249dab8cfacbc228b1da341947240b36fa77d90182701c07bdcf7787"
readonly EXPECTED_SAMPLE_COUNT="54176"
readonly EXPECTED_TRAIN_SAMPLE_COUNT="48768"
readonly EXPECTED_VALIDATION_SAMPLE_COUNT="5408"
readonly NUM_SHARDS="64"
readonly CHUNK_SIZE="64"
readonly EXPECTED_CHUNK_COUNT="977"
readonly EXPECTED_ACTIVE_COHORTS="0,1,2,4"
readonly EXPECTED_COHORT_CHUNK_COUNTS="0:218,1:219,2:413,4:127"
readonly EXPECTED_NONEMPTY_COHORT_SHARDS="256"
readonly EXPECTED_RANK_SAMPLE_COUNTS="6707,6855,6771,6803,6816,6693,6770,6761"
readonly EXPECTED_RANK_CHUNK_COUNTS="120,123,121,125,127,118,122,121"
readonly -a ACTIVE_COHORT_DIR_NAMES=(
  cohort-00000
  cohort-00001
  cohort-00002
  cohort-00004
)
readonly FFMPEG_APT_VERSION="7:4.4.2-0ubuntu0.22.04.1"
readonly FFMPEG_RUNTIME_VERSION="4.4.2-0ubuntu0.22.04.1"

[[ "${FASTWAM_DRY_RUN}" == "0" || "${FASTWAM_DRY_RUN}" == "1" ]] ||
  fail "FASTWAM_DRY_RUN must be 0 or 1"
[[ "${FASTWAM_STORAGE_ROOT}" == /* ]] ||
  fail "FASTWAM_STORAGE_ROOT must be an absolute path"
[[ -d "${FASTWAM_REPO_DIR}" ]] ||
  fail "FASTWAM_REPO_DIR does not exist: ${FASTWAM_REPO_DIR}"
FASTWAM_REPO_DIR="$(cd -- "${FASTWAM_REPO_DIR}" && pwd -P)"
[[ "${FASTWAM_REPO_DIR}" == "${DEFAULT_REPO_DIR}" ]] ||
  fail "FASTWAM_REPO_DIR must resolve to the checkout containing this launcher"

GIT_COMMIT="$(git -C "${FASTWAM_REPO_DIR}" rev-parse HEAD)"
GIT_SHORT="${GIT_COMMIT:0:7}"
CAMPAIGN_ID="selection_${SELECTION_SHA256:0:8}_${GIT_SHORT}"
RUN_ID="${RUN_ID:-${CAMPAIGN_ID}}"
ATTEMPT_ID="${ATTEMPT_ID:-$(date -u +%Y-%m-%d_%H-%M-%S)_${BASHPID}}"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] ||
  fail "RUN_ID may contain only letters, digits, dot, underscore, and hyphen"
[[ "${ATTEMPT_ID}" =~ ^[A-Za-z0-9._-]+$ ]] ||
  fail "ATTEMPT_ID may contain only letters, digits, dot, underscore, and hyphen"

PERSISTENT_ROOT="$(realpath -m -- "${FASTWAM_STORAGE_ROOT}/FastWAM")"
DEFAULT_OUTPUT_BASE="${PERSISTENT_ROOT}/formal_runs/stage2/labels/libero_stage2_gate_labels_2cam224"
RAW_JOB_DIR="${FASTWAM_LIBERO_STAGE2_LABEL_JOB:-${DEFAULT_OUTPUT_BASE}/${RUN_ID}}"
[[ "${RAW_JOB_DIR}" == /* ]] ||
  fail "FASTWAM_LIBERO_STAGE2_LABEL_JOB must be an absolute path"
JOB_DIR="$(realpath -m -- "${RAW_JOB_DIR}")"
[[ "${JOB_DIR}" == "${PERSISTENT_ROOT}/"* ]] ||
  fail "label job must stay under ${PERSISTENT_ROOT}"

LOG_DIR="${JOB_DIR}/logs"
ATTEMPT_LOG="${LOG_DIR}/attempt-${ATTEMPT_ID}.log"
SUCCESS_PATH="${JOB_DIR}/generation_success-${COVERAGE_TIER}-${COVERAGE_SHA256:0:8}.json"
LOCK_PATH="${JOB_DIR}/.generation.lock"

export FASTWAM_LIBERO_STAGE2_LABEL_JOB="${JOB_DIR}"
export FASTWAM_LIBERO_STAGE3_BASE_CHECKPOINT="${BASE_CHECKPOINT}"
export FASTWAM_LIBERO_STAGE3_BASE_SHA256="${BASE_SHA256}"
export FASTWAM_LIBERO_STAGE3_ADAPTER="${ADAPTER_CHECKPOINT}"
export FASTWAM_LIBERO_STAGE3_ADAPTER_SHA256="${ADAPTER_SHA256}"
export FASTWAM_LIBERO_STAGE3_DATA_MANIFEST="${DATA_MANIFEST}"
export FASTWAM_LIBERO_STAGE3_DATA_MANIFEST_SHA256="${DATA_MANIFEST_SHA256}"
export FASTWAM_LIBERO_STAGE3_VAE="${VAE_CHECKPOINT}"
export FASTWAM_LIBERO_STATS="${NORMALIZATION_STATS}"
export FASTWAM_LIBERO_STAGE2_SELECTION_DIR="${SELECTION_DIR}"
export FASTWAM_LIBERO_STAGE2_SELECTION_SHA256="${SELECTION_SHA256}"
export FASTWAM_LIBERO_STAGE2_COVERAGE_SHA256="${COVERAGE_SHA256}"
export FASTWAM_FFMPEG_APT_VERSION="${FFMPEG_APT_VERSION}"
export FASTWAM_FFMPEG_RUNTIME_VERSION="${FFMPEG_RUNTIME_VERSION}"

export CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}"
export PATH="${FASTWAM_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${FASTWAM_REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DIFFSYNTH_MODEL_BASE_PATH="/root/feihong/FastWAM/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD="true"
export HF_HUB_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

# torchrun owns these values. Never let a stale outer distributed job leak
# rank identity into this single-node campaign.
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK NODE_RANK

PREPARE_COMMAND=(
  "${PYTHON_BIN}"
  "${FASTWAM_REPO_DIR}/scripts/prepare_gate_label_selection.py"
  --data-manifest "${DATA_MANIFEST}"
  --expected-manifest-sha256 "${DATA_MANIFEST_SHA256}"
  --output-dir "${SELECTION_DIR}"
)

COMMAND=(
  "${TORCHRUN_BIN}"
  --standalone
  --nnodes=1
  --nproc_per_node=8
  --max_restarts=0
  "${FASTWAM_REPO_DIR}/scripts/generate_gate_labels.py"
  "task=libero_stage2_gate_labels_2cam224"
  "output_dir=${JOB_DIR}"
  "base.checkpoint=${BASE_CHECKPOINT}"
  "base.expected_sha256=${BASE_SHA256}"
  "adapter.checkpoint=${ADAPTER_CHECKPOINT}"
  "adapter.expected_sha256=${ADAPTER_SHA256}"
  "assets.vae.path=${VAE_CHECKPOINT}"
  "assets.vae.expected_sha256=${VAE_SHA256}"
  "assets.normalization_stats.path=${NORMALIZATION_STATS}"
  "assets.normalization_stats.expected_sha256=${NORMALIZATION_STATS_SHA256}"
  "data.train.pretrained_norm_stats=${NORMALIZATION_STATS}"
  "data.train.video_backend=torchcodec"
  "data.train.strict_data_mode=true"
  "data_manifest.path=${DATA_MANIFEST}"
  "data_manifest.expected_sha256=${DATA_MANIFEST_SHA256}"
  "label_selection.directory=${SELECTION_DIR}"
  "label_selection.expected_sha256=${SELECTION_SHA256}"
  "label_coverage.tier=${COVERAGE_TIER}"
  "label_coverage.expected_sha256=${COVERAGE_SHA256}"
  "episode_split.path=${SELECTION_DIR}/episode_split.json"
  "episode_split.validation_fraction=0.1"
  "episode_split.split_seed=42"
  "episode_split.expected_assignment_sha256=${SPLIT_ASSIGNMENT_SHA256}"
  "labeling.base_seed=42"
  "labeling.num_seed_pairs=2"
  "labeling.relative_margin=0.05"
  "labeling.relative_gain_epsilon=1.0e-12"
  "labeling.num_inference_steps=10"
  "labeling.sigma_shift=null"
  "labeling.rand_device=cpu"
  "labeling.tiled=false"
  "labeling.num_shards=${NUM_SHARDS}"
  "labeling.chunk_size=${CHUNK_SIZE}"
  "labeling.shard_indices=null"
  "labeling.contract_file=label_contract.json"
  "labeling.runtime_config_file=label_runtime_config.json"
  "runtime.repo_dir=${FASTWAM_REPO_DIR}"
  "runtime.require_clean_git=true"
  "runtime.require_cuda=true"
  "runtime.device=cuda"
  "runtime.mixed_precision=bf16"
)

cat <<EOF
[stage2-labels]
  benchmark=LIBERO
  topology=1x8
  world_size=8
  rank_shards=rank_r_handles_r_r+8_..._r+56
  selection_sha256=${SELECTION_SHA256}
  coverage_tier=${COVERAGE_TIER}
  coverage_sha256=${COVERAGE_SHA256}
  expected_sample_ids_sha256=${EXPECTED_SAMPLE_IDS_SHA256}
  planned_sample_count=${EXPECTED_SAMPLE_COUNT}
  train_sample_count=${EXPECTED_TRAIN_SAMPLE_COUNT}
  validation_sample_count=${EXPECTED_VALIDATION_SAMPLE_COUNT}
  num_shards=${NUM_SHARDS}
  chunk_size=${CHUNK_SIZE}
  planned_chunk_count=${EXPECTED_CHUNK_COUNT}
  active_cohort_indices=${EXPECTED_ACTIVE_COHORTS}
  cohort_chunk_counts=${EXPECTED_COHORT_CHUNK_COUNTS}
  nonempty_cohort_shards=${EXPECTED_NONEMPTY_COHORT_SHARDS}
  rank_sample_counts=${EXPECTED_RANK_SAMPLE_COUNTS}
  rank_chunk_counts=${EXPECTED_RANK_CHUNK_COUNTS}
  adapter_step=30000
  adapter_sha256=${ADAPTER_SHA256}
  data_manifest_sha256=${DATA_MANIFEST_SHA256}
  ffmpeg_apt_version=${FFMPEG_APT_VERSION}
  ffmpeg_runtime_version=${FFMPEG_RUNTIME_VERSION}
  num_seed_pairs=2
  num_inference_steps=10
  run_id=${RUN_ID}
  attempt_id=${ATTEMPT_ID}
  output_dir=${JOB_DIR}
  attempt_log=${ATTEMPT_LOG}
  generation_success=${SUCCESS_PATH}
  resume=same_selection_and_git_campaign
  formal_merge_allowed_after_generation_success=true
  merge_completed=false
  next_step=python scripts/merge_gate_labels.py
EOF
print_command "${PREPARE_COMMAND[@]}"
print_command "${COMMAND[@]}"

if [[ "${FASTWAM_DRY_RUN}" == "1" ]]; then
  echo "[dry-run] formal 8-H100 label generation is fully planned; no files were written"
  exit 0
fi

preflight() {
  [[ -x "${PYTHON_BIN}" ]] || fail "missing Python environment: ${PYTHON_BIN}"
  [[ -x "${TORCHRUN_BIN}" ]] || fail "missing torchrun: ${TORCHRUN_BIN}"
  command -v setsid >/dev/null 2>&1 ||
    fail "setsid is required for reliable torchrun signal cleanup"
  command -v flock >/dev/null 2>&1 ||
    fail "flock is required to prevent overlapping formal label writers"
  [[ -f "${SCRIPT_DIR}/ensure_torchcodec_runtime.sh" ]] ||
    fail "missing TorchCodec runtime helper"
  local asset
  for asset in \
    "${BASE_CHECKPOINT}" \
    "${ADAPTER_CHECKPOINT}" \
    "${DATA_MANIFEST}" \
    "${NORMALIZATION_STATS}" \
    "${VAE_CHECKPOINT}"; do
    [[ -f "${asset}" ]] || fail "locked artifact is missing: ${asset}"
  done
  git -C "${FASTWAM_REPO_DIR}" diff --quiet || fail "tracked worktree is dirty"
  git -C "${FASTWAM_REPO_DIR}" diff --cached --quiet || fail "Git index is dirty"
  local untracked_source
  untracked_source="$(
    git -C "${FASTWAM_REPO_DIR}" ls-files --others --exclude-standard -- \
      src configs scripts tests
  )"
  [[ -z "${untracked_source}" ]] ||
    fail "untracked source/config/script/test files: ${untracked_source}"
}

cd "${FASTWAM_REPO_DIR}"
[[ -x "${PYTHON_BIN}" ]] || fail "missing Python environment: ${PYTHON_BIN}"
[[ -f "${FASTWAM_REPO_DIR}/scripts/prepare_gate_label_selection.py" ]] ||
  fail "missing selection preparation entrypoint"
echo "[prepare] creating or validating immutable nested selection artifacts"
"${PREPARE_COMMAND[@]}"
"${PYTHON_BIN}" - \
  "${SELECTION_DIR}" \
  "${DATA_MANIFEST}" \
  "${SELECTION_SHA256}" \
  "${COVERAGE_TIER}" \
  "${COVERAGE_SHA256}" \
  "${EXPECTED_SAMPLE_IDS_SHA256}" \
  "${SPLIT_ASSIGNMENT_SHA256}" \
  "${EXPECTED_SAMPLE_COUNT}" \
  "${EXPECTED_TRAIN_SAMPLE_COUNT}" \
  "${EXPECTED_VALIDATION_SAMPLE_COUNT}" \
  "${EXPECTED_ACTIVE_COHORTS}" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

from fastwam.gating.selection import load_selection_artifacts


(
    selection_dir_raw,
    data_manifest_raw,
    selection_sha256,
    coverage_tier,
    coverage_sha256,
    sample_ids_sha256,
    split_assignment_sha256,
    expected_sample_count_raw,
    expected_train_count_raw,
    expected_validation_count_raw,
    expected_active_cohorts_raw,
) = sys.argv[1:]
manifest = json.loads(Path(data_manifest_raw).read_text(encoding="utf-8"))
artifacts = load_selection_artifacts(
    selection_dir_raw,
    data_manifest=manifest,
)
descriptor = artifacts.descriptor
coverage = artifacts.coverages.get(coverage_tier)
if descriptor.get("selection_sha256") != selection_sha256:
    raise SystemExit("prepared selection SHA256 differs from formal contract")
if artifacts.episode_split.get("assignment_sha256") != split_assignment_sha256:
    raise SystemExit("prepared episode split differs from formal contract")
if not isinstance(coverage, dict):
    raise SystemExit(f"prepared selection lacks coverage tier {coverage_tier!r}")
if coverage.get("coverage_sha256") != coverage_sha256:
    raise SystemExit("prepared coverage SHA256 differs from formal contract")
if coverage.get("sample_ids_sha256") != sample_ids_sha256:
    raise SystemExit("prepared coverage sample IDs differ from formal contract")
if coverage.get("sample_count") != int(expected_sample_count_raw):
    raise SystemExit("prepared coverage sample count differs from formal contract")
expected_split_counts = {
    "train": int(expected_train_count_raw),
    "validation": int(expected_validation_count_raw),
}
if coverage.get("split_counts") != expected_split_counts:
    raise SystemExit("prepared coverage split counts differ from formal contract")
expected_active_cohorts = [
    int(value) for value in expected_active_cohorts_raw.split(",")
]
if coverage.get("active_cohort_indices") != expected_active_cohorts:
    raise SystemExit("prepared coverage cohorts differ from formal contract")
print(
    "[selection] validated "
    f"sha256={selection_sha256} tier={coverage_tier} "
    f"samples={coverage['sample_count']} cohorts={expected_active_cohorts}"
)
PY

# Selection preparation is metadata-only. Run the full environment, artifact,
# and clean-Git preflight only after its immutable receipts have been checked.
preflight

# shellcheck source=ensure_torchcodec_runtime.sh
source "${SCRIPT_DIR}/ensure_torchcodec_runtime.sh"

"${PYTHON_BIN}" - <<'PY'
import torch

names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
print(f"[gpu] count={len(names)} names={names}")
if len(names) != 8 or any("H100" not in name.upper() for name in names):
    raise SystemExit(f"expected exactly eight visible H100 GPUs, found {names}")
PY

# Existing JOB_DIR content is intentional: every existing immutable chunk is
# validated and resumed by the production label loader.
mkdir -p "${LOG_DIR}"
if [[ -e "${LOCK_PATH}" || -L "${LOCK_PATH}" ]]; then
  [[ -f "${LOCK_PATH}" && ! -L "${LOCK_PATH}" ]] ||
    fail "formal label lock path is not a regular file: ${LOCK_PATH}"
fi
exec 8>"${LOCK_PATH}"
flock -n 8 ||
  fail "could not acquire ${LOCK_PATH}; another launcher holds it or the shared filesystem lacks flock support"
if ! (set -o noclobber; : > "${ATTEMPT_LOG}") 2>/dev/null; then
  fail "attempt log already exists or cannot be atomically claimed; choose a unique ATTEMPT_ID: ${ATTEMPT_LOG}"
fi

LAUNCHER_PID="${BASHPID}"
progress_monitor() {
  # This helper must not keep the job lock alive if the parent is SIGKILLed.
  exec 8>&-
  while sleep 300; do
    kill -0 "${LAUNCHER_PID}" 2>/dev/null || exit 0
    local cohort_dir count partial_count
    count=0
    for cohort_dir in "${ACTIVE_COHORT_DIR_NAMES[@]}"; do
      [[ -d "${JOB_DIR}/${cohort_dir}" ]] || continue
      partial_count="$(find "${JOB_DIR}/${cohort_dir}" -mindepth 2 -maxdepth 2 -type f -name 'chunk-*.json' -print 2>/dev/null | wc -l | tr -d ' ')"
      count="$((count + partial_count))"
    done
    printf '[progress] time=%s chunks=%s/%s job_dir=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "${count}" \
      "${EXPECTED_CHUNK_COUNT}" \
      "${JOB_DIR}" | tee -a "${ATTEMPT_LOG}"
  done
}

MONITOR_PID=""
LAUNCH_PID=""
VERIFY_PID=""
TEE_PID=""
LOG_FD_OPEN="0"
cleanup_monitor() {
  if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
    MONITOR_PID=""
  fi
}
terminate_launch() {
  if [[ -n "${LAUNCH_PID}" ]]; then
    if kill -0 "${LAUNCH_PID}" 2>/dev/null; then
      # setsid makes LAUNCH_PID the process-group ID. Fall back to its single
      # PID only if the process group has already disappeared.
      kill -TERM -- "-${LAUNCH_PID}" 2>/dev/null ||
        kill -TERM "${LAUNCH_PID}" 2>/dev/null || true
    fi
    wait "${LAUNCH_PID}" 2>/dev/null || true
    LAUNCH_PID=""
  fi
}
terminate_verify() {
  if [[ -n "${VERIFY_PID}" ]]; then
    if kill -0 "${VERIFY_PID}" 2>/dev/null; then
      kill -TERM -- "-${VERIFY_PID}" 2>/dev/null ||
        kill -TERM "${VERIFY_PID}" 2>/dev/null || true
    fi
    wait "${VERIFY_PID}" 2>/dev/null || true
    VERIFY_PID=""
  fi
}
close_log_stream() {
  if [[ "${LOG_FD_OPEN}" == "1" ]]; then
    exec 9>&-
    LOG_FD_OPEN="0"
  fi
  if [[ -n "${TEE_PID}" ]]; then
    wait "${TEE_PID}" 2>/dev/null || true
    TEE_PID=""
  fi
}
cleanup_children() {
  local original_status="$?"
  cleanup_monitor
  terminate_launch
  close_log_stream
  terminate_verify
  return "${original_status}"
}
on_signal() {
  trap - INT TERM HUP
  exit 130
}
trap cleanup_children EXIT
trap on_signal INT TERM HUP

{
  echo "[attempt] id=${ATTEMPT_ID} git_commit=${GIT_COMMIT}"
  print_command "${COMMAND[@]}"
} | tee -a "${ATTEMPT_LOG}"

progress_monitor &
MONITOR_PID="$!"

# Keep torchrun in its own process group so a scheduler signal to this shell
# can terminate every local rank. A dedicated process-substitution FD lets us
# independently accept both the torchrun and tee exit statuses.
# The log consumer must likewise not own the generation lock; torchrun and
# the parent retain it until generation/postcheck have ended.
exec 9> >(exec 8>&-; exec tee -a "${ATTEMPT_LOG}")
TEE_PID="$!"
LOG_FD_OPEN="1"
setsid "${COMMAND[@]}" >&9 2>&1 &
LAUNCH_PID="$!"
set +e
wait "${LAUNCH_PID}"
LAUNCH_STATUS="$?"
set -e
LAUNCH_PID=""
cleanup_monitor
exec 9>&-
LOG_FD_OPEN="0"
set +e
wait "${TEE_PID}"
TEE_STATUS="$?"
set -e
TEE_PID=""
if [[ "${LAUNCH_STATUS}" != "0" ]]; then
  fail "torchrun failed with exit code ${LAUNCH_STATUS}; rerun the same command/RUN_ID to resume"
fi
if [[ "${TEE_STATUS}" != "0" ]]; then
  fail "tee failed with exit code ${TEE_STATUS}; generation status is not accepted"
fi

echo "[verify] rebuilding canonical cohort/shard plan and validating every chunk"
setsid "${PYTHON_BIN}" - \
  "${JOB_DIR}" \
  "${DATA_MANIFEST}" \
  "${SELECTION_DIR}" \
  "${GIT_COMMIT}" \
  "${BASE_SHA256}" \
  "${ADAPTER_SHA256}" \
  "${DATA_MANIFEST_SHA256}" \
  "${NORMALIZATION_STATS_SHA256}" \
  "${VAE_SHA256}" \
  "${SELECTION_SHA256}" \
  "${COVERAGE_TIER}" \
  "${COVERAGE_SHA256}" \
  "${EXPECTED_SAMPLE_IDS_SHA256}" \
  "${SPLIT_ASSIGNMENT_SHA256}" \
  "${EXPECTED_SAMPLE_COUNT}" \
  "${EXPECTED_TRAIN_SAMPLE_COUNT}" \
  "${EXPECTED_VALIDATION_SAMPLE_COUNT}" \
  "${EXPECTED_CHUNK_COUNT}" \
  "${EXPECTED_ACTIVE_COHORTS}" \
  "${EXPECTED_COHORT_CHUNK_COUNTS}" \
  "${EXPECTED_NONEMPTY_COHORT_SHARDS}" \
  "${EXPECTED_RANK_SAMPLE_COUNTS}" \
  "${EXPECTED_RANK_CHUNK_COUNTS}" \
  "${SUCCESS_PATH}" <<'PY' &
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import sys
import tempfile

from fastwam.alignment.checkpointing import canonical_json_sha256, sha256_file
from fastwam.gating.artifacts import (
    build_label_artifact_context,
    load_complete_label_chunk_from_context,
)
from fastwam.gating.label_job import iter_label_chunks
from fastwam.gating.selection import load_selection_artifacts


(
    job_dir_raw,
    data_manifest_raw,
    selection_dir_raw,
    git_commit,
    base_sha256,
    adapter_sha256,
    data_manifest_sha256,
    stats_sha256,
    vae_sha256,
    selection_sha256,
    coverage_tier,
    coverage_sha256,
    expected_sample_ids_sha256,
    split_assignment_sha256,
    expected_sample_count_raw,
    expected_train_count_raw,
    expected_validation_count_raw,
    expected_chunk_count_raw,
    expected_active_cohorts_raw,
    expected_cohort_chunk_counts_raw,
    expected_nonempty_cohort_shards_raw,
    expected_rank_sample_counts_raw,
    expected_rank_chunk_counts_raw,
    success_path_raw,
) = sys.argv[1:]
job_dir = Path(job_dir_raw).resolve()
data_manifest_path = Path(data_manifest_raw).resolve()
selection_dir = Path(selection_dir_raw).resolve()
success_path = Path(os.path.abspath(os.fspath(success_path_raw)))
expected_success_path = (
    job_dir / f"generation_success-{coverage_tier}-{coverage_sha256[:8]}.json"
)
if success_path != expected_success_path:
    raise SystemExit(f"unexpected success marker path: {success_path}")
if success_path.parent.resolve(strict=True) != job_dir:
    raise SystemExit(f"success marker parent contains a symlink: {success_path.parent}")
expected_sample_count = int(expected_sample_count_raw)
expected_train_count = int(expected_train_count_raw)
expected_validation_count = int(expected_validation_count_raw)
expected_chunk_count = int(expected_chunk_count_raw)
expected_active_cohorts = [
    int(value) for value in expected_active_cohorts_raw.split(",")
]
expected_cohort_chunk_counts = {
    int(cohort): int(count)
    for cohort, count in (
        value.split(":", maxsplit=1)
        for value in expected_cohort_chunk_counts_raw.split(",")
    )
}
expected_nonempty_cohort_shards = int(expected_nonempty_cohort_shards_raw)
expected_rank_sample_counts = [
    int(value) for value in expected_rank_sample_counts_raw.split(",")
]
expected_rank_chunk_counts = [
    int(value) for value in expected_rank_chunk_counts_raw.split(",")
]


def load_mapping(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"{label} is unreadable: {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must contain a JSON object: {path}")
    return value


manifest = load_mapping(data_manifest_path, "data manifest")
selection = load_selection_artifacts(
    selection_dir,
    data_manifest=manifest,
)
split = dict(selection.episode_split)
descriptor = selection.descriptor
coverage = selection.coverages.get(coverage_tier)
if descriptor.get("selection_sha256") != selection_sha256:
    raise SystemExit("formal label selection SHA256 mismatch")
if not isinstance(coverage, dict):
    raise SystemExit(f"formal selection lacks coverage tier {coverage_tier!r}")
if coverage.get("selection_sha256") != selection_sha256:
    raise SystemExit("formal coverage does not bind the selection")
if coverage.get("coverage_sha256") != coverage_sha256:
    raise SystemExit("formal label coverage SHA256 mismatch")
if coverage.get("sample_count") != expected_sample_count:
    raise SystemExit("formal label coverage sample count mismatch")
if coverage.get("sample_ids_sha256") != expected_sample_ids_sha256:
    raise SystemExit("formal label coverage sample IDs SHA256 mismatch")
if coverage.get("split_counts") != {
    "train": expected_train_count,
    "validation": expected_validation_count,
}:
    raise SystemExit("formal label coverage split counts mismatch")
if coverage.get("active_cohort_indices") != expected_active_cohorts:
    raise SystemExit("formal label coverage active cohorts mismatch")
if split.get("assignment_sha256") != split_assignment_sha256:
    raise SystemExit("formal label episode split assignment SHA256 mismatch")

contract = load_mapping(job_dir / "label_contract.json", "label contract")
runtime_config = load_mapping(
    job_dir / "label_runtime_config.json", "label runtime config"
)
expected_contract_fields = {
    "kind": "stage2_gate_label_contract",
    "schema_version": 2,
    "base_checkpoint_sha256": base_sha256,
    "adapter_checkpoint_sha256": adapter_sha256,
    "data_manifest_sha256": data_manifest_sha256,
    "normalization_stats_sha256": stats_sha256,
    "vae_sha256": vae_sha256,
    "episode_assignment_sha256": split_assignment_sha256,
    "base_seed": 42,
    "num_seed_pairs": 2,
    "relative_margin": 0.05,
    "relative_gain_epsilon": 1e-12,
    "num_inference_steps": 10,
    "sigma_shift": None,
    "rand_device": "cpu",
    "tiled": False,
    "num_shards": 64,
    "chunk_size": 64,
    "chunk_plan_algorithm": (
        "sample_id_sorted_fixed_chunks_per_cohort_shard_v1"
    ),
    "shard_algorithm": "sample_sha256_prefix_mod_v1",
    "seed_algorithm": "stage2_pair_seed_v1",
    "label_rule": "e10_lt_one_minus_margin_times_e0_v1",
}
for field, expected in expected_contract_fields.items():
    actual = contract.get(field)
    if actual != expected:
        raise SystemExit(
            f"formal label contract mismatch for {field}: "
            f"expected={expected!r}, actual={actual!r}"
        )
expected_git_identity = {
    "commit": git_commit,
    "tracked_dirty": False,
    "untracked_source_files": [],
}
if contract.get("git_identity") != expected_git_identity:
    raise SystemExit("formal label contract Git identity mismatch")
if manifest.get("manifest_sha256") != data_manifest_sha256:
    raise SystemExit("formal label data manifest SHA256 mismatch")
if runtime_config.get("kind") != "stage2_label_runtime_config":
    raise SystemExit("formal label runtime kind mismatch")
if runtime_config.get("mixed_precision") != "bf16":
    raise SystemExit("formal label runtime mixed precision mismatch")
if runtime_config.get("model", {}).get("_target_") != (
    "fastwam.runtime.create_fastwam_unified_aligned"
):
    raise SystemExit("formal label runtime model target mismatch")
if contract.get("label_runtime_config_sha256") != canonical_json_sha256(
    runtime_config
):
    raise SystemExit("formal label runtime config SHA256 mismatch")

context = build_label_artifact_context(
    contract=contract,
    data_manifest=manifest,
    episode_split=split,
)


def lexical_chunk_path(path_value: Path) -> tuple[Path, Path]:
    # Preserve the lexical path until lstat has rejected the leaf and every
    # ancestor symlink. Resolving first would silently accept a redirected file.
    path = Path(os.path.abspath(os.fspath(path_value)))
    try:
        relative_path = path.relative_to(job_dir)
    except ValueError as error:
        raise SystemExit(f"planned chunk escapes label job: {path}") from error
    return path, relative_path


def validate_chunk(
    path: Path,
    *,
    cohort_index: int,
    shard_index: int,
    chunk_index: int,
    planned_sample_ids: tuple[str, ...],
) -> tuple[Path, int]:
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SystemExit(f"planned chunk is not a regular file: {path}")
    resolved_path = path.resolve(strict=True)
    try:
        resolved_path.relative_to(job_dir)
    except ValueError as error:
        raise SystemExit(f"planned chunk resolves outside label job: {path}") from error
    if resolved_path != path:
        raise SystemExit(f"planned chunk path contains a symlink: {path}")
    chunk = load_complete_label_chunk_from_context(
        path,
        context=context,
        planned_sample_ids=planned_sample_ids,
        selection_sha256=selection_sha256,
        cohort_index=cohort_index,
    )
    if (
        chunk.get("cohort_index") != cohort_index
        or chunk.get("shard_index") != shard_index
        or chunk.get("chunk_index") != chunk_index
    ):
        raise SystemExit(f"chunk coordinates disagree with its plan: {path}")
    rows = chunk.get("rows")
    if not isinstance(rows, list):
        raise SystemExit(f"chunk rows are invalid: {path}")
    return path.relative_to(job_dir), len(rows)


expected_paths: set[Path] = set()
active_bindings: dict[Path, tuple[int, int, int, tuple[str, ...]]] = {}
missing: list[str] = []
chunk_inventory: list[dict[str, object]] = []
planned_sample_ids: list[str] = []
rank_sample_counts = [0] * 8
rank_chunk_counts = [0] * 8
cohort_chunk_counts = {cohort: 0 for cohort in expected_active_cohorts}
nonempty_cohort_shards: set[tuple[int, int]] = set()
planned_samples = 0
planned_chunks = 0
validated_rows = 0

plans = iter_label_chunks(
    context=context,
    output_dir=job_dir,
    chunk_size=64,
    shard_indices=None,
    selection_artifacts=selection,
    coverage_tier=coverage_tier,
)
try:
    for plan in plans:
        if plan.cohort_index is None:
            raise SystemExit("formal cohort plan lacks cohort_index")
        planned_chunks += 1
        planned_samples += len(plan.samples)
        planned_sample_ids.extend(plan.planned_sample_ids)
        rank = plan.shard_index % 8
        rank_sample_counts[rank] += len(plan.samples)
        rank_chunk_counts[rank] += 1
        if plan.cohort_index not in cohort_chunk_counts:
            raise SystemExit(
                f"formal plan emitted an inactive cohort: {plan.cohort_index}"
            )
        cohort_chunk_counts[plan.cohort_index] += 1
        nonempty_cohort_shards.add((plan.cohort_index, plan.shard_index))
        path, relative_path = lexical_chunk_path(plan.path)
        binding = (
            plan.cohort_index,
            plan.shard_index,
            plan.chunk_index,
            plan.planned_sample_ids,
        )
        if path in active_bindings:
            raise SystemExit(f"canonical plan repeats a chunk path: {path}")
        active_bindings[path] = binding
        expected_paths.add(path)
        if not os.path.lexists(path):
            missing.append(str(path))
            continue
        relative_path, row_count = validate_chunk(
            path,
            cohort_index=plan.cohort_index,
            shard_index=plan.shard_index,
            chunk_index=plan.chunk_index,
            planned_sample_ids=plan.planned_sample_ids,
        )
        validated_rows += row_count
        chunk_inventory.append(
            {
                "relative_path": relative_path.as_posix(),
                "chunk_sha256": sha256_file(path),
                "row_count": row_count,
            }
        )
finally:
    close = getattr(plans, "close", None)
    if callable(close):
        close()

if planned_samples != expected_sample_count:
    raise SystemExit(f"canonical plan sample count mismatch: {planned_samples}")
if planned_chunks != expected_chunk_count:
    raise SystemExit(f"canonical plan chunk count mismatch: {planned_chunks}")
if rank_sample_counts != expected_rank_sample_counts:
    raise SystemExit(
        f"canonical per-rank sample counts mismatch: {rank_sample_counts}"
    )
if rank_chunk_counts != expected_rank_chunk_counts:
    raise SystemExit(
        f"canonical per-rank chunk counts mismatch: {rank_chunk_counts}"
    )
if cohort_chunk_counts != expected_cohort_chunk_counts:
    raise SystemExit(
        f"canonical per-cohort chunk counts mismatch: {cohort_chunk_counts}"
    )
if len(nonempty_cohort_shards) != expected_nonempty_cohort_shards:
    raise SystemExit(
        "canonical nonempty cohort-shard count mismatch: "
        f"{len(nonempty_cohort_shards)}"
    )
if len(planned_sample_ids) != len(set(planned_sample_ids)):
    raise SystemExit("canonical formal plan repeats sample IDs")
if canonical_json_sha256(sorted(planned_sample_ids)) != expected_sample_ids_sha256:
    raise SystemExit("canonical formal plan sample IDs SHA256 mismatch")
if missing:
    raise SystemExit(
        f"formal label chunks are missing ({len(missing)}): {missing[:20]}"
    )

# A campaign directory is keyed by selection+Git, not coverage. Preserve chunks
# from known inactive cohorts so formal -> cap expansion resumes already-computed
# work, but reject any path that is not part of the immutable master selection.
all_cohort_indices = {int(row["cohort_index"]) for row in selection.rows}
master_tiers = [
    name
    for name, candidate in selection.coverages.items()
    if set(candidate["active_cohort_indices"]) == all_cohort_indices
]
if len(master_tiers) != 1:
    raise SystemExit("selection must expose exactly one full master coverage tier")
master_tier = master_tiers[0]
known_bindings: dict[Path, tuple[int, int, int, tuple[str, ...]]] = {}
master_plans = iter_label_chunks(
    context=context,
    output_dir=job_dir,
    chunk_size=64,
    shard_indices=None,
    selection_artifacts=selection,
    coverage_tier=master_tier,
)
try:
    for plan in master_plans:
        if plan.cohort_index is None:
            raise SystemExit("master cohort plan lacks cohort_index")
        path, _ = lexical_chunk_path(plan.path)
        binding = (
            plan.cohort_index,
            plan.shard_index,
            plan.chunk_index,
            plan.planned_sample_ids,
        )
        if path in known_bindings:
            raise SystemExit(f"master selection repeats a chunk path: {path}")
        known_bindings[path] = binding
finally:
    close = getattr(master_plans, "close", None)
    if callable(close):
        close()
for path, binding in active_bindings.items():
    if known_bindings.get(path) != binding:
        raise SystemExit(f"formal plan is not nested in master selection: {path}")

discovered_paths: set[Path] = set()
for candidate in job_dir.rglob("chunk-*.json"):
    path, _ = lexical_chunk_path(candidate)
    if not os.path.lexists(path):
        continue
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SystemExit(f"discovered chunk is not a regular file: {path}")
    resolved_path = path.resolve(strict=True)
    try:
        resolved_path.relative_to(job_dir)
    except ValueError as error:
        raise SystemExit(f"discovered chunk resolves outside label job: {path}") from error
    if resolved_path != path:
        raise SystemExit(f"discovered chunk path contains a symlink: {path}")
    discovered_paths.add(path)
unknown = sorted(str(path) for path in discovered_paths - set(known_bindings))
if unknown:
    raise SystemExit(
        f"formal label job contains unknown chunks ({len(unknown)}): {unknown[:20]}"
    )
undiscovered_active = sorted(str(path) for path in expected_paths - discovered_paths)
if undiscovered_active:
    raise SystemExit(
        "formal label discovered chunk set lacks active chunks: "
        f"{undiscovered_active[:20]}"
    )
for path in sorted(discovered_paths - expected_paths):
    cohort_index, shard_index, chunk_index, sample_ids = known_bindings[path]
    validate_chunk(
        path,
        cohort_index=cohort_index,
        shard_index=shard_index,
        chunk_index=chunk_index,
        planned_sample_ids=sample_ids,
    )

if validated_rows != expected_sample_count:
    raise SystemExit(f"validated label row count mismatch: {validated_rows}")
if len(chunk_inventory) != expected_chunk_count:
    raise SystemExit("formal chunk inventory count differs from canonical plan")
chunk_inventory_sha256 = canonical_json_sha256(chunk_inventory)

# This marker contains only immutable formal-tier identity and coverage. It is
# byte stable if later coverage tiers add known inactive cohort chunks.
success = {
    "active_cohort_indices": expected_active_cohorts,
    "adapter_checkpoint_sha256": adapter_sha256,
    "benchmark": "LIBERO",
    "chunk_count": planned_chunks,
    "chunk_inventory_sha256": chunk_inventory_sha256,
    "contract_sha256": contract["contract_sha256"],
    "coverage_sha256": coverage_sha256,
    "coverage_tier": coverage_tier,
    "data_manifest_sha256": data_manifest_sha256,
    "formal_merge_allowed": True,
    "git_commit": git_commit,
    "kind": "stage2_label_generation_success",
    "merge_completed": False,
    "num_shards": 64,
    "planned_chunk_count": planned_chunks,
    "planned_sample_count": planned_samples,
    "sample_count": validated_rows,
    "sample_ids_sha256": expected_sample_ids_sha256,
    "schema_version": 2,
    "selection_sha256": selection_sha256,
}
encoded = (json.dumps(success, indent=2, sort_keys=True) + "\n").encode("utf-8")
descriptor, temporary_raw = tempfile.mkstemp(
    dir=success_path.parent,
    prefix=f".{success_path.name}.partial.",
)
temporary = Path(temporary_raw)
try:
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, success_path)
    except FileExistsError:
        no_follow = getattr(os, "O_NOFOLLOW", None)
        if no_follow is None:
            raise SystemExit("O_NOFOLLOW is required for the success marker")
        flags = os.O_RDONLY | os.O_NONBLOCK | no_follow
        try:
            existing_fd = os.open(success_path, flags)
        except OSError as error:
            raise SystemExit(
                f"refusing unsafe existing success marker: {success_path}"
            ) from error
        with os.fdopen(existing_fd, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise SystemExit(
                    f"refusing non-regular existing success marker: {success_path}"
                )
            existing_bytes = stream.read()
            if existing_bytes != encoded:
                raise SystemExit(
                    f"existing success marker differs from verified job: {success_path}"
                )
            if metadata.st_nlink != 1:
                # A crash after link(2) published the marker but before the
                # temporary name was removed can leave exactly two names for
                # the same inode. Recover only that fully verified, local
                # residue; any other hard-link topology remains fail-closed.
                partial_prefix = f".{success_path.name}.partial."
                residue_candidates: list[Path] = []
                with os.scandir(success_path.parent) as entries:
                    for entry in entries:
                        if not entry.name.startswith(partial_prefix):
                            continue
                        candidate_metadata = entry.stat(follow_symlinks=False)
                        if (
                            stat.S_ISREG(candidate_metadata.st_mode)
                            and candidate_metadata.st_dev == metadata.st_dev
                            and candidate_metadata.st_ino == metadata.st_ino
                        ):
                            residue_candidates.append(Path(entry.path))
                if metadata.st_nlink != 2 or len(residue_candidates) != 1:
                    raise SystemExit(
                        "refusing unsafe hard-linked success marker: "
                        f"{success_path}"
                    )
                residue = residue_candidates[0]
                residue.unlink()
                refreshed = os.fstat(stream.fileno())
                if refreshed.st_nlink != 1:
                    raise SystemExit(
                        "success marker crash-residue recovery did not "
                        f"restore one link: {success_path}"
                    )
finally:
    temporary.unlink(missing_ok=True)
directory_no_follow = getattr(os, "O_NOFOLLOW", None)
if directory_no_follow is None:
    raise SystemExit("O_NOFOLLOW is required for the success marker directory")
directory_flags = (
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | directory_no_follow
)
directory_fd = os.open(success_path.parent, directory_flags)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)

print(
    "[verify] canonical formal label job passed: "
    f"samples={validated_rows} chunks={planned_chunks} "
    f"cohorts={expected_active_cohorts} shards=64"
)
print(f"[verify] generation_success={success_path}")
print(f"[verify] chunk_inventory_sha256={chunk_inventory_sha256}")
print("[verify] formal_merge_allowed=true merge_completed=false")
PY
VERIFY_PID="$!"
set +e
wait "${VERIFY_PID}"
VERIFY_STATUS="$?"
set -e
VERIFY_PID=""
if [[ "${VERIFY_STATUS}" != "0" ]]; then
  fail "canonical label verification failed with exit code ${VERIFY_STATUS}"
fi


trap - EXIT INT TERM HUP
flock -u 8
exec 8>&-
echo "[ok] LIBERO Stage 2 formal label generation is complete"
echo "[ok] generation_success=${SUCCESS_PATH}"
echo "[next] run the separate strict merge step before Gate training"
