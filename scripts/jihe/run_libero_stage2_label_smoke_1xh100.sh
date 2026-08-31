#!/usr/bin/env bash
set -euo pipefail

# One-sample, one-H100 acceptance smoke for LIBERO Stage 2 E0/E10 label
# generation.  The sparse shard contract is deliberately smoke-only and must
# never be merged into the formal 64-shard label set.

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
  fail "this one-click launcher takes no arguments; use environment variables only for runtime/output selection"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
FASTWAM_REPO_DIR="${FASTWAM_REPO_DIR:-${DEFAULT_REPO_DIR}}"
FASTWAM_ENV="${FASTWAM_ENV:-/root/.venvs/fastwam}"
PYTHON_BIN="${FASTWAM_ENV}/bin/python"
FASTWAM_DRY_RUN="${FASTWAM_DRY_RUN:-0}"
RUN_ID="${RUN_ID:-$(date -u +%Y-%m-%d_%H-%M-%S)}"
SMOKE_OUTPUT_BASE="${SMOKE_OUTPUT_BASE:-/root/feihong/FastWAM/formal_runs/smokes/stage2}"
VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-0}"

# Frozen LIBERO Stage 3 final identities.  These are intentionally not
# overridable: a path remap or a new artifact requires a reviewed launcher.
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
readonly SPLIT_ASSIGNMENT_SHA256="78bd013dcd49dcafb01898e4c1e8ac5d00c26bee81536a1b5ff40aebd2098704"
readonly NUM_SHARDS="1048576"
readonly SHARD_INDEX="780575"
readonly EXPECTED_SAMPLE_ID="11a8900dcffbe91f4cd0b56128430af4e45cdb61f76864be00317747db3dcc4c"

[[ "${FASTWAM_DRY_RUN}" == "0" || "${FASTWAM_DRY_RUN}" == "1" ]] ||
  fail "FASTWAM_DRY_RUN must be 0 or 1"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] ||
  fail "RUN_ID may contain only letters, digits, dot, underscore, and hyphen"
[[ "${SMOKE_OUTPUT_BASE}" == /* ]] ||
  fail "SMOKE_OUTPUT_BASE must be an absolute path"
[[ -d "${FASTWAM_REPO_DIR}" ]] ||
  fail "FASTWAM_REPO_DIR does not exist: ${FASTWAM_REPO_DIR}"
FASTWAM_REPO_DIR="$(cd -- "${FASTWAM_REPO_DIR}" && pwd -P)"
[[ "${FASTWAM_REPO_DIR}" == "${DEFAULT_REPO_DIR}" ]] ||
  fail "FASTWAM_REPO_DIR must resolve to the checkout containing this launcher"

GIT_COMMIT="$(git -C "${FASTWAM_REPO_DIR}" rev-parse HEAD)"
GIT_SHORT="${GIT_COMMIT:0:7}"
RAW_SMOKE_ROOT="${FASTWAM_SMOKE_ROOT:-${SMOKE_OUTPUT_BASE}/libero_1xh100_${GIT_SHORT}_${RUN_ID}}"
[[ "${RAW_SMOKE_ROOT}" == /* ]] ||
  fail "FASTWAM_SMOKE_ROOT must be an absolute path"
PERSISTENT_ROOT="$(realpath -m -- /root/feihong/FastWAM)"
SMOKE_ROOT="$(realpath -m -- "${RAW_SMOKE_ROOT}")"
if [[ "${FASTWAM_DRY_RUN}" == "0" ]]; then
  [[ "${SMOKE_ROOT}" == "${PERSISTENT_ROOT}/"* ]] ||
    fail "FASTWAM_SMOKE_ROOT must stay under ${PERSISTENT_ROOT}"
fi

JOB_DIR="${SMOKE_ROOT}/label_job"
FRESH_LOG="${SMOKE_ROOT}/fresh.log"
RESUME_LOG="${SMOKE_ROOT}/resume.log"
FRESH_RECEIPT="${SMOKE_ROOT}/fresh_verification_receipt.json"
RESUME_RECEIPT="${SMOKE_ROOT}/resume_verification_receipt.json"
VERIFY_LOG="${SMOKE_ROOT}/verification.log"
VERIFIER="${FASTWAM_REPO_DIR}/scripts/verify_libero_stage2_label_smoke.py"

# The task config resolves these variables, while the duplicated CLI values
# below make every identity visible in the launch record and fail closed if the
# task defaults drift.
export FASTWAM_LIBERO_STAGE2_LABEL_JOB="${JOB_DIR}"
export FASTWAM_LIBERO_STAGE3_BASE_CHECKPOINT="${BASE_CHECKPOINT}"
export FASTWAM_LIBERO_STAGE3_BASE_SHA256="${BASE_SHA256}"
export FASTWAM_LIBERO_STAGE3_ADAPTER="${ADAPTER_CHECKPOINT}"
export FASTWAM_LIBERO_STAGE3_ADAPTER_SHA256="${ADAPTER_SHA256}"
export FASTWAM_LIBERO_STAGE3_DATA_MANIFEST="${DATA_MANIFEST}"
export FASTWAM_LIBERO_STAGE3_DATA_MANIFEST_SHA256="${DATA_MANIFEST_SHA256}"
export FASTWAM_LIBERO_STAGE3_VAE="${VAE_CHECKPOINT}"
export FASTWAM_LIBERO_STATS="${NORMALIZATION_STATS}"

export CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}"
export PATH="${FASTWAM_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${FASTWAM_REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DIFFSYNTH_MODEL_BASE_PATH="/root/feihong/FastWAM/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD="true"
export HF_HUB_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

# An inherited distributed environment would make an explicit shard subset
# invalid.  This smoke is intentionally one process on one visible H100.
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK

COMMAND=(
  "${PYTHON_BIN}"
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
  "episode_split.path=${JOB_DIR}/episode_split.json"
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
  "labeling.chunk_size=1"
  "labeling.shard_indices=[${SHARD_INDEX}]"
  "labeling.contract_file=label_contract.json"
  "labeling.runtime_config_file=label_runtime_config.json"
  "runtime.repo_dir=${FASTWAM_REPO_DIR}"
  "runtime.require_clean_git=true"
  "runtime.require_cuda=true"
  "runtime.device=cuda"
  "runtime.mixed_precision=bf16"
)

VERIFY_BASE=(
  "${PYTHON_BIN}"
  "${VERIFIER}"
  --job-dir "${JOB_DIR}"
  --expected-git-commit "${GIT_COMMIT}"
  --data-manifest "${DATA_MANIFEST}"
)

preflight() {
  [[ -x "${PYTHON_BIN}" ]] || fail "missing Python environment: ${PYTHON_BIN}"
  [[ -f "${VERIFIER}" ]] || fail "missing Stage 2 smoke verifier: ${VERIFIER}"
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

verify_counts() {
  local log_path="$1"
  local phase="$2"
  local expected_written="$3"
  local expected_resumed="$4"
  local expected_inferred="$5"
  "${PYTHON_BIN}" - \
    "${log_path}" \
    "${phase}" \
    "${expected_written}" \
    "${expected_resumed}" \
    "${expected_inferred}" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
phase = sys.argv[2]
expected = (1, 1, int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
pattern = re.compile(
    r"LabelJobResult\(planned_chunk_count=(\d+), "
    r"planned_sample_count=(\d+), written_chunk_count=(\d+), "
    r"resumed_chunk_count=(\d+), inferred_sample_count=(\d+),"
)
matches = [tuple(map(int, row)) for row in pattern.findall(path.read_text(encoding="utf-8"))]
if matches != [expected]:
    raise SystemExit(
        f"{phase} label counts mismatch: expected one {expected}, observed {matches}"
    )
print(f"[verify] phase={phase} counts={expected}")
PY
}

run_phase() {
  local phase="$1"
  local log_path="$2"
  local expected_written="$3"
  local expected_resumed="$4"
  local expected_inferred="$5"
  echo "[smoke] phase=${phase} job_dir=${JOB_DIR}"
  print_command "${COMMAND[@]}"
  if [[ "${FASTWAM_DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "${COMMAND[@]}" 2>&1 | tee "${log_path}"
  verify_counts \
    "${log_path}" \
    "${phase}" \
    "${expected_written}" \
    "${expected_resumed}" \
    "${expected_inferred}"
}

run_verifier() {
  local phase="$1"
  local receipt="$2"
  local command=("${VERIFY_BASE[@]}" --receipt "${receipt}")
  echo "[smoke] phase=verify-${phase}"
  print_command "${command[@]}"
  "${command[@]}" | tee -a "${VERIFY_LOG}"
}

compare_receipts() {
  "${PYTHON_BIN}" - "${FRESH_RECEIPT}" "${RESUME_RECEIPT}" <<'PY'
from pathlib import Path
import json
import sys

fresh = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
resumed = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
for name, payload in (("fresh", fresh), ("resume", resumed)):
    if payload.get("status") != "pass":
        raise SystemExit(f"{name} verifier did not pass")
    if payload.get("formal_merge_allowed") is not False:
        raise SystemExit(f"{name} verifier did not forbid formal merge")
if fresh.get("artifact_sha256") != resumed.get("artifact_sha256"):
    raise SystemExit("fresh/resume artifact SHA256 maps differ")
for field in ("contract_sha256", "chunk_sha256", "sample_id", "e0", "e10", "label", "sample_weight"):
    if fresh.get(field) != resumed.get(field):
        raise SystemExit(f"fresh/resume receipt field differs: {field}")
print("[verify] fresh/resume artifact SHA256 maps are identical")
print("[verify] formal_merge_allowed=false")
PY
}

cat <<EOF
[stage2-label-smoke]
  benchmark=LIBERO
  workflow=fresh_singleton -> strict_resume_same_directory -> exact_verification
  gpu_count=1
  adapter_step=30000
  adapter_sha256=${ADAPTER_SHA256}
  data_manifest_sha256=${DATA_MANIFEST_SHA256}
  num_seed_pairs=2
  num_inference_steps=10
  num_shards=${NUM_SHARDS}
  shard_index=${SHARD_INDEX}
  chunk_size=1
  expected_sample_id=${EXPECTED_SAMPLE_ID}
  smoke_root=${SMOKE_ROOT}
  label_job_dir=${JOB_DIR}
  git_commit=${GIT_COMMIT}
  formal_merge_allowed=false
EOF

echo "[warning] NON-MERGEABLE smoke artifacts; never mix them with the formal 64-shard label job"

if [[ "${FASTWAM_DRY_RUN}" == "1" ]]; then
  echo "[smoke] phase=fresh job_dir=${JOB_DIR}"
  print_command "${COMMAND[@]}"
  echo "[smoke] phase=verification-fresh"
  print_command "${VERIFY_BASE[@]}" --receipt "${FRESH_RECEIPT}"
  echo "[smoke] phase=resume job_dir=${JOB_DIR}"
  print_command "${COMMAND[@]}"
  echo "[smoke] phase=verification-resume"
  print_command "${VERIFY_BASE[@]}" --receipt "${RESUME_RECEIPT}"
  echo "[dry-run] fresh, resume, and exact verification are fully planned; no files were written"
  exit 0
fi

[[ ! -e "${SMOKE_ROOT}" ]] ||
  fail "SMOKE_ROOT already exists; use a new RUN_ID or SMOKE_ROOT: ${SMOKE_ROOT}"
preflight
cd "${FASTWAM_REPO_DIR}"

# shellcheck source=ensure_torchcodec_runtime.sh
source "${SCRIPT_DIR}/ensure_torchcodec_runtime.sh"

"${PYTHON_BIN}" - <<'PY'
import torch

names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print(f"[gpu] visible={names}")
if len(names) != 1 or "H100" not in names[0].upper():
    raise SystemExit(f"expected exactly one visible H100, found {names}")
PY

mkdir -p "$(dirname -- "${SMOKE_ROOT}")"
mkdir "${SMOKE_ROOT}"
mkdir "${JOB_DIR}"

run_phase "fresh" "${FRESH_LOG}" 1 0 1
run_verifier "fresh" "${FRESH_RECEIPT}"
run_phase "resume" "${RESUME_LOG}" 0 1 0
run_verifier "resume" "${RESUME_RECEIPT}"
compare_receipts

PARTIAL_SUCCESS="${SMOKE_ROOT}/SUCCESS.partial.$$"
printf "benchmark=LIBERO\ngit_commit=%s\njob_dir=%s\nreceipt=%s\nformal_merge_allowed=false\n" \
  "${GIT_COMMIT}" "${JOB_DIR}" "${RESUME_RECEIPT}" > "${PARTIAL_SUCCESS}"
mv "${PARTIAL_SUCCESS}" "${SMOKE_ROOT}/SUCCESS"

echo "[ok] LIBERO Stage 2 singleton label smoke passed"
echo "[ok] receipt=${RESUME_RECEIPT}"
echo "[ok] formal_merge_allowed=false"
echo "[warning] NON-MERGEABLE smoke artifacts; never mix them with the formal 64-shard label job"
