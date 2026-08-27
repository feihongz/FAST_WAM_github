#!/usr/bin/env bash
set -euo pipefail

# Internal one-job orchestrator used by the two public JiHe smoke entrypoints.

fail() {
  echo "[error] $*" >&2
  exit 1
}

print_command() {
  printf "[launch]"
  printf " %q" "$@"
  printf "\n"
}

[[ "$#" == "1" ]] || fail "usage: $0 {libero|robotwin}"
BENCHMARK="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
FASTWAM_REPO_DIR="${FASTWAM_REPO_DIR:-${DEFAULT_REPO_DIR}}"
FASTWAM_ENV="${FASTWAM_ENV:-/root/.venvs/fastwam}"
FASTWAM_STORAGE_ROOT="${FASTWAM_STORAGE_ROOT:-/root/feihong}"
SMOKE_OUTPUT_BASE="${SMOKE_OUTPUT_BASE:-${FASTWAM_STORAGE_ROOT}/FastWAM/formal_runs/smokes/stage3}"
RUN_ID="${RUN_ID:-$(date -u +%Y-%m-%d_%H-%M-%S)}"
FASTWAM_DRY_RUN="${FASTWAM_DRY_RUN:-0}"
VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
CANONICAL_VAE="/root/feihong/FastWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
ACTIVE_CHILD_PID=""

forward_signal() {
  local signal="$1"
  local exit_status="$2"
  trap - TERM INT
  if [[ -n "${ACTIVE_CHILD_PID}" ]] && kill -0 "${ACTIVE_CHILD_PID}" 2>/dev/null; then
    kill -s "${signal}" -- "${ACTIVE_CHILD_PID}" 2>/dev/null || true
    wait "${ACTIVE_CHILD_PID}" 2>/dev/null || true
  fi
  exit "${exit_status}"
}

trap 'forward_signal TERM 143' TERM
# A background child can inherit SIGINT ignored from a non-interactive shell.
# Terminate it with SIGTERM while preserving the conventional wrapper status.
trap 'forward_signal TERM 130' INT

case "${BENCHMARK}" in
  libero)
    BENCHMARK_LABEL="LIBERO"
    OUTPUT_PREFIX="libero_8gpu"
    TRAIN_LAUNCHER="${SCRIPT_DIR}/train_libero_stage3_alignment_8xh100.sh"
    FRESH_MASTER_PORT="${MASTER_PORT:-29531}"
    RESUME_MASTER_PORT="${RESUME_MASTER_PORT:-29533}"
    EXPECTED_BASE_CHECKPOINT_SHA256="17a5588cc2b8d162219c9daf818614f614ee4a7921933a4a26c5d678111330e9"
    EXPECTED_DATA_MANIFEST_SHA256="08da49109a57b55c67f3fa4ac31fbfa44e44dd541a194a5d3420838537d0d320"
    LOCKED_TASK_ENV=(
      "FASTWAM_STAGE3_BASE_CHECKPOINT=/root/feihong/FastWAM/formal_runs/FAST_WAM_github/libero_unified_shared_2cam224_1e-4/2026-07-01_00-44-20/checkpoints/weights/latest.pt"
      "FASTWAM_STAGE3_BASE_SHA256=${EXPECTED_BASE_CHECKPOINT_SHA256}"
      "FASTWAM_STAGE3_VAE=${CANONICAL_VAE}"
      "FASTWAM_STAGE3_DATA_MANIFEST=/root/feihong/FastWAM/formal_runs/contracts/stage3/libero_current_273465f_1693e/libero_stage3_data_manifest.json"
      "FASTWAM_STAGE3_DATA_MANIFEST_SHA256=${EXPECTED_DATA_MANIFEST_SHA256}"
      "FASTWAM_LIBERO_STATS=/root/feihong/FastWAM/formal_runs/FAST_WAM_github/libero_unified_shared_2cam224_1e-4/2026-07-01_00-44-20/dataset_stats.json"
      "FASTWAM_STAGE3_EXPECTED_DATASET_LENGTH=273465"
      "FASTWAM_STAGE3_EXPECTED_DATASET_EPISODES=1693"
    )
    ;;
  robotwin)
    BENCHMARK_LABEL="RoboTwin-2.0"
    OUTPUT_PREFIX="robotwin_8gpu"
    TRAIN_LAUNCHER="${SCRIPT_DIR}/train_robotwin_stage3_alignment_8xh100.sh"
    FRESH_MASTER_PORT="${MASTER_PORT:-29532}"
    RESUME_MASTER_PORT="${RESUME_MASTER_PORT:-29534}"
    EXPECTED_BASE_CHECKPOINT_SHA256="368a99ca9575a78d01f4cdcdee8820ec74d30c4528cf7aff07b83361a17cbbda"
    EXPECTED_DATA_MANIFEST_SHA256="1190b75b1ef19a7abd949bdff5679da59afa7e51a043eeb43663cf2c4495173c"
    LOCKED_TASK_ENV=(
      "FASTWAM_ROBOTWIN_STAGE3_BASE_CHECKPOINT=/root/feihong/FastWAM/formal_runs/FAST_WAM_github/robotwin_unified_shared_3cam_384_1e-4/2026-07-01_00-51-30/checkpoints/weights/latest.pt"
      "FASTWAM_ROBOTWIN_STAGE3_BASE_SHA256=${EXPECTED_BASE_CHECKPOINT_SHA256}"
      "FASTWAM_ROBOTWIN_STAGE3_VAE=${CANONICAL_VAE}"
      "FASTWAM_ROBOTWIN_STAGE3_DATA_MANIFEST=/root/feihong/FastWAM/formal_runs/contracts/stage3/robotwin_train_6011575f_27225e/robotwin_stage3_data_manifest.json"
      "FASTWAM_ROBOTWIN_STAGE3_DATA_MANIFEST_SHA256=${EXPECTED_DATA_MANIFEST_SHA256}"
      "FASTWAM_ROBOTWIN_TEXT_CACHE_INDEX_DESCRIPTOR=/root/feihong/FastWAM/formal_runs/contracts/stage3/robotwin_train_6011575f_27225e/robotwin_text_cache_index.json"
      "FASTWAM_ROBOTWIN_STATS=/root/feihong/FastWAM/datasets/robotwin2.0/dataset_stats.json"
      "FASTWAM_ROBOTWIN_STAGE3_EXPECTED_DATASET_LENGTH=6011575"
      "FASTWAM_ROBOTWIN_STAGE3_EXPECTED_DATASET_EPISODES=27225"
    )
    ;;
  *)
    fail "unsupported benchmark: ${BENCHMARK}"
    ;;
esac

[[ "${FASTWAM_DRY_RUN}" == "0" || "${FASTWAM_DRY_RUN}" == "1" ]] ||
  fail "FASTWAM_DRY_RUN must be 0 or 1"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] ||
  fail "RUN_ID may contain only letters, digits, dot, underscore, and hyphen"
[[ "${SMOKE_OUTPUT_BASE}" == /* ]] ||
  fail "SMOKE_OUTPUT_BASE must be an absolute persistent-storage path"
[[ -f "${TRAIN_LAUNCHER}" ]] || fail "missing training launcher: ${TRAIN_LAUNCHER}"
[[ -d "${FASTWAM_REPO_DIR}" ]] || fail "FASTWAM_REPO_DIR does not exist"
FASTWAM_REPO_DIR="$(cd -- "${FASTWAM_REPO_DIR}" && pwd -P)"
[[ "${FASTWAM_REPO_DIR}" == "${DEFAULT_REPO_DIR}" ]] ||
  fail "FASTWAM_REPO_DIR must resolve to the checkout containing this launcher"

validate_port() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || fail "${name} must be an integer TCP port"
  local numeric_value=$((10#${value}))
  ((numeric_value >= 1 && numeric_value <= 65535)) ||
    fail "${name} must be between 1 and 65535"
}

validate_port "MASTER_PORT" "${FRESH_MASTER_PORT}"
validate_port "RESUME_MASTER_PORT" "${RESUME_MASTER_PORT}"
[[ "${FRESH_MASTER_PORT}" != "${RESUME_MASTER_PORT}" ]] ||
  fail "fresh and resume must use different master ports"

GIT_COMMIT="$(git -C "${FASTWAM_REPO_DIR}" rev-parse HEAD)"
GIT_SHORT="${GIT_COMMIT:0:7}"
RAW_SMOKE_ROOT="${SMOKE_ROOT:-${SMOKE_OUTPUT_BASE}/${OUTPUT_PREFIX}_${GIT_SHORT}_${RUN_ID}}"
[[ "${RAW_SMOKE_ROOT}" == /* ]] || fail "SMOKE_ROOT must be an absolute path"
PERSISTENT_FASTWAM_ROOT="$(realpath -m -- "${FASTWAM_STORAGE_ROOT}/FastWAM")"
SMOKE_ROOT="$(realpath -m -- "${RAW_SMOKE_ROOT}")"
[[ "${SMOKE_ROOT}" == "${PERSISTENT_FASTWAM_ROOT}/"* ]] ||
  fail "SMOKE_ROOT must stay on the configured FastWAM persistent-storage tree"

FRESH_DIR="${SMOKE_ROOT}/uninterrupted"
RESUMED_DIR="${SMOKE_ROOT}/resume_from_step1"
RESUME_SOURCE="${FRESH_DIR}/checkpoints/states/step_000001"
FRESH_FINAL_STATE="${FRESH_DIR}/checkpoints/states/step_000002"
RESUMED_FINAL_STATE="${RESUMED_DIR}/checkpoints/states/step_000002"
FRESH_EXPORT="${FRESH_DIR}/checkpoints/exports/step_000002.pt"
RESUMED_EXPORT="${RESUMED_DIR}/checkpoints/exports/step_000002.pt"
RECEIPT="${SMOKE_ROOT}/verification_receipt.json"

SMOKE_ARGS=(
  "training.max_steps=2"
  "checkpoint.save_every=1"
  "checkpoint.keep_last=2"
  "checkpoint.save_final=true"
)

run_phase() {
  local phase="$1"
  local output_dir="$2"
  local resume_state="$3"
  local master_port="$4"
  echo "[smoke] phase=${phase} benchmark=${BENCHMARK_LABEL} output_dir=${output_dir}"
  env \
    "${LOCKED_TASK_ENV[@]}" \
    FASTWAM_REPO_DIR="${FASTWAM_REPO_DIR}" \
    FASTWAM_ENV="${FASTWAM_ENV}" \
    FASTWAM_STORAGE_ROOT="${FASTWAM_STORAGE_ROOT}" \
    FASTWAM_DRY_RUN="${FASTWAM_DRY_RUN}" \
    CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}" \
    NPROC_PER_NODE="8" \
    SENSECORE_ACCELERATE_DEVICE_COUNT="8" \
    NNODES="1" \
    NODE_RANK="0" \
    RUN_ID="${RUN_ID}_${phase}" \
    OUTPUT_DIR="${output_dir}" \
    LOG_FILE="${output_dir}/launch.log" \
    RESUME_STATE="${resume_state}" \
    MASTER_PORT="${master_port}" \
    bash "${TRAIN_LAUNCHER}" "${SMOKE_ARGS[@]}" &
  ACTIVE_CHILD_PID="$!"
  local phase_status=0
  wait "${ACTIVE_CHILD_PID}" || phase_status="$?"
  ACTIVE_CHILD_PID=""
  return "${phase_status}"
}

require_artifacts() {
  local required_path
  for required_path in "$@"; do
    [[ -e "${required_path}" ]] || fail "smoke artifact is missing: ${required_path}"
  done
}

preflight() {
  [[ -x "${FASTWAM_ENV}/bin/python" ]] ||
    fail "missing Python environment: ${FASTWAM_ENV}"
  [[ -f "${FASTWAM_REPO_DIR}/scripts/verify_stage3_resume_equivalence.py" ]] ||
    fail "missing Stage3 resume verifier"
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

cat <<EOF
[stage3-smoke]
  benchmark=${BENCHMARK_LABEL}
  workflow=fresh_step_1_2 -> resume_from_step_1 -> exact_verification
  smoke_root=${SMOKE_ROOT}
  git_commit=${GIT_COMMIT}
EOF

if [[ "${FASTWAM_DRY_RUN}" != "1" && -e "${SMOKE_ROOT}" ]]; then
  fail "SMOKE_ROOT already exists; use a new RUN_ID or SMOKE_ROOT: ${SMOKE_ROOT}"
fi
if [[ "${FASTWAM_DRY_RUN}" != "1" ]]; then
  preflight
  mkdir -p "$(dirname -- "${SMOKE_ROOT}")"
  mkdir "${SMOKE_ROOT}" ||
    fail "SMOKE_ROOT already exists; use a new RUN_ID or SMOKE_ROOT: ${SMOKE_ROOT}"
fi

run_phase "fresh" "${FRESH_DIR}" "" "${FRESH_MASTER_PORT}"
if [[ "${FASTWAM_DRY_RUN}" != "1" ]]; then
  require_artifacts \
    "${RESUME_SOURCE}/COMPLETE" \
    "${FRESH_FINAL_STATE}/COMPLETE" \
    "${FRESH_EXPORT}"
fi
run_phase "resume" "${RESUMED_DIR}" "${RESUME_SOURCE}" "${RESUME_MASTER_PORT}"

if [[ "${FASTWAM_DRY_RUN}" == "1" ]]; then
  echo "[dry-run] fresh, resume, and exact verification are fully planned; no files were written"
  exit 0
fi

require_artifacts \
  "${RESUMED_FINAL_STATE}/COMPLETE" \
  "${RESUMED_EXPORT}"

FRESH_MANIFEST="${FRESH_FINAL_STATE}/manifest.json"
TRAINING_CONTRACT_SHA256="$(
  "${FASTWAM_ENV}/bin/python" -c \
    'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["training_contract_sha256"])' \
    "${FRESH_MANIFEST}"
)"
[[ "${TRAINING_CONTRACT_SHA256}" =~ ^[0-9a-f]{64}$ ]] ||
  fail "fresh manifest has an invalid training_contract_sha256"

export CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}"
export PYTHONPATH="${FASTWAM_REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
VERIFY_COMMAND=(
  "${FASTWAM_ENV}/bin/python"
  "${FASTWAM_REPO_DIR}/scripts/verify_stage3_resume_equivalence.py"
  "${FRESH_FINAL_STATE}"
  "${RESUME_SOURCE}"
  "${RESUMED_FINAL_STATE}"
  "${FRESH_EXPORT}"
  "${RESUMED_EXPORT}"
  --expected-final-step 2
  --expected-resume-step 1
  --expected-world-size 8
  --expected-zero-stage 2
  --expected-batch-size-per-rank 2
  --expected-gradient-accumulation-steps 3
  --expected-base-checkpoint-sha256 "${EXPECTED_BASE_CHECKPOINT_SHA256}"
  --expected-data-manifest-sha256 "${EXPECTED_DATA_MANIFEST_SHA256}"
  --expected-training-contract-sha256 "${TRAINING_CONTRACT_SHA256}"
  --expected-git-commit "${GIT_COMMIT}"
)

echo "[smoke] phase=verify benchmark=${BENCHMARK_LABEL}"
print_command "${VERIFY_COMMAND[@]}"
PARTIAL_RECEIPT="${RECEIPT}.partial.$$"
"${VERIFY_COMMAND[@]}" | tee "${PARTIAL_RECEIPT}"
"${FASTWAM_ENV}/bin/python" -c \
  'import json, sys; payload = json.load(open(sys.argv[1], encoding="utf-8")); assert payload["status"] == "ok"' \
  "${PARTIAL_RECEIPT}"
mv "${PARTIAL_RECEIPT}" "${RECEIPT}"
echo "[ok] ${BENCHMARK_LABEL} Stage3 8xH100 smoke passed"
echo "[ok] receipt=${RECEIPT}"
PARTIAL_SUCCESS="${SMOKE_ROOT}/SUCCESS.partial.$$"
printf "benchmark=%s\ngit_commit=%s\nreceipt=%s\n" \
  "${BENCHMARK_LABEL}" "${GIT_COMMIT}" "${RECEIPT}" > "${PARTIAL_SUCCESS}"
# Treat the final rename as the workflow commit point. Avoid a pending signal
# producing both a durable SUCCESS marker and a failed scheduler status.
trap '' TERM INT
mv "${PARTIAL_SUCCESS}" "${SMOKE_ROOT}/SUCCESS"
