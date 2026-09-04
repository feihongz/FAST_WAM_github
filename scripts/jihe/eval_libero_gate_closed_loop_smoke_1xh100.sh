#!/usr/bin/env bash
set -euo pipefail

# One-H100 closed-loop connectivity smoke for the final LIBERO Stage 3
# Adapter + selected Stage 2 Gate. This is not a formal success-rate result.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
FASTWAM_REPO_DIR="${FASTWAM_REPO_DIR:-${DEFAULT_REPO_DIR}}"
FASTWAM_ENV="${FASTWAM_ENV:-/root/.venvs/fastwam}"
PYTHON_BIN="${PYTHON_BIN:-${FASTWAM_ENV}/bin/python}"
RUN_ID="${RUN_ID:-$(date -u +%Y-%m-%d_%H-%M-%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/feihong/FastWAM/evaluate_results/stage2_gate_closed_loop_smoke/libero/${RUN_ID}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/launch.log}"
FASTWAM_DRY_RUN="${FASTWAM_DRY_RUN:-0}"

BASE_CKPT="${BASE_CKPT:-/root/feihong/FastWAM/formal_runs/FAST_WAM_github/libero_unified_shared_2cam224_1e-4/2026-07-01_00-44-20/checkpoints/weights/latest.pt}"
BASE_SHA256="${BASE_SHA256:-17a5588cc2b8d162219c9daf818614f614ee4a7921933a4a26c5d678111330e9}"
ADAPTER_PATH="${ADAPTER_PATH:-/root/feihong/FastWAM/formal_runs/stage3/full/libero_stage3_alignment_2cam224_1e-4/2026-08-30_10-29-08/checkpoints/exports/step_030000.pt}"
ADAPTER_SHA256="${ADAPTER_SHA256:-cbc593bc6ce99c0249a65e5c7cef754c9a1d7ea602f81fdae2b8cb158a25858c}"
DATA_SHA256="${DATA_SHA256:-08da49109a57b55c67f3fa4ac31fbfa44e44dd541a194a5d3420838537d0d320}"
CONTRACT_SHA256="${CONTRACT_SHA256:-84ee86f32912ca96fa058b02ce7997362b8350e73f4e0f4377bc8728af3e6d98}"
GLOBAL_STEP="${GLOBAL_STEP:-30000}"

GATE_CHECKPOINT="${GATE_CHECKPOINT:-/root/feihong/FastWAM/formal_runs/stage2/gate/libero_stage2_gate_2cam224_20ep/22a8d65_2026-09-02_03-29-13/gate_run/gate_best.pt}"
GATE_SHA256="${GATE_SHA256:-67db6f46abe67f5c6a4417b60864f0ad0535edf8f911d9e4d11eaed137b9b722}"
GATE_LABEL_MANIFEST_SHA256="${GATE_LABEL_MANIFEST_SHA256:-d6dc98a6a36c30150db30000c86d07c7a1e7d90b1dc5d1a5a60e02126c22b3e0}"
GATE_EPISODE_SPLIT_SHA256="${GATE_EPISODE_SPLIT_SHA256:-a77efa24249dab8cfacbc228b1da341947240b36fa77d90182701c07bdcf7787}"
GATE_TRAINING_CONFIG_SHA256="${GATE_TRAINING_CONFIG_SHA256:-cf2faedeca8fb69c2dfa1ace6178c797a28bf9b0211b2d98273b7cf0e9567060}"
GATE_GIT_COMMIT="${GATE_GIT_COMMIT:-22a8d659edffae07bba05dfb6ce0957af312faa7}"
GATE_THRESHOLD="${GATE_THRESHOLD:-0.5}"

STATS_PATH="${STATS_PATH:-/root/feihong/FastWAM/formal_runs/FAST_WAM_github/libero_unified_shared_2cam224_1e-4/2026-07-01_00-44-20/dataset_stats.json}"
VAE_PATH="${VAE_PATH:-/root/feihong/FastWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
FASTWAM_LIBERO_ROOT="${FASTWAM_LIBERO_ROOT:-/root/feihong/FastWAM/third_party/LIBERO}"
FASTWAM_LIBERO_DATASETS="${FASTWAM_LIBERO_DATASETS:-/root/feihong/FastWAM/datasets/libero_mujoco3.3.2}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${OUTPUT_DIR}/libero_config}"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
TASK_ID="${TASK_ID:-0}"
NUM_TRIALS="${NUM_TRIALS:-1}"
REPLAN_STEPS="${REPLAN_STEPS:-32}"
TIMING_WARMUP_QUERIES="${TIMING_WARMUP_QUERIES:-3}"
NUM_INFERENCE_STEPS="10"

fail() {
  echo "[error] $*" >&2
  exit 1
}

print_command() {
  printf "[launch]"
  printf " %q" "$@"
  printf "\n"
}

[[ "${OUTPUT_DIR}" == /* ]] || fail "OUTPUT_DIR must be absolute"
[[ "${NUM_TRIALS}" =~ ^[1-9][0-9]*$ ]] || fail "NUM_TRIALS must be positive"
[[ "${TIMING_WARMUP_QUERIES}" =~ ^[0-9]+$ ]] || fail "TIMING_WARMUP_QUERIES must be non-negative"
[[ "${GATE_THRESHOLD}" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]] || fail "GATE_THRESHOLD must be in [0,1]"
for override in "$@"; do
  case "${override}" in
    task=*|ckpt=*|gpu_id=*|model.load_text_encoder=*|EVALUATION.output_dir=*|EVALUATION.routing_mode=*|EVALUATION.use_manifest_text_cache=*|EVALUATION.inference_mode=*|EVALUATION.gate_*=*|EVALUATION.stage3_*=*|EVALUATION.num_inference_steps=*|EVALUATION.visualize_future_video=*)
      fail "Locked Gate smoke setting cannot be overridden: ${override}"
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATH="${FASTWAM_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${FASTWAM_LIBERO_ROOT}:${FASTWAM_REPO_DIR}:${FASTWAM_REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/root/feihong/FastWAM/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD="true"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export FASTWAM_STAGE3_VAE="${VAE_PATH}"
export FASTWAM_LIBERO_ROOT
export LIBERO_CONFIG_PATH

COMMAND=(
  "${PYTHON_BIN}" experiments/libero/eval_libero_single.py
  "task=libero_stage3_alignment_2cam224_1e-4"
  "ckpt=${BASE_CKPT}"
  "gpu_id=0"
  "model.load_text_encoder=false"
  "EVALUATION.task_suite_name=${TASK_SUITE}"
  "EVALUATION.task_id=${TASK_ID}"
  "EVALUATION.num_trials=${NUM_TRIALS}"
  "EVALUATION.output_dir=${OUTPUT_DIR}"
  "EVALUATION.dataset_stats_path=${STATS_PATH}"
  "EVALUATION.stage3_adapter_path=${ADAPTER_PATH}"
  "EVALUATION.stage3_adapter_sha256=${ADAPTER_SHA256}"
  "EVALUATION.stage3_base_sha256=${BASE_SHA256}"
  "EVALUATION.stage3_data_manifest_sha256=${DATA_SHA256}"
  "EVALUATION.stage3_training_contract_sha256=${CONTRACT_SHA256}"
  "EVALUATION.stage3_global_step=${GLOBAL_STEP}"
  "EVALUATION.routing_mode=gate"
  "EVALUATION.use_manifest_text_cache=true"
  "EVALUATION.gate_checkpoint=${GATE_CHECKPOINT}"
  "EVALUATION.gate_checkpoint_sha256=${GATE_SHA256}"
  "EVALUATION.gate_threshold=${GATE_THRESHOLD}"
  "EVALUATION.gate_expected_label_manifest_sha256=${GATE_LABEL_MANIFEST_SHA256}"
  "EVALUATION.gate_expected_episode_split_assignment_sha256=${GATE_EPISODE_SPLIT_SHA256}"
  "EVALUATION.gate_expected_training_config_sha256=${GATE_TRAINING_CONFIG_SHA256}"
  "EVALUATION.gate_expected_git_commit=${GATE_GIT_COMMIT}"
  "EVALUATION.gate_expected_git_tracked_dirty=false"
  "EVALUATION.gate_expected_git_untracked_source_files=[]"
  "EVALUATION.replan_steps=${REPLAN_STEPS}"
  "EVALUATION.num_inference_steps=${NUM_INFERENCE_STEPS}"
  "EVALUATION.visualize_future_video=false"
  "EVALUATION.timing_enabled=true"
  "EVALUATION.timing_warmup_queries=${TIMING_WARMUP_QUERIES}"
  "EVALUATION.save_query_metrics=true"
  "EVALUATION.save_videos=false"
  "$@"
)

cat <<EOF
[libero-gate-closed-loop-smoke]
  benchmark=LIBERO
  gpu_count=1
  route=gate threshold=${GATE_THRESHOLD} video_nfe=0_or_${NUM_INFERENCE_STEPS}
  task=${TASK_SUITE}/${TASK_ID} trials=${NUM_TRIALS}
  adapter_step=${GLOBAL_STEP}
  output_dir=${OUTPUT_DIR}
  note=connectivity-only; not a formal success-rate result
EOF
print_command "${COMMAND[@]}"

if [[ "${FASTWAM_DRY_RUN}" == "1" ]]; then
  exit 0
fi

cd "${FASTWAM_REPO_DIR}"
[[ -x "${PYTHON_BIN}" ]] || fail "Missing Python: ${PYTHON_BIN}"
[[ -f "${BASE_CKPT}" ]] || fail "Missing base checkpoint: ${BASE_CKPT}"
[[ -f "${ADAPTER_PATH}" ]] || fail "Missing Adapter: ${ADAPTER_PATH}"
[[ -f "${GATE_CHECKPOINT}" ]] || fail "Missing Gate checkpoint: ${GATE_CHECKPOINT}"
[[ -f "${STATS_PATH}" ]] || fail "Missing normalization stats: ${STATS_PATH}"
[[ -f "${VAE_PATH}" ]] || fail "Missing VAE: ${VAE_PATH}"
[[ -d "${FASTWAM_LIBERO_ROOT}/libero/libero" ]] || fail "Missing LIBERO root: ${FASTWAM_LIBERO_ROOT}"
[[ -d "${FASTWAM_LIBERO_DATASETS}" ]] || fail "Missing LIBERO datasets: ${FASTWAM_LIBERO_DATASETS}"
[[ ! -e "${OUTPUT_DIR}" ]] || fail "OUTPUT_DIR already exists: ${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}" "${LIBERO_CONFIG_PATH}"
benchmark_root="${FASTWAM_LIBERO_ROOT}/libero/libero"
{
  echo "assets: ${benchmark_root}/assets"
  echo "bddl_files: ${benchmark_root}/bddl_files"
  echo "benchmark_root: ${benchmark_root}"
  echo "datasets: ${FASTWAM_LIBERO_DATASETS}"
  echo "init_states: ${benchmark_root}/init_files"
} > "${LIBERO_CONFIG_PATH}/config.yaml"

exec > >(tee -a "${LOG_FILE}") 2>&1
"${PYTHON_BIN}" - <<'PY'
import torch

names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print(f"[gpu] visible={names}")
if len(names) != 1 or "H100" not in names[0].upper():
    raise SystemExit(f"expected exactly one visible H100, found {names}")
PY
echo "[preflight] evaluator will verify the complete base/Adapter/Gate/data identity chain"
print_command "${COMMAND[@]}"
exec "${COMMAND[@]}"
