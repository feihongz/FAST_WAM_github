#!/usr/bin/env bash
set -euo pipefail

[[ "$#" == "1" ]] || {
  echo "[error] usage: $0 {libero|robotwin}" >&2
  exit 2
}

BENCHMARK_KEY="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
FASTWAM_STORAGE_ROOT="${FASTWAM_STORAGE_ROOT:-/root/feihong}"
export FASTWAM_REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
export FASTWAM_ENV="/root/.venvs/fastwam"

case "${BENCHMARK_KEY}" in
  libero)
    BENCHMARK="LIBERO"
    TASK_NAME="libero_stage3_alignment_2cam224_1e-4"
    TRAIN_LAUNCHER="${SCRIPT_DIR}/train_libero_stage3_alignment_8xh100.sh"
    MAX_STEPS="30000"
    STEPS_PER_EPOCH="5697"
    DATASET_EXPOSURE="5.266 epochs / 1,440,000 windows"
    SAVE_EVERY="1000"
    KEEP_LAST="31"
    EXPECTED_TRAIN_TIME="19-23 hours"

    export FASTWAM_STAGE3_BASE_CHECKPOINT="/root/feihong/FastWAM/formal_runs/FAST_WAM_github/libero_unified_shared_2cam224_1e-4/2026-07-01_00-44-20/checkpoints/weights/latest.pt"
    export FASTWAM_STAGE3_BASE_SHA256="17a5588cc2b8d162219c9daf818614f614ee4a7921933a4a26c5d678111330e9"
    export FASTWAM_STAGE3_VAE="/root/feihong/FastWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
    export FASTWAM_STAGE3_DATA_MANIFEST="/root/feihong/FastWAM/formal_runs/contracts/stage3/libero_current_273465f_1693e/libero_stage3_data_manifest.json"
    export FASTWAM_STAGE3_DATA_MANIFEST_SHA256="08da49109a57b55c67f3fa4ac31fbfa44e44dd541a194a5d3420838537d0d320"
    export FASTWAM_LIBERO_STATS="/root/feihong/FastWAM/formal_runs/FAST_WAM_github/libero_unified_shared_2cam224_1e-4/2026-07-01_00-44-20/dataset_stats.json"
    export FASTWAM_STAGE3_EXPECTED_DATASET_LENGTH="273465"
    export FASTWAM_STAGE3_EXPECTED_DATASET_EPISODES="1693"
    ;;
  robotwin)
    BENCHMARK="RoboTwin-2.0"
    TASK_NAME="robotwin_stage3_alignment_3cam384_1e-4"
    TRAIN_LAUNCHER="${SCRIPT_DIR}/train_robotwin_stage3_alignment_8xh100.sh"
    MAX_STEPS="40000"
    STEPS_PER_EPOCH="125241"
    DATASET_EXPOSURE="0.3194 epoch / 1,920,000 windows"
    SAVE_EVERY="500"
    KEEP_LAST="41"
    EXPECTED_TRAIN_TIME="168-192 hours"

    export FASTWAM_ROBOTWIN_STAGE3_BASE_CHECKPOINT="/root/feihong/FastWAM/formal_runs/FAST_WAM_github/robotwin_unified_shared_3cam_384_1e-4/2026-07-01_00-51-30/checkpoints/weights/latest.pt"
    export FASTWAM_ROBOTWIN_STAGE3_BASE_SHA256="368a99ca9575a78d01f4cdcdee8820ec74d30c4528cf7aff07b83361a17cbbda"
    export FASTWAM_ROBOTWIN_STAGE3_VAE="/root/feihong/FastWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
    export FASTWAM_ROBOTWIN_STAGE3_DATA_MANIFEST="/root/feihong/FastWAM/formal_runs/contracts/stage3/robotwin_train_6011575f_27225e/robotwin_stage3_data_manifest.json"
    export FASTWAM_ROBOTWIN_STAGE3_DATA_MANIFEST_SHA256="1190b75b1ef19a7abd949bdff5679da59afa7e51a043eeb43663cf2c4495173c"
    export FASTWAM_ROBOTWIN_TEXT_CACHE_INDEX_DESCRIPTOR="/root/feihong/FastWAM/formal_runs/contracts/stage3/robotwin_train_6011575f_27225e/robotwin_text_cache_index.json"
    export FASTWAM_ROBOTWIN_STATS="/root/feihong/FastWAM/datasets/robotwin2.0/dataset_stats.json"
    export FASTWAM_ROBOTWIN_STAGE3_EXPECTED_DATASET_LENGTH="6011575"
    export FASTWAM_ROBOTWIN_STAGE3_EXPECTED_DATASET_EPISODES="27225"
    ;;
  *)
    echo "[error] unknown benchmark: ${BENCHMARK_KEY}" >&2
    exit 2
    ;;
esac

RUN_ID="${RUN_ID:-$(date -u +%Y-%m-%d_%H-%M-%S)}"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "[error] RUN_ID may contain only letters, digits, '.', '_' and '-'" >&2
  exit 2
}

# Each attempt gets a new output directory. A resumed attempt points at the
# previous attempt through RESUME_STATE, while retaining the exact same formal
# training contract below.
export RUN_ID
export FASTWAM_OUTPUT_BASE="${FASTWAM_STORAGE_ROOT}/FastWAM/formal_runs/stage3/full"
export OUTPUT_DIR="${FASTWAM_OUTPUT_BASE}/${TASK_NAME}/${RUN_ID}"
export LOG_FILE="${OUTPUT_DIR}/launch.log"

if [[ -n "${RESUME_STATE:-}" && "${RESUME_STATE}" == *"/pilots/"* ]]; then
  echo "[error] a 200-step pilot state cannot resume a formal Stage 3 run" >&2
  exit 2
fi

cat <<EOF
[stage3-full]
  benchmark=${BENCHMARK}
  task=${TASK_NAME}
  max_steps=${MAX_STEPS}
  steps_per_epoch=${STEPS_PER_EPOCH}
  data_exposure=${DATASET_EXPOSURE}
  global_batch=48
  optimizer=AdamW(lr=1e-4,betas=0.9/0.95,weight_decay=1e-4)
  schedule=5%_warmup_then_cosine_to_1e-6
  checkpoint_every=${SAVE_EVERY}
  checkpoint_keep_last=${KEEP_LAST}
  expected_wall_time=${EXPECTED_TRAIN_TIME}
  output_dir=${OUTPUT_DIR}
  resume_state=${RESUME_STATE:-null}
  pilot_resume_allowed=false
  wandb=disabled_local_launch_log_is_authoritative
EOF

exec "${TRAIN_LAUNCHER}" \
  "training.max_steps=${MAX_STEPS}" \
  "training.num_epochs=10" \
  "training.learning_rate=1.0e-4" \
  "training.weight_decay=1.0e-4" \
  "training.betas=[0.9,0.95]" \
  "training.max_grad_norm=1.0" \
  "training.lr_scheduler_type=cosine" \
  "training.warmup_ratio=0.05" \
  "training.seed=42" \
  "stage3.num_solver_steps=10" \
  "stage3.sigma_shift=null" \
  "stage3.k_sampling=uniform" \
  "stage3.helpful_relative_margin=0.05" \
  "stage3.lambda_action=1.0" \
  "stage3.lambda_align=1.0" \
  "stage3.lambda_safe=0.5" \
  "checkpoint.save_every=${SAVE_EVERY}" \
  "checkpoint.keep_last=${KEEP_LAST}" \
  "checkpoint.save_final=true" \
  "runtime.log_every=100"
