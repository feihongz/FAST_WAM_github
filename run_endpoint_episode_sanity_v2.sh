#!/usr/bin/env bash
set -euo pipefail

cd /root/feihong/FAST_WAM_github

export DIFFSYNTH_MODEL_BASE_PATH=/root/public/models
export DIFFSYNTH_SKIP_DOWNLOAD=true
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export LIBERO_CONFIG_PATH=/root/feihong/FastWAM/libero_config_fastwam_local
export PYTHONPATH=/root/feihong/LIBERO:/root/feihong/FAST_WAM_github:/root/feihong/FAST_WAM_github/src
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

PYTHON_BIN=/root/.venvs/fastwam/bin/python
CKPT=/root/feihong/FastWAM/formal_runs/FAST_WAM_github/libero_unified_shared_2cam224_1e-4/2026-07-01_00-44-20/checkpoints/weights/latest.pt
STATS=/root/feihong/FastWAM/formal_runs/FAST_WAM_github/libero_unified_shared_2cam224_1e-4/2026-07-01_00-44-20/dataset_stats.json
BASE_OUT=/root/feihong/FastWAM/evaluate_results/libero_endpoint_sanity_T10/endpoint_sanity_20260715_custom_endpoints_v2
SUITE=libero_10
TASK_ID=0
TRIALS=50
mkdir -p "${BASE_OUT}"

run_case() {
  local name="$1"
  local mode="$2"
  local prefix_steps="$3"
  local force_custom="$4"
  local out="${BASE_OUT}/${name}"
  local log="${BASE_OUT}/${name}.log"
  mkdir -p "${out}"
  if ls "${out}/${SUITE}"/gpu*_task"${TASK_ID}"_results.json >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] Skip existing ${name}"
    return 0
  fi
  echo "[$(date '+%F %T')] Start ${name}: mode=${mode} prefix=${prefix_steps} force=${force_custom}"
  local args=(
    experiments/libero/eval_libero_single.py
    task=libero_unified_shared_2cam224_1e-4
    ckpt="${CKPT}"
    EVALUATION.task_suite_name="${SUITE}"
    EVALUATION.task_id="${TASK_ID}"
    gpu_id=0
    EVALUATION.num_trials="${TRIALS}"
    EVALUATION.output_dir="${out}"
    model.redirect_common_files=false
    EVALUATION.dataset_stats_path="${STATS}"
    EVALUATION.inference_mode="${mode}"
    EVALUATION.num_inference_steps=10
    EVALUATION.save_videos=false
    EVALUATION.retry_invalid_episodes=true
    EVALUATION.max_invalid_episode_retries=100
    EVALUATION.black_screen_filter=true
  )
  if [ "${mode}" = "prefix" ]; then
    args+=(EVALUATION.video_prefix_steps="${prefix_steps}")
    args+=(EVALUATION.force_custom_prefix="${force_custom}")
  fi
  "${PYTHON_BIN}" "${args[@]}" > "${log}" 2>&1
  "${PYTHON_BIN}" experiments/libero/summarize_results.py --output_dir="${out}" >> "${log}" 2>&1 || true
  echo "[$(date '+%F %T')] Done ${name}"
}

run_pair() {
  local left_name="$1" left_mode="$2" left_prefix="$3" left_force="$4"
  local right_name="$5" right_mode="$6" right_prefix="$7" right_force="$8"
  run_case "${left_name}" "${left_mode}" "${left_prefix}" "${left_force}" &
  local left_pid=$!
  run_case "${right_name}" "${right_mode}" "${right_prefix}" "${right_force}" &
  local right_pid=$!
  local rc=0
  wait "${left_pid}" || rc=1
  wait "${right_pid}" || rc=1
  return "${rc}"
}

run_pair wo_orig wo 0 false n0_custom prefix 0 true
run_pair w_orig w 10 false n10_custom prefix 10 true

"${PYTHON_BIN}" - <<'PY2'
from pathlib import Path
import json
base=Path('/root/feihong/FastWAM/evaluate_results/libero_endpoint_sanity_T10/endpoint_sanity_20260715_custom_endpoints_v2')
for name in ['wo_orig','n0_custom','w_orig','n10_custom']:
    files=list((base/name).rglob('gpu*_task*_results.json'))
    if not files:
        print(name, 'missing')
        continue
    data=json.loads(files[0].read_text())
    s=data.get('successes',0); e=data.get('total_episodes',0)
    print(name, f'{s}/{e}', f'{100*s/e:.2f}%' if e else '-')
PY2
