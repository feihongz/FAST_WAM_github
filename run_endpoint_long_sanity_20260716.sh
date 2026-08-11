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
export PYTHON_BIN=/root/.venvs/fastwam/bin/python
export USE_TMUX=0
export MONITORING_INTERVAL=30
export STATUS_INTERVAL=120

PYTHON_BIN=/root/.venvs/fastwam/bin/python
CKPT=/root/feihong/FastWAM/formal_runs/FAST_WAM_github/libero_unified_shared_2cam224_1e-4/2026-07-01_00-44-20/checkpoints/weights/latest.pt
STATS=/root/feihong/FastWAM/formal_runs/FAST_WAM_github/libero_unified_shared_2cam224_1e-4/2026-07-01_00-44-20/dataset_stats.json
BASE_OUT=/root/feihong/FastWAM/evaluate_results/libero_endpoint_sanity_T10/endpoint_sanity_long_only_20260716
mkdir -p "${BASE_OUT}"

run_case() {
  local name="$1"
  local mode="$2"
  local prefix_steps="$3"
  local out="${BASE_OUT}/${name}"
  local log="${BASE_OUT}/${name}.manager.log"
  mkdir -p "${out}"
  if [ -f "${out}/summary.json" ]; then
    echo "[$(date '+%F %T')] Skip ${name}; summary exists"
    return 0
  fi

  echo "[$(date '+%F %T')] Start ${name}: mode=${mode} prefix=${prefix_steps} suite=libero_10"
  local args=(
    experiments/libero/run_libero_manager.py
    task=libero_unified_shared_2cam224_1e-4
    ckpt="${CKPT}"
    model.redirect_common_files=false
    EVALUATION.dataset_stats_path="${STATS}"
    EVALUATION.output_dir="${out}"
    EVALUATION.inference_mode="${mode}"
    EVALUATION.num_inference_steps=10
    EVALUATION.num_trials=50
    EVALUATION.save_videos=false
    EVALUATION.retry_invalid_episodes=true
    EVALUATION.max_invalid_episode_retries=100
    EVALUATION.black_screen_filter=true
    MULTIRUN.task_suite_names=[libero_10]
    MULTIRUN.num_gpus=1
    MULTIRUN.max_tasks_per_gpu=1
  )
  if [ "${mode}" = "prefix" ]; then
    args+=(EVALUATION.video_prefix_steps="${prefix_steps}")
    args+=(EVALUATION.force_custom_prefix=true)
  fi

  "${PYTHON_BIN}" "${args[@]}" > "${log}" 2>&1
  echo "[$(date '+%F %T')] Done ${name}"
}

run_pair() {
  local left_name="$1" left_mode="$2" left_prefix="$3"
  local right_name="$4" right_mode="$5" right_prefix="$6"
  run_case "${left_name}" "${left_mode}" "${left_prefix}" &
  local left_pid=$!
  run_case "${right_name}" "${right_mode}" "${right_prefix}" &
  local right_pid=$!
  local rc=0
  wait "${left_pid}" || rc=1
  wait "${right_pid}" || rc=1
  return "${rc}"
}

run_pair wo_orig wo 0 n0_custom prefix 0
run_pair w_orig w 10 n10_custom prefix 10

"${PYTHON_BIN}" - <<'PY2'
from pathlib import Path
import json
base=Path('/root/feihong/FastWAM/evaluate_results/libero_endpoint_sanity_T10/endpoint_sanity_long_only_20260716')
for name in ['wo_orig','n0_custom','w_orig','n10_custom']:
    summary=base/name/'summary.json'
    if not summary.exists():
        print(name, 'missing')
        continue
    data=json.loads(summary.read_text())
    suites=data.get('suite_stats', {})
    s=suites.get('libero_10', {})
    succ=s.get('total_successes', 0)
    trials=s.get('total_trials', 0)
    avg=data.get('overall', {}).get('average_success_rate')
    print(name, f'libero_10={succ}/{trials}', f'{100*succ/trials:.2f}%' if trials else '-', 'avg=', avg)
PY2
