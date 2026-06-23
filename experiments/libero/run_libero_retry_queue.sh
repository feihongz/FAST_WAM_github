#!/usr/bin/env bash
set -euo pipefail

TASK_FILE="${1:?Usage: bash experiments/libero/run_libero_retry_queue.sh <task_file>}"

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must be set}"
CONFIG="${CONFIG:?CONFIG must be set}"
CKPT="${CKPT:?CKPT must be set}"
NUM_TRIALS="${NUM_TRIALS:-50}"
MAX_RETRIES="${MAX_RETRIES:-20}"
RETRY_SLEEP="${RETRY_SLEEP:-20}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
else
    NUM_GPUS="${NUM_GPUS:?Set CUDA_VISIBLE_DEVICES or NUM_GPUS}"
    mapfile -t GPU_ARRAY < <(seq 0 "$((NUM_GPUS - 1))")
fi

mkdir -p "$OUTPUT_DIR"/{task_logs,task_status,failed_attempts}
cp "$TASK_FILE" "$OUTPUT_DIR/tasks.txt"

echo "Retry queue output: $OUTPUT_DIR"
echo "Config: $CONFIG"
echo "Checkpoint: $CKPT"
echo "Trials per task: $NUM_TRIALS"
echo "GPUs: ${GPU_ARRAY[*]}"
echo "Max retries per task: $MAX_RETRIES"

sanitize_task_name() {
    local suite="$1"
    local task_id="$2"
    echo "${suite}_task${task_id}"
}

run_one_task() {
    local suite="$1"
    local task_id="$2"
    local gpu_id="$3"
    local task_name
    task_name="$(sanitize_task_name "$suite" "$task_id")"
    local final_result="$OUTPUT_DIR/$suite/gpu${gpu_id}_task${task_id}_results.json"
    local status_file="$OUTPUT_DIR/task_status/${task_name}.status"
    local attempt=1

    if [[ -s "$final_result" ]]; then
        echo "[$(date '+%F %T')] SKIP existing result: $suite task_id=$task_id"
        echo "SUCCESS|$gpu_id|0|$(date +%s)|existing" > "$status_file"
        return 0
    fi

    while (( attempt <= MAX_RETRIES )); do
        local attempt_dir="$OUTPUT_DIR/attempts/${task_name}/attempt_${attempt}"
        local log_file="$OUTPUT_DIR/task_logs/${task_name}_gpu${gpu_id}_attempt${attempt}.log"
        mkdir -p "$attempt_dir"
        echo "[$(date '+%F %T')] START $suite task_id=$task_id gpu=$gpu_id attempt=$attempt"

        set +e
        CUDA_VISIBLE_DEVICES="$gpu_id" $PYTHON_BIN experiments/libero/eval_libero_single.py \
            task="$CONFIG" ckpt="$CKPT" \
            EVALUATION.task_suite_name="$suite" EVALUATION.task_id="$task_id" gpu_id="$gpu_id" \
            EVALUATION.num_trials="$NUM_TRIALS" EVALUATION.output_dir="$attempt_dir" \
            $EXTRA_ARGS > "$log_file" 2>&1
        local rc=$?
        set -e

        local attempt_result="$attempt_dir/$suite/gpu${gpu_id}_task${task_id}_results.json"
        if [[ "$rc" -eq 0 && -s "$attempt_result" ]]; then
            mkdir -p "$OUTPUT_DIR/$suite"
            cp -a "$attempt_dir/$suite/." "$OUTPUT_DIR/$suite/"
            echo "SUCCESS|$gpu_id|$rc|$(date +%s)|$log_file|attempt=$attempt" > "$status_file"
            echo "[$(date '+%F %T')] SUCCESS $suite task_id=$task_id gpu=$gpu_id attempt=$attempt"
            return 0
        fi

        local failed_dir="$OUTPUT_DIR/failed_attempts/${task_name}_attempt${attempt}_rc${rc}_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$(dirname "$failed_dir")"
        mv "$attempt_dir" "$failed_dir" 2>/dev/null || true
        echo "[$(date '+%F %T')] FAILED $suite task_id=$task_id gpu=$gpu_id attempt=$attempt rc=$rc log=$log_file"
        echo "FAILED_ATTEMPT|$gpu_id|$rc|$(date +%s)|$log_file|attempt=$attempt" > "$status_file"
        attempt=$((attempt + 1))
        sleep "$RETRY_SLEEP"
    done

    echo "FAILED_FINAL|$gpu_id|1|$(date +%s)|max_retries=$MAX_RETRIES" > "$status_file"
    echo "[$(date '+%F %T')] FAILED_FINAL $suite task_id=$task_id gpu=$gpu_id"
    return 1
}

worker() {
    local gpu_idx="$1"
    local gpu_id="${GPU_ARRAY[$gpu_idx]}"
    local n_gpus="${#GPU_ARRAY[@]}"
    local line_no=0
    local failures=0

    while IFS=, read -r suite task_id; do
        [[ -z "${suite:-}" || -z "${task_id:-}" ]] && continue
        if (( line_no % n_gpus == gpu_idx )); then
            run_one_task "$suite" "$task_id" "$gpu_id" || failures=$((failures + 1))
        fi
        line_no=$((line_no + 1))
    done < "$TASK_FILE"

    return "$failures"
}

pids=()
for idx in "${!GPU_ARRAY[@]}"; do
    worker "$idx" > "$OUTPUT_DIR/task_logs/worker_gpu${GPU_ARRAY[$idx]}.log" 2>&1 &
    pids+=("$!")
done

overall_rc=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        overall_rc=1
    fi
done

exit "$overall_rc"
