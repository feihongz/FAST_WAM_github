#!/usr/bin/env bash

# Resolve a FastWAM environment suitable for LeRobot-format training.
# A caller-provided FASTWAM_ENV is authoritative; automatic fallback is only
# used when FASTWAM_ENV is unset.
resolve_fastwam_train_env() {
  local requested_env="${FASTWAM_ENV:-}"
  local candidates=()
  local candidate
  local checked=()

  if [[ -n "${requested_env}" ]]; then
    candidates=("${requested_env}")
  else
    candidates=(
      "/root/.venvs/fastwam"
    )
  fi

  for candidate in "${candidates[@]}"; do
    checked+=("${candidate}")
    [[ -x "${candidate}/bin/python" ]] || continue
    [[ -x "${candidate}/bin/accelerate" ]] || continue
    if ! "${candidate}/bin/python" - <<'PY_CHECK_FASTWAM_ENV' >/dev/null 2>&1
import importlib.util

required = [
    "torch",
    "fastwam",
    "hydra",
    "accelerate",
    "deepspeed",
    "wandb",
    "av",
    "datasets",
    "torchvision",
]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("missing modules: " + ", ".join(missing))
PY_CHECK_FASTWAM_ENV
    then
      continue
    fi

    FASTWAM_ENV="${candidate}"
    export FASTWAM_ENV
    echo "[env] selected FASTWAM_ENV=${FASTWAM_ENV}"
    return 0
  done

  if [[ -n "${requested_env}" ]]; then
    echo "[error] Explicit FASTWAM_ENV is not a usable FastWAM training environment: ${requested_env}" >&2
  else
    echo "[error] No usable FastWAM training environment was found." >&2
  fi
  echo "[error] Checked candidates:" >&2
  printf '  - %s\n' "${checked[@]}" >&2
  echo "[error] A valid environment needs bin/python, bin/accelerate, and all FastWAM training dependencies." >&2
  return 1
}
