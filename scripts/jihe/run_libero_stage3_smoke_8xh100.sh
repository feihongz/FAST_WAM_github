#!/usr/bin/env bash
set -euo pipefail

[[ "$#" == "0" ]] || {
  echo "[error] this one-click launcher takes no arguments; use environment variables for optional overrides" >&2
  exit 1
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "${SCRIPT_DIR}/_run_stage3_smoke_8xh100.sh" libero
