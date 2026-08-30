#!/usr/bin/env bash
set -euo pipefail

[[ "$#" == "0" ]] || {
  echo "[error] this formal launcher takes no arguments" >&2
  exit 2
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "${SCRIPT_DIR}/_run_stage3_full_8xh100.sh" libero
