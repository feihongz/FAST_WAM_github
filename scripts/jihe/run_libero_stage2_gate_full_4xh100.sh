#!/usr/bin/env bash
set -euo pipefail

# Formal profile: 20-epoch ceiling with patience-based early stopping.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export FASTWAM_GATE_PROFILE=formal
exec bash "${SCRIPT_DIR}/_run_libero_stage2_gate_4xh100.sh" "$@"
