#!/usr/bin/env bash
set -euo pipefail

# Reviewed formal profile: fresh 20-epoch ceiling with patience-based early stop.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export FASTWAM_GATE_PROFILE=formal
exec bash "${SCRIPT_DIR}/run_libero_stage2_gate_smoke_1xh100.sh" "$@"
