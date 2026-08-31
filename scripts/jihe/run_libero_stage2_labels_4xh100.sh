#!/usr/bin/env bash
set -euo pipefail

# Compatibility entrypoint retained so an already copied one-line command
# cannot silently launch the obsolete four-rank topology.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
echo "[compat] forwarding the retired 4xH100 command to the formal 8xH100 launcher" >&2
exec bash "${SCRIPT_DIR}/run_libero_stage2_labels_8xh100.sh" "$@"
