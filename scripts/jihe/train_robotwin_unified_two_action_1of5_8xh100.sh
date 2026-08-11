#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_VARIANT=unified_two_action exec bash "${SCRIPT_DIR}/train_robotwin_1of5_8xh100.sh" "$@"
