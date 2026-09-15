#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
"$PYTHON" "$ROOT/src/parc_router/successor_selector_v9_dualview_runtime_20260720.py" --self-test
