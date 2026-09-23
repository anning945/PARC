#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
"$PYTHON" "$ROOT/src/parc_router/parc_runtime.py" --self-test
