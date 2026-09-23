#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
export PYTHONHASHSEED=0
cd "$ROOT"

"$PYTHON" scripts/parc_release_checksums.py check
PYTHON="$PYTHON" bash scripts/run_parc_self_test.sh
"$PYTHON" -m unittest discover -s tests -v
PYTHON="$PYTHON" bash scripts/run_parc_example.sh "${1:-$ROOT/outputs/synthetic_routing_output.json}"
echo "PASS PARC release verification"
