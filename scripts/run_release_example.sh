#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
MODEL="$ROOT/artifacts/parc_selector/parc_frozen_selector.joblib"
INPUT="$ROOT/examples/synthetic_task_input.json"
OUTPUT="${1:-$ROOT/examples/synthetic_routing_output.json}"

"$PYTHON" - <<'PY'
import sys

expected = {
    "numpy": "2.2.6",
    "joblib": "1.5.3",
    "sklearn": "1.7.2",
}
for name, version in expected.items():
    module = __import__(name)
    observed = module.__version__
    if observed != version:
        raise SystemExit(
            f"{name}=={version} is required; found {observed}. "
            "Create the documented Python 3.10 environment first."
        )
print("Dependency versions: OK")
PY

"$PYTHON" "$ROOT/src/parc_router/parc_runtime.py" \
  --model "$MODEL" \
  --task-input "$INPUT" \
  --output "$OUTPUT"

echo "Wrote routing-only example: $OUTPUT"
