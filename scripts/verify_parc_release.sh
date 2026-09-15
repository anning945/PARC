#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
cd "$ROOT"

if [[ ! -f SHA256SUMS ]]; then
  echo "Missing root SHA256SUMS" >&2
  exit 1
fi

"$PYTHON" - <<'PY'
import hashlib
from pathlib import Path

root = Path.cwd()
for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    expected, relative = line.split(maxsplit=1)
    path = root / relative
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise SystemExit(f"SHA256 mismatch: {relative}: {digest}")
print("SHA256SUMS: OK")

artifact_root = root / "artifacts" / "parc_selector"
for line in (artifact_root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    expected, relative = line.split(maxsplit=1)
    path = root / relative
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise SystemExit(f"Artifact SHA256 mismatch: {relative}: {digest}")
print("Artifact SHA256SUMS: OK")
PY
PYTHON="$PYTHON" bash scripts/run_parc_self_test.sh
PYTHON="$PYTHON" bash scripts/run_parc_example.sh "${1:-$ROOT/examples/synthetic_routing_output.json}"
echo "PASS PARC release verification"
