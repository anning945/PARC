#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${PARC_DATA_ROOT:-$ROOT/data/benchmarks}"
BAMBOO_ROOT="${PARC_BAMBOO_ROOT:-$DATA_ROOT/BAMBOO}"
ETHIC_ROOT="${PARC_ETHIC_ROOT:-$DATA_ROOT/ETHIC}"
HELMET_ROOT="${PARC_HELMET_ROOT:-$DATA_ROOT/HELMET/data/kilt}"
LONGTABLE_ROOT="${PARC_LONGTABLE_ROOT:-$DATA_ROOT/LongTableBench}"
OUTPUT_DIR="${PARC_SCHEMA_OUTPUT:-$ROOT/outputs/parc_schema_check}"
STUB_OUTPUT="${PARC_SELECTOR_STUB_OUTPUT:-}"

for path in "$BAMBOO_ROOT" "$ETHIC_ROOT" "$HELMET_ROOT" "$LONGTABLE_ROOT"; do
  if [[ ! -d "$path" ]]; then
    echo "Missing benchmark path: $path" >&2
    echo "See docs/reproduction_workflow.md for the expected layout." >&2
    exit 2
  fi
done

ARGS=(
  --bamboo-root "$BAMBOO_ROOT"
  --ethic-root "$ETHIC_ROOT"
  --helmet-data-root "$HELMET_ROOT"
  --longtable-root "$LONGTABLE_ROOT"
  --output-dir "$OUTPUT_DIR"
)
if [[ -n "$STUB_OUTPUT" ]]; then
  ARGS+=(--selector-stub-output "$STUB_OUTPUT")
fi

PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" "$ROOT/src/parc_benchmarks/parc_benchmark_adapters.py" "${ARGS[@]}"
