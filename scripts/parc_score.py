#!/usr/bin/env python3
"""Score a complete PARC generation file with pinned official evaluators."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src" / "parc_pipeline"),
    str(ROOT / "src" / "parc_benchmarks"),
]

import parc_benchmark_adapters as adapters
from parc_contracts import (
    assert_fields,
    evaluation_identity,
    read_completed,
    unit_identity,
    validate_formatter,
    validate_inventory,
)
from parc_io import atomic_json, canonical_sha256, sha256_file
from parc_scoring import iter_gold, load_official_scorers, summarize


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--repos-root",
        type=Path,
        required=True,
        help="Directory containing pinned BAMBOO, ETHIC, and HELMET repositories",
    )
    parser.add_argument("--bamboo-root", type=Path, required=True)
    parser.add_argument("--ethic-root", type=Path, required=True)
    parser.add_argument("--helmet-root", type=Path, required=True)
    parser.add_argument("--longtable-root", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_formatter(args.longtable_root)
    generation_rows, generation_manifest = read_completed(args.generation)
    if generation_manifest["config"]["protocol"] != "PARC-generation-v1":
        raise ValueError("not a PARC generation artifact")
    generated: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for record in generation_rows:
        unit_key = (
            str(record["family"]),
            str(record["evaluation_key"]),
            int(record["round_index"]),
        )
        method = str(record["method"])
        if method not in {"dense", "parc"}:
            raise ValueError(f"unknown generation method: {method}")
        methods = generated.setdefault(unit_key, {})
        if method in methods:
            raise ValueError(f"duplicate generation record: {method}/{unit_key}")
        if not isinstance(record["prediction"], str):
            raise ValueError("prediction must be a string")
        compression = float(record["chunk_compression"])
        if (method == "dense" and compression != 0.0) or (
            method == "parc" and not 0.0 < compression < 1.0
        ):
            raise ValueError("invalid recorded chunk compression")
        methods[method] = record
    incomplete = [
        key for key, methods in generated.items() if set(methods) != {"dense", "parc"}
    ]
    if incomplete:
        raise ValueError(f"generation contains {len(incomplete)} unpaired units")
    expected = {}
    for evaluation in adapters.iter_all_evaluations(
        bamboo_root=args.bamboo_root.resolve(),
        ethic_root=args.ethic_root.resolve(),
        helmet_data_root=args.helmet_root.resolve(),
        longtable_root=args.longtable_root.resolve(),
    ):
        identity_hash = evaluation_identity(evaluation)
        for index, round_spec in enumerate(evaluation.rounds):
            key = (evaluation.family, evaluation.evaluation_key, index)
            if key in expected:
                raise ValueError("duplicate expected unit")
            expected[key] = unit_identity(evaluation, index, identity_hash)
    validate_inventory(expected.values(), ROOT / "configs/parc_data_inventory.json")
    if set(generated) - set(expected):
        raise ValueError("unexpected generated unit")
    for key, methods in generated.items():
        for record in methods.values():
            assert_fields(record, expected[key])
        if (
            methods["dense"]["retrieval_record_sha256"]
            != methods["parc"]["retrieval_record_sha256"]
        ):
            raise ValueError("paired methods did not use the same retrieval artifact")
    gold_all = {}
    for unit in iter_gold(
        adapters,
        args.bamboo_root.resolve(),
        args.ethic_root.resolve(),
        args.helmet_root.resolve(),
        args.longtable_root.resolve(),
    ):
        if unit.key in gold_all:
            raise ValueError("duplicate gold unit")
        if unit.key not in expected or unit.task != expected[unit.key]["task"]:
            raise ValueError("gold/adapter identity mismatch")
        gold_all[unit.key] = unit
    if set(gold_all) != set(expected):
        raise ValueError("gold coverage mismatch")
    if args.allow_partial:
        missing_gold = set(generated) - set(gold_all)
        if missing_gold:
            raise ValueError(f"missing gold units: {len(missing_gold)}")
        gold = {key: gold_all[key] for key in generated}
    else:
        gold = gold_all
        if set(generated) != set(gold):
            raise ValueError(
                f"full inventory mismatch: generated={len(generated)}, gold={len(gold)}"
            )
    scorers = load_official_scorers(
        args.repos_root.resolve(), args.longtable_root.resolve(), sha256_file
    )
    rows, summary = summarize(generated, gold, scorers)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    task_path = output_dir / "parc_task_scores.csv"
    with task_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    artifact = {
        "status": "COMPLETE" if not args.allow_partial else "PARTIAL_EXPLICIT",
        "protocol": "PARC-official-scoring-v1",
        "generation_sha256": sha256_file(args.generation),
        "official_scorer_sha256": dict(scorers.source_hashes),
        "summary": summary,
        "gold_inventory_sha256": canonical_sha256(
            [{"key": unit.key, "gold": unit.gold} for unit in gold_all.values()]
        ),
        "task_scores_csv": task_path.name,
        "task_scores_sha256": sha256_file(task_path),
        "contains_predictions": False,
        "contains_gold_answers": False,
    }
    atomic_json(output_dir / "parc_score_summary.json", artifact)
    print(json.dumps(artifact, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
