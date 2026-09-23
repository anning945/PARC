#!/usr/bin/env python3
"""Reaggregate verified task scores on the fixed current-paper inventory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "configs/parc_paper35_inventory.json"
SCORES = ROOT / "results/parc_paper35_task_results.json"
SUMMARY = ROOT / "results/parc_paper35_summary.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def unit_count(value):
    if isinstance(value, bool) or int(value) != float(value) or int(value) <= 0:
        raise ValueError("scored_units must be a positive integer")
    return int(value)


def fraction(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("expected a finite fraction in [0, 1]")
    return value


def validate_tasks(rows, expected):
    indexed = {}
    for row in rows:
        key = row["task_key"]
        if key in indexed or key != f'{row["family"]}/{row["task"]}':
            raise ValueError("duplicate or inconsistent task identity")
        indexed[key] = row
    targets = {row["task_key"]: row for row in expected}
    if set(indexed) != set(targets):
        raise ValueError("task coverage does not match the declared inventory")
    for key, row in indexed.items():
        if unit_count(row["scored_units"]) != targets[key]["scored_units"]:
            raise ValueError(f"scored-unit count mismatch: {key}")
    return indexed


def aggregate(rows):
    units = sum(unit_count(r["scored_units"]) for r in rows)
    dense = mean(fraction(r["dense_f1"]) for r in rows)
    parc = mean(fraction(r["parc_f1"]) for r in rows)
    return {
        "tasks": len(rows),
        "scored_units": units,
        "dense_macro_f1": dense,
        "parc_macro_f1": parc,
        "delta_f1_pp": 100 * (parc - dense),
        "mean_chunk_compression": math.fsum(
            fraction(r["mean_chunk_compression"]) * unit_count(r["scored_units"])
            for r in rows
        ) / units,
    }


def summarize_published(artifact, inventory):
    if artifact["scope_id"] != inventory["scope_id"]:
        raise ValueError("scope identifier mismatch")
    rows = artifact["tasks"]
    validate_tasks(rows, inventory["tasks"])
    controls = set(artifact["control_source_ids"])
    for row in rows:
        if set(row["qwen_controls"]) != controls:
            raise ValueError("component coverage mismatch")
        if row["parc_compressed_units"] != row["scored_units"]:
            raise ValueError("published all-pools-compressed claim is unsupported")
        if fraction(row["parc_compression"]) <= 0:
            raise ValueError("expected positive compression")
        for model in ("qwen", "llama"):
            delta = fraction(row[f"{model}_parc_f1"]) - fraction(row[f"{model}_dense_f1"])
            if not math.isclose(delta, row[f"{model}_delta"], rel_tol=0, abs_tol=1e-12):
                raise ValueError("stored F1 difference mismatch")
    result = {
        "scope_id": inventory["scope_id"],
        "status": artifact["status"],
        "aggregation": inventory["aggregation"],
        "compressed_units": sum(r["parc_compressed_units"] for r in rows),
        "models": {},
        "qwen_controls": {},
    }
    for model in ("qwen", "llama"):
        normalized = [dict(r, dense_f1=r[f"{model}_dense_f1"],
                           parc_f1=r[f"{model}_parc_f1"],
                           mean_chunk_compression=r["parc_compression"]) for r in rows]
        result["models"][model] = {
            "overall": aggregate(normalized),
            "families": {
                family: aggregate([r for r in normalized if r["family"] == family])
                for family in sorted({r["family"] for r in rows})
            },
        }
    for control in sorted(controls):
        normalized = []
        for row in rows:
            value = row["qwen_controls"][control]
            if value["compressed_units"] != row["scored_units"]:
                raise ValueError("component compressed-unit count mismatch")
            normalized.append(dict(row, dense_f1=row["qwen_dense_f1"],
                                   parc_f1=value["f1"],
                                   mean_chunk_compression=value["mean_chunk_compression"]))
        point = aggregate(normalized)
        point["method_macro_f1"] = point.pop("parc_macro_f1")
        result["qwen_controls"][control] = point
    return result


def summarize_replay(summary_path, inventory):
    artifact = read_json(summary_path)
    if artifact["status"] != "COMPLETE" or artifact["protocol"] != "PARC-official-scoring-v1":
        raise ValueError("replay requires COMPLETE official scoring, not partial results")
    task_path = summary_path.parent / artifact["task_scores_csv"]
    if hashlib.sha256(task_path.read_bytes()).hexdigest() != artifact["task_scores_sha256"]:
        raise ValueError("replay task-score checksum mismatch")
    with task_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["task_key"] = f'{row["family"]}/{row["task"]}'
    full = read_json(ROOT / "configs/parc_data_inventory.json")["evaluation_tasks"]
    indexed = validate_tasks(rows, full)
    selected = [indexed[r["task_key"]] for r in inventory["tasks"]]
    validate_tasks(selected, inventory["tasks"])
    return {
        "scope_id": inventory["scope_id"],
        "status": "REAGGREGATED_COMPLETE_REPLAY_TASK_SCORES",
        "source_score_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "aggregation": inventory["aggregation"],
        "overall": aggregate(selected),
        "families": {family: aggregate([r for r in selected if r["family"] == family])
                     for family in sorted({r["family"] for r in selected})},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Check the released summary")
    mode.add_argument("--replay-summary", type=Path, help="A new COMPLETE parc_score_summary.json")
    args = parser.parse_args()
    inventory = read_json(INVENTORY)
    result = (summarize_replay(args.replay_summary, inventory) if args.replay_summary
              else summarize_published(read_json(SCORES), inventory))
    if args.check and result != read_json(SUMMARY):
        raise ValueError("released summary differs from recomputed task aggregates")
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print("PASS current 35-task summary check" if args.check else encoded, end="\n" if args.check else "")


if __name__ == "__main__":
    main()
