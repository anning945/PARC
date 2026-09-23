#!/usr/bin/env python3
"""Validate and merge all retrieval or generation shards, without rescoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/parc_pipeline"))

from parc_contracts import finish_run, read_completed, validate_inventory
from parc_io import canonical_json, sha256_file, stable_shard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        args.output.exists()
        or args.output.with_suffix(args.output.suffix + ".manifest.json").exists()
    ):
        raise ValueError("merge output already exists; choose a new path")
    manifests, rows, keys, indices = [], [], set(), set()
    for path in args.inputs:
        values, manifest = read_completed(path)
        scope = manifest["scope"]
        if scope.get("limit") is not None or scope.get("limit_evaluations") is not None:
            raise ValueError("cannot merge smoke runs into a formal artifact")
        if manifests and manifest["config_sha256"] != manifests[0]["config_sha256"]:
            raise ValueError("shard configurations differ")
        if scope["shard_count"] != len(args.inputs) or scope["shard_index"] in indices:
            raise ValueError("missing or duplicate shard index")
        indices.add(scope["shard_index"])
        for row in values:
            key = (row.get("method", "retrieval"), row["query_key"])
            if key in keys:
                raise ValueError("duplicate key across shards")
            if (
                stable_shard(row["evaluation_key"], len(args.inputs))
                != scope["shard_index"]
            ):
                raise ValueError("record in wrong shard")
            keys.add(key)
        rows.extend(values)
        manifests.append(manifest)
    if indices != set(range(len(args.inputs))):
        raise ValueError("incomplete shard set")
    protocol = manifests[0]["config"]["protocol"]
    if protocol == "PARC-generation-v1":
        if {row["method"] for row in rows} != {"dense", "parc"}:
            raise ValueError("unexpected generation methods")
        for method in ("dense", "parc"):
            validate_inventory(
                (row for row in rows if row["method"] == method),
                ROOT / "configs/parc_data_inventory.json",
            )
    elif protocol == "PARC-retrieval-materialization-v1":
        validate_inventory(rows, ROOT / "configs/parc_data_inventory.json")
    else:
        raise ValueError("unsupported pipeline artifact")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in sorted(
            rows, key=lambda row: (row["query_key"], row.get("method", ""))
        ):
            handle.write(canonical_json(row) + "\n")
    temporary.replace(args.output)
    scope = {
        "shard_count": 1,
        "shard_index": 0,
        "limit": None,
        "merged_shards": [sha256_file(path) for path in args.inputs],
    }
    result = finish_run(args.output, manifests[0]["config"], scope, len(rows))
    print(json.dumps({"status": result["status"], "records": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
