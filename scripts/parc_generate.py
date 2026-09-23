#!/usr/bin/env python3
"""Generate Dense and PARC answers from materialized retrieval records."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src" / "parc_pipeline"),
    str(ROOT / "src" / "parc_benchmarks"),
]

import parc_benchmark_adapters as adapters
from parc_contracts import (
    assert_fields,
    begin_run,
    check_record,
    environment_identity,
    evaluation_identity,
    finish_run,
    read_completed,
    resolve_model,
    seal_record,
    source_identity,
    unit_identity,
    validate_formatter,
    validate_inventory,
)
from parc_generation import build_prompt_plan, generate_one, load_generator, pack_prompt
from parc_io import (
    append_jsonl,
    canonical_sha256,
    load_unique_jsonl,
    sha256_file,
    stable_shard,
)
from parc_retrieval import selected_record

PROTOCOL = "PARC-generation-v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/parc_generation.json"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--bamboo-root", type=Path, required=True)
    parser.add_argument("--ethic-root", type=Path, required=True)
    parser.add_argument("--helmet-root", type=Path, required=True)
    parser.add_argument("--longtable-root", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit-evaluations", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args(argv)
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("invalid shard selection")
    if args.limit_evaluations is not None and args.limit_evaluations <= 0:
        parser.error("--limit-evaluations must be positive")
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    generation_config = json.loads(args.config.read_text(encoding="utf-8"))
    retrieval_rows, retrieval_manifest = read_completed(args.retrieval)
    if retrieval_manifest["config"]["protocol"] != "PARC-retrieval-materialization-v1":
        raise ValueError("not a PARC retrieval artifact")
    retrieval = {(row["query_key"],): row for row in retrieval_rows}
    if len(retrieval) != len(retrieval_rows):
        raise ValueError("duplicate retrieval query keys")
    if args.limit_evaluations is None:
        validate_inventory(retrieval_rows, ROOT / "configs/parc_data_inventory.json")
    model_path, model_identity = resolve_model(args.model, args.model_revision)
    config = {
        "protocol": PROTOCOL,
        "generation_config_sha256": sha256_file(args.config),
        "formatter_sha256": validate_formatter(args.longtable_root),
        "retrieval_sha256": sha256_file(args.retrieval),
        "model": model_identity,
        "device_type": args.device.split(":")[0],
        "dtype": args.dtype,
        "source_sha256": source_identity(),
        "environment": environment_identity(),
    }
    scope = {
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "limit_evaluations": args.limit_evaluations,
    }
    config_sha256 = begin_run(args.output, config, scope)
    existing = load_unique_jsonl(args.output, key_fields=("method", "query_key"))
    for record in existing.values():
        check_record(record, config_sha256)
    model, tokenizer = load_generator(
        str(model_path),
        args.device,
        args.dtype,
        int(generation_config["decoding"]["seed"]),
    )
    counts: Counter[str] = Counter(record["method"] for record in existing.values())
    started = time.perf_counter()
    evaluation_count = 0
    seen: set[tuple[str, str]] = set()
    evaluations = adapters.iter_all_evaluations(
        bamboo_root=args.bamboo_root.resolve(),
        ethic_root=args.ethic_root.resolve(),
        helmet_data_root=args.helmet_root.resolve(),
        longtable_root=args.longtable_root.resolve(),
    )
    for evaluation in evaluations:
        if (
            stable_shard(evaluation.evaluation_key, args.shard_count)
            != args.shard_index
        ):
            continue
        evaluation_count += 1
        if (
            args.limit_evaluations is not None
            and evaluation_count > args.limit_evaluations
        ):
            break
        input_hash = evaluation_identity(evaluation)
        histories = {"dense": [], "parc": []}
        for round_index, round_spec in enumerate(evaluation.rounds):
            retrieval_record = retrieval.get((round_spec.query_key,))
            if retrieval_record is None:
                raise ValueError(f"missing retrieval record for {round_spec.query_key}")
            identity = unit_identity(evaluation, round_index, input_hash)
            assert_fields(retrieval_record, identity)
            assert_fields(
                retrieval_record,
                selected_record(
                    retrieval_record["methods"], retrieval_record["selected_method"]
                ),
            )
            for method in ("dense", "parc"):
                key = (method, round_spec.query_key)
                if key in seen:
                    raise ValueError("duplicate adapter generation key")
                seen.add(key)
                context = {
                    **identity,
                    "method": method,
                    "retrieval_record_sha256": retrieval_record["record_sha256"],
                    "history_sha256": canonical_sha256(histories[method]),
                }
                if key in existing:
                    assert_fields(existing[key], context)
                    histories[method].append(str(existing[key]["prediction"]))
                    continue
                indices = (
                    retrieval_record["dense_indices"]
                    if method == "dense"
                    else retrieval_record["selected_indices"]
                )
                plan = build_prompt_plan(
                    evaluation,
                    indices,
                    round_index,
                    histories[method],
                    args.longtable_root.resolve(),
                    adapters,
                )
                packed = pack_prompt(tokenizer, plan, generation_config)
                generated = generate_one(
                    model,
                    tokenizer,
                    packed,
                    args.device,
                    int(
                        generation_config["max_new_tokens_by_family"][evaluation.family]
                    ),
                    bool(generation_config["decoding"]["use_cache"]),
                )
                record = {
                    "protocol": PROTOCOL,
                    "config_sha256": config_sha256,
                    **context,
                    "prompt_sha256": canonical_sha256(packed["prompt"]),
                    "selected_method": "FullPool_Dense_k8"
                    if method == "dense"
                    else retrieval_record["selected_method"],
                    "chunk_compression": 0.0
                    if method == "dense"
                    else float(retrieval_record["chunk_compression"]),
                    **generated,
                }
                record = seal_record(record)
                append_jsonl(args.output, record)
                existing[key] = record
                histories[method].append(str(record["prediction"]))
                counts[method] += 1
        if evaluation_count == 1 or evaluation_count % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "stage": "generation",
                        "evaluations": evaluation_count,
                        "records": sum(counts.values()),
                        "elapsed_s": round(time.perf_counter() - started, 1),
                    }
                ),
                flush=True,
            )
    expected_complete = args.shard_count == 1 and args.limit_evaluations is None
    if not seen or set(existing) != seen:
        raise ValueError("generation output has missing or extra keys")
    if expected_complete:
        for method in ("dense", "parc"):
            validate_inventory(
                (row for row in existing.values() if row["method"] == method),
                ROOT / "configs/parc_data_inventory.json",
            )
    manifest = finish_run(args.output, config, scope, sum(counts.values()))
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "scope": scope,
                "records": manifest["records"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
