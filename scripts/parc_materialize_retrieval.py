#!/usr/bin/env python3
"""Materialize PARC retrieval observations and frozen routing decisions."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src" / "parc_pipeline"),
    str(ROOT / "src" / "parc_benchmarks"),
    str(ROOT / "src" / "parc_router"),
]

import parc_benchmark_adapters as adapters
import parc_runtime as runtime
from parc_contracts import (
    assert_fields,
    begin_run,
    check_record,
    environment_identity,
    evaluation_identity,
    finish_run,
    resolve_model,
    seal_record,
    source_identity,
    unit_identity,
    validate_formatter,
    validate_inventory,
)
from parc_io import (
    append_jsonl,
    canonical_sha256,
    load_unique_jsonl,
    sha256_file,
    stable_shard,
)
from parc_retrieval import (
    SentenceTransformerEmbedder,
    TransformersReranker,
    materialize_methods,
    selected_record,
)

PROTOCOL = "PARC-retrieval-materialization-v1"


def chunks_key(chunks: Sequence[str]) -> str:
    return canonical_sha256(["PARC-chunks-v1", *chunks])


def cached_encode(cache_dir: Path, embedder: Any, texts: Sequence[str]) -> np.ndarray:
    import hashlib
    import os
    import tempfile

    path = cache_dir / f"{chunks_key(texts)}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            values = archive["values"]
            checksum = str(archive["sha256"].item())
        if hashlib.sha256(values.tobytes()).hexdigest() != checksum:
            raise ValueError("embedding cache SHA256 mismatch")
    else:
        # Match the formal cache boundary for both chunks and queries.
        values = (
            np.asarray(embedder.encode(texts), dtype=np.float32)
            .astype(np.float16)
            .astype(np.float32)
        )
        checksum = hashlib.sha256(values.tobytes()).hexdigest()
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, values=values, sha256=checksum)
        os.replace(temporary, path)
    if (
        values.dtype != np.float32
        or values.ndim != 2
        or len(values) != len(texts)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("invalid embedding cache matrix")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bamboo-root", type=Path, required=True)
    parser.add_argument("--ethic-root", type=Path, required=True)
    parser.add_argument("--helmet-root", type=Path, required=True)
    parser.add_argument("--longtable-root", type=Path, required=True)
    parser.add_argument("--embedder", required=True)
    parser.add_argument("--reranker", required=True)
    parser.add_argument("--embedder-revision")
    parser.add_argument("--reranker-revision")
    parser.add_argument(
        "--selector",
        type=Path,
        default=ROOT / "artifacts/parc_selector/parc_frozen_selector.joblib",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--rerank-batch-size", type=int, default=64)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args(argv)
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("invalid shard selection")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if (
        min(
            args.embed_batch_size,
            args.rerank_batch_size,
            args.reranker_max_length,
            args.progress_every,
        )
        <= 0
    ):
        parser.error("batch sizes, max length, and progress interval must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError(
            "launch retrieval with PYTHONHASHSEED=0 to fix BM25 term accumulation order"
        )
    embedder_path, embedder_identity = resolve_model(
        args.embedder, args.embedder_revision
    )
    reranker_path, reranker_identity = resolve_model(
        args.reranker, args.reranker_revision
    )
    config = {
        "protocol": PROTOCOL,
        "formatter_sha256": validate_formatter(args.longtable_root),
        "dataset_roots": {
            "BAMBOO": str(args.bamboo_root.resolve()),
            "ETHIC": str(args.ethic_root.resolve()),
            "HELMET": str(args.helmet_root.resolve()),
            "LongTableBench": str(args.longtable_root.resolve()),
        },
        "embedder": embedder_identity,
        "reranker": reranker_identity,
        "embed_batch_size": args.embed_batch_size,
        "rerank_batch_size": args.rerank_batch_size,
        "reranker_max_length": args.reranker_max_length,
        "device_type": args.device.split(":")[0],
        "source_sha256": source_identity(),
        "environment": environment_identity(),
        "selector_sha256": sha256_file(args.selector.resolve()),
        "selector_manifest_sha256": sha256_file(
            args.selector.resolve().with_name(runtime.MODEL_MANIFEST_NAME)
        ),
    }
    scope = {
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "limit": args.limit,
    }
    config_sha256 = begin_run(args.output, config, scope)
    existing = load_unique_jsonl(args.output, key_fields=("query_key",))
    for record in existing.values():
        check_record(record, config_sha256)
    embedder = SentenceTransformerEmbedder(
        str(embedder_path), args.device, args.embed_batch_size
    )
    reranker = TransformersReranker(
        str(reranker_path),
        args.device,
        args.rerank_batch_size,
        args.reranker_max_length,
    )
    cache_dir = args.cache_dir.resolve() / canonical_sha256(
        {
            "model": embedder_identity["sha256"],
            "environment": config["environment"],
            "batch_size": args.embed_batch_size,
            "device_type": config["device_type"],
        }
    )
    counts: Counter[str] = Counter(record["family"] for record in existing.values())
    started = time.perf_counter()
    visited = 0
    completed = len(existing)
    seen: set[tuple[str]] = set()
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
        input_hash = evaluation_identity(evaluation)
        for round_index, round_spec in enumerate(evaluation.rounds):
            visited += 1
            if args.limit is not None and visited > args.limit:
                break
            key = (round_spec.query_key,)
            if key in seen:
                raise ValueError("duplicate adapter query key")
            seen.add(key)
            identity = unit_identity(evaluation, round_index, input_hash)
            if key in existing:
                assert_fields(existing[key], identity)
                continue
            chunk_embeddings = cached_encode(cache_dir, embedder, evaluation.chunks)
            query_embedding = cached_encode(cache_dir, embedder, [round_spec.question])[
                0
            ]
            methods = materialize_methods(
                round_spec.question,
                evaluation.chunks,
                chunk_embeddings,
                query_embedding,
                reranker,
            )
            public_query = evaluation.runtime_query(round_index, methods)
            routed = runtime.route_public_task([public_query], args.selector.resolve())
            decision = routed["decisions"][0]
            selection = selected_record(methods, str(decision["selected_method"]))
            record = {
                "protocol": PROTOCOL,
                "config_sha256": config_sha256,
                **identity,
                "methods": methods,
                "selection_source": decision["selection_source"],
                **selection,
            }
            record = seal_record(record)
            append_jsonl(args.output, record)
            existing[(round_spec.query_key,)] = record
            counts[evaluation.family] += 1
            completed += 1
            if completed == 1 or completed % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "stage": "retrieval",
                            "completed": completed,
                            "elapsed_s": round(time.perf_counter() - started, 1),
                        }
                    ),
                    flush=True,
                )
        if args.limit is not None and visited >= args.limit:
            break
    expected_complete = args.shard_count == 1 and args.limit is None
    if not seen or set(existing) != seen:
        raise ValueError("retrieval output has missing or extra keys")
    if expected_complete:
        validate_inventory(existing.values(), ROOT / "configs/parc_data_inventory.json")
    manifest = finish_run(args.output, config, scope, completed)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "scope": scope,
                "records": completed,
                "families": dict(counts),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
