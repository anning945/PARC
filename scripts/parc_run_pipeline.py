#!/usr/bin/env python3
"""Run retrieval, generation, and official scoring from one JSON config."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("retrieval", "generation", "scoring"),
        default=("retrieval", "generation", "scoring"),
    )
    return parser.parse_args(argv)


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(
        command, cwd=ROOT, check=True, env={**os.environ, "PYTHONHASHSEED": "0"}
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data, models, runtime = config["data"], config["models"], config["runtime"]
    output_dir = Path(runtime["output_dir"])
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    cache_dir = Path(runtime["cache_dir"])
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    retrieval_path = output_dir / "parc_retrieval.jsonl"
    generation_path = output_dir / "parc_generation.jsonl"
    common_data = [
        "--bamboo-root",
        data["bamboo_root"],
        "--ethic-root",
        data["ethic_root"],
        "--helmet-root",
        data["helmet_root"],
        "--longtable-root",
        data["longtable_root"],
    ]
    if "retrieval" in args.stages:
        run(
            [
                sys.executable,
                "scripts/parc_materialize_retrieval.py",
                *common_data,
                "--embedder",
                models["embedder"],
                "--reranker",
                models["reranker"],
                "--output",
                str(retrieval_path),
                "--cache-dir",
                str(cache_dir),
                "--device",
                runtime["device"],
                *(
                    ["--embedder-revision", models["embedder_revision"]]
                    if models.get("embedder_revision")
                    else []
                ),
                *(
                    ["--reranker-revision", models["reranker_revision"]]
                    if models.get("reranker_revision")
                    else []
                ),
            ]
        )
    if "generation" in args.stages:
        run(
            [
                sys.executable,
                "scripts/parc_generate.py",
                *common_data,
                "--retrieval",
                str(retrieval_path),
                "--output",
                str(generation_path),
                "--model",
                models["generator"],
                "--device",
                runtime["device"],
                "--dtype",
                runtime.get("dtype", "bfloat16"),
                *(
                    ["--model-revision", models["generator_revision"]]
                    if models.get("generator_revision")
                    else []
                ),
            ]
        )
    if "scoring" in args.stages:
        run(
            [
                sys.executable,
                "scripts/parc_score.py",
                *common_data,
                "--generation",
                str(generation_path),
                "--output-dir",
                str(output_dir / "scores"),
                "--repos-root",
                config["official_repositories_root"],
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
