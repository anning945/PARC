"""Identity, resume, and complete-inventory contracts shared by pipeline CLIs."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import os
import platform
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from parc_io import atomic_json, canonical_sha256, iter_jsonl, sha256_file

ROOT = Path(__file__).resolve().parents[2]


def source_identity() -> str:
    paths = sorted((ROOT / "src").rglob("*.py")) + sorted(
        (ROOT / "scripts").glob("parc_*.py")
    )
    return canonical_sha256(
        {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}
    )


def environment_identity() -> dict[str, str]:
    result = {
        "python": platform.python_version(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "unset"),
    }
    for package in (
        "numpy",
        "scikit-learn",
        "torch",
        "transformers",
        "sentence-transformers",
        "pandas",
        "langchain-text-splitters",
        "beautifulsoup4",
        "sqlparse",
        "tabulate",
        "inflect",
        "roman",
        "nltk",
        "word2number",
    ):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def resolve_model(
    value: str, revision: str | None = None
) -> tuple[Path, dict[str, Any]]:
    """Resolve once, hash actual weights/tokenizer files, and load that snapshot."""
    path = Path(value).expanduser()
    if not path.is_dir():
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(repo_id=value, revision=revision))
    path = path.resolve()
    files = {
        item.relative_to(path).as_posix(): sha256_file(item)
        for item in sorted(path.rglob("*"))
        if item.is_file() and ".cache" not in item.relative_to(path).parts
    }
    if not files:
        raise ValueError(f"empty model directory: {path}")
    return path, {
        "requested": value,
        "revision": revision,
        "files": files,
        "sha256": canonical_sha256(files),
    }


def evaluation_identity(evaluation: Any) -> str:
    # Includes prompt inputs, table row order, and released demonstrations,
    # but never untouched test answers.
    return canonical_sha256(dataclasses.asdict(evaluation))


def validate_formatter(root: Path) -> str:
    value = sha256_file(root / "eval/reformat.py")
    if value != "c35555b97646749529ebe0faef493e37c98f73e1e99419d8b9745965e7e514f7":
        raise ValueError("LongTableBench formatter does not match the pinned revision")
    return value


def unit_identity(evaluation: Any, round_index: int, input_hash: str) -> dict[str, Any]:
    return {
        "family": evaluation.family,
        "task": evaluation.task,
        "evaluation_key": evaluation.evaluation_key,
        "round_index": round_index,
        "query_key": evaluation.rounds[round_index].query_key,
        "input_sha256": input_hash,
    }


def assert_fields(record: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(
                f"record {field} mismatch for {record.get('query_key', '?')}"
            )


def seal_record(record: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    return {**body, "record_sha256": canonical_sha256(body)}


def check_record(record: Mapping[str, Any], config_hash: str) -> None:
    assert_fields(record, {"config_sha256": config_hash})
    if record.get("record_sha256") != seal_record(record)["record_sha256"]:
        raise ValueError("record SHA256 mismatch")


def manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def begin_run(path: Path, config: Mapping[str, Any], scope: Mapping[str, Any]) -> str:
    config_hash = canonical_sha256(config)
    sidecar = manifest_path(path)
    if sidecar.exists():
        previous = json.loads(sidecar.read_text(encoding="utf-8"))
        assert_fields(
            previous,
            {
                "config": dict(config),
                "config_sha256": config_hash,
                "scope": dict(scope),
            },
        )
        if previous.get("status") == "COMPLETE" and previous.get(
            "output_sha256"
        ) != sha256_file(path):
            raise ValueError("completed output SHA256 mismatch")
    elif path.exists():
        raise ValueError("cannot resume an output without its configuration manifest")
    atomic_json(
        sidecar,
        {
            "status": "RUNNING",
            "config": dict(config),
            "config_sha256": config_hash,
            "scope": dict(scope),
        },
    )
    return config_hash


def finish_run(
    path: Path, config: Mapping[str, Any], scope: Mapping[str, Any], records: int
) -> dict[str, Any]:
    manifest = {
        "status": "COMPLETE",
        "config": dict(config),
        "config_sha256": canonical_sha256(config),
        "scope": dict(scope),
        "records": records,
        "output_sha256": sha256_file(path),
    }
    atomic_json(manifest_path(path), manifest)
    return manifest


def read_completed(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path(path).read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE":
        raise ValueError("upstream stage is not complete")
    if manifest.get("config_sha256") != canonical_sha256(manifest["config"]):
        raise ValueError("manifest configuration SHA256 mismatch")
    if manifest.get("output_sha256") != sha256_file(path):
        raise ValueError("upstream output SHA256 mismatch")
    rows = list(iter_jsonl(path))
    if len(rows) != manifest["records"] or not rows:
        raise ValueError("manifest record count mismatch or empty output")
    for row in rows:
        check_record(row, manifest["config_sha256"])
    return rows, manifest


def validate_inventory(
    records: Iterable[Mapping[str, Any]], inventory_path: Path
) -> None:
    expected_rows = json.loads(inventory_path.read_text(encoding="utf-8"))[
        "evaluation_tasks"
    ]
    expected = {
        (row["family"], row["task"]): (row["evaluations"], row["scored_units"])
        for row in expected_rows
    }
    counts: Counter[tuple[str, str]] = Counter()
    evaluations: dict[tuple[str, str], set[str]] = defaultdict(set)
    seen: set[str] = set()
    for row in records:
        key = str(row["query_key"])
        if key in seen:
            raise ValueError("duplicate query in complete inventory")
        seen.add(key)
        task = (str(row["family"]), str(row["task"]))
        counts[task] += 1
        evaluations[task].add(str(row["evaluation_key"]))
    observed = {task: (len(evaluations[task]), count) for task, count in counts.items()}
    if observed != expected:
        raise ValueError(
            "complete evaluation inventory mismatch (task names, evaluations, or scored units)"
        )
