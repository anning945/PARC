"""Official-scorer loading and complete-inventory PARC aggregation."""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import string
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping

import numpy as np

EXPECTED_SCORER_SHA256 = {
    "BAMBOO/evaluate.py": "ada960cedf2a6bbc01694b1a941ce68439998397b703f0ef88716a35af339ee6",
    "ETHIC/utils.py": "6c5873b24d43216f0758f340fc3914c3cd086e05afcf0ce2f465218ab3548a56",
    "HELMET/utils.py": "86ceef1f3acbd9865eec8d95a5464a6e2d315e1d82ea6b9ae690582bf0ad1a50",
    "HELMET/data.py": "559977d8f97357c77f2bc1a554e7d02dffebce99f450ba034864ca3e07c09348",
    "HELMET/eval.py": "99eaaebe11efaa4c2d7932e6120cdbc2ad5104e273e352c299f2455285125909",
    "LongTableBench/eval/metrics.py": "df37ecc2116cd62218af4da019fc234674fc4ce7fae3d14057aac0f3e3fe3b2d",
}


@dataclass(frozen=True)
class GoldUnit:
    family: str
    task: str
    evaluation_key: str
    round_index: int
    gold: Any
    user_message: str = ""

    @property
    def key(self) -> tuple[str, str, int]:
        return self.family, self.evaluation_key, self.round_index


@dataclass(frozen=True)
class OfficialScorers:
    bamboo: Callable[..., Any]
    ethic: Callable[..., Any]
    helmet_f1: Callable[..., Any]
    helmet_max: Callable[..., Any]
    helmet_parse: Callable[..., Any]
    longtable: Callable[..., Any]
    source_hashes: Mapping[str, str]


def _extract_functions(
    path: Path, names: set[str], namespace: MutableMapping[str, Any]
) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    missing = names - {node.name for node in selected}
    if missing:
        raise ValueError(f"{path}: missing official functions {sorted(missing)}")
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)


def load_official_scorers(
    repos_root: Path, longtable_root: Path, sha256_file: Callable[[Path], str]
) -> OfficialScorers:
    from sklearn.metrics import f1_score, precision_score, recall_score

    paths = {
        "BAMBOO/evaluate.py": repos_root / "BAMBOO/evaluate.py",
        "ETHIC/utils.py": repos_root / "ETHIC/utils.py",
        "HELMET/utils.py": repos_root / "HELMET/utils.py",
        "HELMET/data.py": repos_root / "HELMET/data.py",
        "HELMET/eval.py": repos_root / "HELMET/eval.py",
        "LongTableBench/eval/metrics.py": longtable_root / "eval/metrics.py",
    }
    hashes: dict[str, str] = {}
    for relative, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        hashes[relative] = sha256_file(path)
        if hashes[relative] != EXPECTED_SCORER_SHA256[relative]:
            raise ValueError(
                f"{relative}: scorer hash does not match the pinned paper revision"
            )
    bamboo_ns: dict[str, Any] = {
        "precision_score": precision_score,
        "recall_score": recall_score,
        "f1_score": f1_score,
    }
    _extract_functions(paths["BAMBOO/evaluate.py"], {"calculate_metrics"}, bamboo_ns)
    ethic_ns: dict[str, Any] = {"re": re}
    _extract_functions(
        paths["ETHIC/utils.py"],
        {"calculate_f1_score", "calculate_lcs", "calculate_score"},
        ethic_ns,
    )
    helmet_ns: dict[str, Any] = {
        "re": re,
        "string": string,
        "unicodedata": unicodedata,
        "Counter": Counter,
    }
    _extract_functions(
        paths["HELMET/utils.py"],
        {
            "normalize_answer",
            "f1_score",
            "drqa_metric_max_over_ground_truths",
            "parse_output",
        },
        helmet_ns,
    )
    path = paths["LongTableBench/eval/metrics.py"]
    spec = importlib.util.spec_from_file_location(
        "parc_official_longtable_metrics", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "cal_score3", None)):
        raise ValueError("LongTableBench cal_score3 is missing")
    return OfficialScorers(
        bamboo=bamboo_ns["calculate_metrics"],
        ethic=ethic_ns["calculate_score"],
        helmet_f1=helmet_ns["f1_score"],
        helmet_max=helmet_ns["drqa_metric_max_over_ground_truths"],
        helmet_parse=helmet_ns["parse_output"],
        longtable=module.cal_score3,
        source_hashes=hashes,
    )


def iter_gold(
    adapters: Any,
    bamboo_root: Path,
    ethic_root: Path,
    helmet_root: Path,
    longtable_root: Path,
) -> Iterable[GoldUnit]:
    for task in adapters.BAMBOO_TASKS:
        path = bamboo_root / "datasets" / f"{task}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                if line.strip():
                    row = json.loads(line)
                    yield GoldUnit(
                        "BAMBOO",
                        task,
                        adapters.opaque_key("BAMBOO-eval", task, row_index),
                        0,
                        row["answer"],
                    )
    for task, (archive_name, member_name, _) in adapters.ETHIC_SPECS.items():
        with zipfile.ZipFile(ethic_root / archive_name) as archive:
            with archive.open(member_name, "r") as handle:
                for row_index, raw in enumerate(handle):
                    row = json.loads(raw)
                    yield GoldUnit(
                        "ETHIC",
                        task,
                        adapters.opaque_key("ETHIC-eval", task, row["ID"], row_index),
                        0,
                        row["Answer"],
                        str(row["User_msg"]),
                    )
    for task, spec in adapters.HELMET_SPECS.items():
        with (helmet_root / str(spec["test"])).open(encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                if task == "kilt_popqa_3" and not adapters.helmet_popqa_eligible(row):
                    continue
                yield GoldUnit(
                    "HELMET",
                    task,
                    adapters.opaque_key(
                        "HELMET-eval",
                        task,
                        row_index,
                        row[str(spec["key"])],
                        row.get("depth", ""),
                    ),
                    0,
                    row["answers"],
                )
    for relative in adapters.LONGTABLE_QUESTION_FILES:
        rows = json.loads((longtable_root / relative).read_text(encoding="utf-8"))
        multi = relative.endswith("longtablebench_multi.json")
        length_band, content_type = (
            relative.split("/")[2],
            "multi" if multi else "single",
        )
        for row_index, item in enumerate(rows):
            answers = (
                [row["answer"] for row in item["QA"]] if multi else [item["answer"]]
            )
            for table_format in adapters.LONGTABLE_FORMATS:
                task = f"{length_band}_{content_type}_{table_format}"
                key = adapters.opaque_key(
                    "LongTableBench-eval", relative, row_index, table_format
                )
                for round_index, answer in enumerate(answers):
                    yield GoldUnit("LongTableBench", task, key, round_index, answer)


def bamboo_prediction(value: str) -> bool:
    return "yes" in value.strip()[:3].lower()


def bamboo_gold(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError("BAMBOO gold must be bool, 0, or 1")


def score_scalar(
    scorers: OfficialScorers, gold: GoldUnit, prediction: str
) -> dict[str, float]:
    if gold.family == "ETHIC":
        _, score = scorers.ethic(gold.task, gold.user_message, prediction, gold.gold)
        return {"f1": float(score)}
    if gold.family == "HELMET":
        metric = lambda pred, answer: float(scorers.helmet_f1(pred, answer)[0])
        raw = float(scorers.helmet_max(metric, prediction, gold.gold))
        parsed_value = scorers.helmet_parse(prediction, "Answer:")
        parsed = (
            0.0
            if parsed_value is None
            else float(scorers.helmet_max(metric, parsed_value, gold.gold))
        )
        return {"f1": max(raw, parsed), "raw_f1": raw, "parsed_f1": parsed}
    if gold.family == "LongTableBench":
        precision, recall, f1 = scorers.longtable(gold.gold, prediction)
        return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
    raise ValueError(f"unsupported family: {gold.family}")


def summarize(
    records: Mapping[tuple[str, str, int], Mapping[str, Mapping[str, Any]]],
    gold: Mapping[tuple[str, str, int], GoldUnit],
    scorers: OfficialScorers,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not gold or set(records) != set(gold):
        raise ValueError("empty or mismatched scoring inventory")
    methods = ("dense", "parc")
    task_units: dict[
        tuple[str, str], list[tuple[GoldUnit, Mapping[str, Mapping[str, Any]]]]
    ] = defaultdict(list)
    for key in sorted(gold):
        if key not in records:
            raise ValueError(f"missing generated unit: {key}")
        task_units[(gold[key].family, gold[key].task)].append((gold[key], records[key]))
    task_rows: list[dict[str, Any]] = []
    for (family, task), units in sorted(task_units.items()):
        method_scores: dict[str, float] = {}
        conversation_scores: dict[str, float | None] = {
            method: None for method in methods
        }
        for method in methods:
            if family == "BAMBOO":
                labels = [bamboo_gold(unit.gold) for unit, _ in units]
                predictions = [
                    bamboo_prediction(str(rows[method]["prediction"]))
                    for unit, rows in units
                ]
                _, _, f1 = scorers.bamboo(labels, predictions)
                method_scores[method] = float(f1)
            else:
                values = [
                    score_scalar(scorers, unit, str(rows[method]["prediction"]))["f1"]
                    for unit, rows in units
                ]
                if not all(np.isfinite(value) and 0 <= value <= 1 for value in values):
                    raise ValueError("official scorer returned invalid F1")
                method_scores[method] = float(np.mean(values))
                if family == "LongTableBench":
                    conversations: dict[str, list[float]] = defaultdict(list)
                    turns: dict[str, list[int]] = defaultdict(list)
                    for (unit, _), value in zip(units, values):
                        conversations[unit.evaluation_key].append(value)
                        turns[unit.evaluation_key].append(unit.round_index)
                    if any(
                        sorted(indices) != list(range(len(indices)))
                        for indices in turns.values()
                    ):
                        raise ValueError("noncontiguous LongTableBench conversation")
                    conversation_scores[method] = float(
                        np.mean([np.mean(items) for items in conversations.values()])
                    )
        compression = [float(rows["parc"]["chunk_compression"]) for _, rows in units]
        task_rows.append(
            {
                "family": family,
                "task": task,
                "evaluations": len({unit.evaluation_key for unit, _ in units}),
                "scored_units": len(units),
                "dense_f1": method_scores["dense"],
                "parc_f1": method_scores["parc"],
                "delta_f1": method_scores["parc"] - method_scores["dense"],
                "dense_f1_per_conversation": conversation_scores["dense"],
                "parc_f1_per_conversation": conversation_scores["parc"],
                "mean_chunk_compression": float(np.mean(compression)),
                "minimum_chunk_compression": float(np.min(compression)),
            }
        )
    deltas = np.asarray([row["delta_f1"] for row in task_rows], dtype=np.float64)
    rng = np.random.default_rng(20260720)
    bootstrap = np.mean(
        rng.choice(deltas, size=(10000, len(deltas)), replace=True), axis=1
    )
    families: dict[str, Any] = {}
    for family in sorted({row["family"] for row in task_rows}):
        rows = [row for row in task_rows if row["family"] == family]
        family_compression = [
            float(methods["parc"]["chunk_compression"])
            for key, methods in records.items()
            if key[0] == family
        ]
        families[family] = {
            "tasks": len(rows),
            "dense_macro_f1": float(np.mean([row["dense_f1"] for row in rows])),
            "parc_macro_f1": float(np.mean([row["parc_f1"] for row in rows])),
            "mean_chunk_compression": float(np.mean(family_compression)),
        }
    all_compression = [
        float(methods["parc"]["chunk_compression"]) for methods in records.values()
    ]
    summary = {
        "tasks": len(task_rows),
        "scored_units": sum(row["scored_units"] for row in task_rows),
        "dense_macro_f1": float(np.mean([row["dense_f1"] for row in task_rows])),
        "parc_macro_f1": float(np.mean([row["parc_f1"] for row in task_rows])),
        "delta_macro_f1": float(np.mean(deltas)),
        "task_wins": int(np.sum(deltas > 0)),
        "task_ties": int(np.sum(deltas == 0)),
        "task_losses": int(np.sum(deltas < 0)),
        "mean_chunk_compression": float(np.mean(all_compression)),
        "minimum_task_compression": float(
            np.min([row["minimum_chunk_compression"] for row in task_rows])
        ),
        "bootstrap_95_ci_delta": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "bootstrap_unit": "task configuration; descriptive, not the paper's clustered significance test",
        "families": families,
    }
    return task_rows, summary
