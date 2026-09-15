#!/usr/bin/env python3
"""Compatibility runtime for the shift-robust PARC selector.

The learned model is strictly per query and contains no task/profile,
benchmark, family, task-name, or model-identity features.  Task boundaries are
used only for a frozen label-free CJK structural rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import parc_selector as selector
import parc_compat_runtime as legacy


PROTOCOL = "Successor-v3-shift-robust-selector-runtime-20260719"
# Patched to the verified model digest by the freeze/controller stage.
EXPECTED_MODEL_SHA256 = "688016ccdc6446cb1a1782206c3de5545cb23ffeda65bdacfbf6da240b95c102"
EPS = 1e-12

EXPECTED_POLICY = {
    "alpha": 100.0,
    "target": "softsign0.1",
    "weight_scheme": "family",
    "confidence_threshold": 0.02,
    "route_margin_threshold": 0.01,
    "safe_default": selector.SAFE_METHOD,
    "cjk_group_ratio_threshold": 0.4,
    "cjk_dense_normalized_margin_split": 0.1,
    "cjk_query_ratio_threshold": 0.75,
    "cjk_low_margin_method": selector.DOC_ORDER_METHOD,
    "cjk_high_margin_method": selector.LANG_MARGIN_METHOD,
    "task_activation_quota": None,
    "task_prior_weight": 0.0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_schema() -> list[str]:
    names = [f"query.{name}" for name in selector.QUERY_FEATURES]
    names += [
        f"candidate.{name}" for name in selector.CANDIDATE_FEATURES
    ]
    names += [
        f"candidate_minus_dense.{name}"
        for name in selector.CANDIDATE_FEATURES
    ]
    names += [
        f"enhanced_candidate.{name}"
        for name in selector.ENHANCED_MODEL_FEATURES
    ]
    names += [
        f"enhanced_candidate_minus_dense.{name}"
        for name in selector.ENHANCED_MODEL_FEATURES
    ]
    return names


def construct_feature_vectors(
    queries: Sequence[legacy.QueryObservation],
    model: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Construct only query/candidate features; no group profiles enter ridge."""
    dense_method = str(model["dense_baseline"])
    candidate_methods = [str(item) for item in model["candidate_methods"]]
    base_rows: dict[tuple[str, str], dict[str, float]] = {}
    enhanced_rows: dict[tuple[str, str], dict[str, float]] = {}
    required_methods = [
        dense_method,
        str(model["safe_default"]),
        *candidate_methods,
    ]
    for query in queries:
        dense_observation = query.methods[dense_method]
        for method in required_methods:
            key = (query.query_key, method)
            observation = query.methods[method]
            base_rows[key] = legacy.base_feature_row(
                observation,
                dense_observation,
            )
            enhanced_rows[key] = legacy.enhanced_feature_row(
                query.question,
                observation,
            )
    expected_schema = feature_schema()
    if list(model["feature_schema"]) != expected_schema:
        raise ValueError("frozen identity-free feature schema mismatch")
    vectors: dict[str, list[list[float]]] = {
        method: [] for method in candidate_methods
    }
    for query in queries:
        dense_base = base_rows[(query.query_key, dense_method)]
        dense_enhanced = enhanced_rows[(query.query_key, dense_method)]
        for method in candidate_methods:
            candidate_base = base_rows[(query.query_key, method)]
            candidate_enhanced = enhanced_rows[(query.query_key, method)]
            named: dict[str, float] = {}
            named.update(
                {
                    f"query.{field}": dense_base[field]
                    for field in selector.QUERY_FEATURES
                }
            )
            named.update(
                {
                    f"candidate.{field}": candidate_base[field]
                    for field in selector.CANDIDATE_FEATURES
                }
            )
            named.update(
                {
                    f"candidate_minus_dense.{field}": (
                        candidate_base[field] - dense_base[field]
                    )
                    for field in selector.CANDIDATE_FEATURES
                }
            )
            named.update(
                {
                    f"enhanced_candidate.{field}": candidate_enhanced[field]
                    for field in selector.ENHANCED_MODEL_FEATURES
                }
            )
            named.update(
                {
                    f"enhanced_candidate_minus_dense.{field}": (
                        candidate_enhanced[field] - dense_enhanced[field]
                    )
                    for field in selector.ENHANCED_MODEL_FEATURES
                }
            )
            if list(named) != expected_schema:
                raise ValueError(f"{method}: runtime feature order mismatch")
            vectors[method].append(
                [legacy._finite(named[field], field) for field in expected_schema]
            )
    query_cjk_ratio = np.asarray(
        [
            enhanced_rows[(query.query_key, dense_method)][
                "question_cjk_ratio"
            ]
            for query in queries
        ],
        dtype=np.float64,
    )
    dense_top1 = np.asarray(
        [
            enhanced_rows[(query.query_key, dense_method)]["dense_score_top1"]
            for query in queries
        ],
        dtype=np.float64,
    )
    dense_margin = np.asarray(
        [
            enhanced_rows[(query.query_key, dense_method)][
                "dense_score_top2_margin"
            ]
            for query in queries
        ],
        dtype=np.float64,
    )
    profile = {
        "question_cjk_ratio_mean": float(query_cjk_ratio.mean()),
        "dense_normalized_top2_margin_mean": float(
            np.mean(dense_margin / np.maximum(np.abs(dense_top1), EPS))
        ),
        "query_cjk_ratio": query_cjk_ratio,
    }
    return (
        {
            method: np.asarray(rows, dtype=np.float64)
            for method, rows in vectors.items()
        },
        profile,
    )


def predict(artifact: Mapping[str, Any], matrix: np.ndarray) -> np.ndarray:
    means = np.asarray(artifact["means"], dtype=np.float64)
    scales = np.asarray(artifact["scales"], dtype=np.float64)
    coefficients = np.asarray(artifact["coefficients"], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(means):
        raise ValueError("inference feature dimension mismatch")
    if np.any(scales <= 0.0):
        raise ValueError("frozen model contains non-positive scales")
    standardized = (matrix - means) / scales
    quadratic = int(artifact["quadratic_feature_count"])
    design = np.concatenate(
        [
            np.ones((len(matrix), 1), dtype=np.float64),
            standardized,
            standardized[:, :quadratic] ** 2,
        ],
        axis=1,
    )
    if design.shape[1] != len(coefficients):
        raise ValueError("frozen coefficient dimension mismatch")
    values = design @ coefficients
    if not np.all(np.isfinite(values)):
        raise ValueError("selector produced non-finite values")
    return values


def select_methods_from_scores(
    score_matrix: np.ndarray,
    methods: Sequence[str],
    profile: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Pure frozen routing rule, separated for independent unit tests."""
    if score_matrix.ndim != 2 or score_matrix.shape[1] != len(methods):
        raise ValueError("score matrix shape mismatch")
    if len(methods) < 2:
        raise ValueError("at least two learned candidates are required")
    order = np.argsort(-score_matrix, axis=1, kind="stable")
    best_indices = order[:, 0]
    best = score_matrix[np.arange(len(score_matrix)), best_indices]
    second = score_matrix[np.arange(len(score_matrix)), order[:, 1]]
    selected = np.asarray(
        [str(methods[int(index)]) for index in best_indices],
        dtype=object,
    )
    active = (
        (best > float(policy["confidence_threshold"]))
        & (
            (best - second)
            > float(policy["route_margin_threshold"])
        )
    )
    cjk_route: str | None = None
    if (
        float(profile["question_cjk_ratio_mean"])
        > float(policy["cjk_group_ratio_threshold"])
    ):
        cjk_route = (
            str(policy["cjk_low_margin_method"])
            if float(profile["dense_normalized_top2_margin_mean"])
            <= float(policy["cjk_dense_normalized_margin_split"])
            else str(policy["cjk_high_margin_method"])
        )
        cjk_queries = np.asarray(
            profile["query_cjk_ratio"],
            dtype=np.float64,
        ) > float(policy["cjk_query_ratio_threshold"])
        if cjk_queries.shape != active.shape:
            raise ValueError("CJK query profile length mismatch")
        active[:] = cjk_queries
        selected[cjk_queries] = cjk_route
    return selected, active, cjk_route


def load_frozen_model(path: Path) -> tuple[dict[str, Any], str]:
    observed = sha256_file(path)
    if len(EXPECTED_MODEL_SHA256) != 64:
        raise ValueError("runtime is not finalized with a frozen model digest")
    if observed != EXPECTED_MODEL_SHA256:
        raise ValueError(
            f"frozen model SHA256 {observed} != {EXPECTED_MODEL_SHA256}"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("frozen model must be an object")
    required = {
        "protocol",
        "status",
        "policy",
        "safe_default",
        "dense_baseline",
        "candidate_methods",
        "feature_schema",
        "quadratic_feature_schema",
        "models",
        "inference_forbidden_inputs",
    }
    if not required.issubset(value):
        raise ValueError("frozen model missing required fields")
    if value["status"] != "FROZEN_POST_FAILURE_NOT_RECONFIRMED":
        raise ValueError("frozen model status mismatch")
    if value["policy"] != EXPECTED_POLICY:
        raise ValueError("frozen policy mismatch")
    if value["safe_default"] != selector.SAFE_METHOD:
        raise ValueError("safe-default mismatch")
    if value["dense_baseline"] != selector.DENSE_METHOD:
        raise ValueError("Dense-baseline mismatch")
    if value["feature_schema"] != feature_schema():
        raise ValueError("feature schema mismatch")
    if len(value["feature_schema"]) != 212:
        raise ValueError("feature count mismatch")
    if set(value["models"]) != set(value["candidate_methods"]):
        raise ValueError("candidate/model coverage mismatch")
    return value, observed


def route_observations(
    queries: Sequence[legacy.QueryObservation],
    model: Mapping[str, Any],
    model_sha256: str,
) -> dict[str, Any]:
    if not queries:
        raise ValueError("cannot route an empty task")
    vectors, profile = construct_feature_vectors(queries, model)
    methods = [str(item) for item in model["candidate_methods"]]
    score_matrix = np.stack(
        [
            predict(model["models"][method], vectors[method])
            for method in methods
        ],
        axis=1,
    )
    selected, active, cjk_route = select_methods_from_scores(
        score_matrix,
        methods,
        profile,
        model["policy"],
    )
    safe = str(model["safe_default"])
    decisions: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for index, query in enumerate(queries):
        method = str(selected[index]) if active[index] else safe
        counts[method] += 1
        decisions.append(
            {
                "query_key": query.query_key,
                "selected_method": method,
                "selection_source": (
                    "cjk_structural_router"
                    if cjk_route is not None and active[index]
                    else (
                        "frozen_query_selector"
                        if active[index]
                        else "safe_default"
                    )
                ),
            }
        )
    return {
        "status": "COMPLETE_RETRIEVAL_ONLY_ROUTING_V3",
        "protocol": PROTOCOL,
        "model_sha256": model_sha256,
        "queries": len(queries),
        "generation_started_before_routing_complete": False,
        "forbidden_quality_inputs_read": False,
        "task_or_benchmark_identity_read": False,
        "learned_group_profile_features_used": False,
        "cjk_structural_route": cjk_route,
        "selected_method_counts": dict(sorted(counts.items())),
        "decisions": decisions,
    }


def route_public_task(
    values: Sequence[Mapping[str, Any]],
    model_path: Path,
) -> dict[str, Any]:
    model, digest = load_frozen_model(model_path)
    queries = legacy.parse_public_task(values, model)
    return route_observations(queries, model, digest)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--task-input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    values = json.loads(args.task_input.read_text(encoding="utf-8"))
    result = route_public_task(values, args.model)
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "queries": result["queries"],
                "model_sha256": result["model_sha256"],
            },
            sort_keys=True,
        )
    )
