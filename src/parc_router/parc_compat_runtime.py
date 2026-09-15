#!/usr/bin/env python3
"""Compatibility runtime for the frozen PARC selector.

The public routing API accepts one complete task at a time.  Each query
contains only an opaque key, the question, the original chunks, and retrieval
indices/scores for the frozen Dense, safe, and candidate methods.  Task,
benchmark, family, model, gold, prediction, and score fields are rejected.

Task-wide retrieval-only profiles and all routing decisions are materialized
before any generation is allowed to start.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import parc_feature_extractor as enhanced
import parc_selector as selector


PROTOCOL = "Successor-v2-selector-runtime-20260718"
EXPECTED_MODEL_SHA256 = (
    "ac633d046f06791d6b843360f5752a7791d7f7a542bb3b15ef308ab1ed986450"
)
EPS = 1e-12

QUERY_FIELDS = frozenset({"query_key", "question", "chunks", "methods"})
METHOD_FIELDS = frozenset(
    {
        "kept_indices",
        "retrieved_indices",
        "dense_top_k_indices",
        "bm25_top_k_indices",
        "rrf_top_k_indices",
        "retrieved_dense_scores",
        "retrieved_bm25_scores",
        "retrieved_rrf_scores",
        "retrieved_reranker_scores",
    }
)
FORBIDDEN_FIELD_FRAGMENTS = (
    "answer",
    "benchmark",
    "candidate_prediction",
    "dense_delta",
    "dense_prediction",
    "family",
    "f1",
    "generator",
    "gold",
    "label",
    "metric",
    "model_identity",
    "official_score",
    "oracle",
    "prediction",
    "score_target",
    "task_identity",
)


@dataclass(frozen=True)
class MethodObservation:
    """Retrieval-only observation used by feature construction."""

    original_chunks: int
    original_text_chars: int
    candidate_text_chars: int
    retrieved_context: str
    kept_indices: tuple[int, ...]
    retrieved_indices: tuple[int, ...]
    dense_top_k_indices: tuple[int, ...]
    bm25_top_k_indices: tuple[int, ...]
    rrf_top_k_indices: tuple[int, ...]
    retrieved_dense_scores: tuple[float, ...]
    retrieved_bm25_scores: tuple[float, ...]
    retrieved_rrf_scores: tuple[float, ...]
    retrieved_reranker_scores: tuple[float, ...]


@dataclass(frozen=True)
class QueryObservation:
    query_key: str
    question: str
    methods: Mapping[str, MethodObservation]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field}: non-finite value")
    return result


def _finite_vector(value: Any, field: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field}: expected list")
    return tuple(_finite(item, field) for item in value)


def _index_vector(
    value: Any,
    field: str,
    *,
    chunk_count: int,
) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field}: expected list")
    output: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"{field}: booleans are not indices")
        index = int(item)
        if index != item or not 0 <= index < chunk_count:
            raise ValueError(f"{field}: invalid chunk index {item!r}")
        output.append(index)
    if len(output) != len(set(output)):
        raise ValueError(f"{field}: duplicate indices")
    return tuple(output)


def _check_forbidden_fields(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            for fragment in FORBIDDEN_FIELD_FRAGMENTS:
                if fragment in lowered:
                    raise ValueError(
                        f"{path}.{key}: forbidden inference field"
                    )
            _check_forbidden_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_forbidden_fields(child, f"{path}[{index}]")


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    path: str,
) -> None:
    observed = set(value)
    if observed != set(expected):
        missing = sorted(set(expected) - observed)
        extra = sorted(observed - set(expected))
        raise ValueError(
            f"{path}: schema mismatch; missing={missing}, extra={extra}"
        )


def _method_observation_from_public(
    *,
    method: str,
    value: Mapping[str, Any],
    chunks: Sequence[str],
) -> MethodObservation:
    _require_exact_fields(value, METHOD_FIELDS, f"methods.{method}")
    count = len(chunks)
    kept = _index_vector(
        value["kept_indices"],
        f"{method}.kept_indices",
        chunk_count=count,
    )
    retrieved = _index_vector(
        value["retrieved_indices"],
        f"{method}.retrieved_indices",
        chunk_count=count,
    )
    dense = _index_vector(
        value["dense_top_k_indices"],
        f"{method}.dense_top_k_indices",
        chunk_count=count,
    )
    bm25 = _index_vector(
        value["bm25_top_k_indices"],
        f"{method}.bm25_top_k_indices",
        chunk_count=count,
    )
    rrf = _index_vector(
        value["rrf_top_k_indices"],
        f"{method}.rrf_top_k_indices",
        chunk_count=count,
    )
    dense_scores = _finite_vector(
        value["retrieved_dense_scores"],
        f"{method}.retrieved_dense_scores",
    )
    bm25_scores = _finite_vector(
        value["retrieved_bm25_scores"],
        f"{method}.retrieved_bm25_scores",
    )
    rrf_scores = _finite_vector(
        value["retrieved_rrf_scores"],
        f"{method}.retrieved_rrf_scores",
    )
    reranker_scores = _finite_vector(
        value["retrieved_reranker_scores"],
        f"{method}.retrieved_reranker_scores",
    )
    for field, scores in (
        ("retrieved_dense_scores", dense_scores),
        ("retrieved_bm25_scores", bm25_scores),
        ("retrieved_rrf_scores", rrf_scores),
    ):
        if len(scores) != len(retrieved):
            raise ValueError(
                f"{method}.{field}: {len(scores)} scores for "
                f"{len(retrieved)} retrieved indices"
            )
    if reranker_scores and len(reranker_scores) != len(retrieved):
        raise ValueError(
            f"{method}.retrieved_reranker_scores: {len(reranker_scores)} "
            f"scores for {len(retrieved)} retrieved indices"
        )
    if not set(retrieved).issubset(set(kept)):
        raise ValueError(
            f"{method}: retrieved indices are not contained in kept pool"
        )
    reference_lengths = {len(dense), len(bm25), len(rrf)}
    if len(reference_lengths) != 1:
        raise ValueError(
            f"{method}: Dense/BM25/RRF reference lengths differ"
        )
    if len(dense) < len(retrieved):
        raise ValueError(
            f"{method}: reference top-k is shorter than retrieved top-k"
        )
    return MethodObservation(
        original_chunks=count,
        original_text_chars=sum(len(chunk) for chunk in chunks),
        candidate_text_chars=sum(len(chunks[index]) for index in kept),
        retrieved_context="\n\n".join(chunks[index] for index in retrieved),
        kept_indices=kept,
        retrieved_indices=retrieved,
        dense_top_k_indices=dense,
        bm25_top_k_indices=bm25,
        rrf_top_k_indices=rrf,
        retrieved_dense_scores=dense_scores,
        retrieved_bm25_scores=bm25_scores,
        retrieved_rrf_scores=rrf_scores,
        retrieved_reranker_scores=reranker_scores,
    )


def observation_from_retrieval_artifact(
    value: Mapping[str, Any],
) -> MethodObservation:
    """Build an observation from opened-development artifacts for parity tests.

    This helper is intentionally not used by the public inference entry point.
    It permits exact replay of the already-opened development retrieval files,
    which store derived text lengths and retrieved context instead of all
    original chunks.
    """

    original_chunks = int(value["original_chunks"])
    kept = tuple(int(item) for item in value["kept_indices"])
    retrieved = tuple(int(item) for item in value["retrieved_indices"])
    for name, indices in (
        ("kept_indices", kept),
        ("retrieved_indices", retrieved),
        ("dense_top_k_indices", value["dense_top_k_indices"]),
        ("bm25_top_k_indices", value["bm25_top_k_indices"]),
        ("rrf_top_k_indices", value["rrf_top_k_indices"]),
    ):
        parsed = tuple(int(item) for item in indices)
        if any(not 0 <= item < original_chunks for item in parsed):
            raise ValueError(f"development replay {name}: invalid index")
        if len(parsed) != len(set(parsed)):
            raise ValueError(f"development replay {name}: duplicate index")
    return MethodObservation(
        original_chunks=original_chunks,
        original_text_chars=int(value["original_text_chars"]),
        candidate_text_chars=int(value["candidate_text_chars"]),
        retrieved_context=str(value["retrieved_context"]),
        kept_indices=kept,
        retrieved_indices=retrieved,
        dense_top_k_indices=tuple(
            int(item) for item in value["dense_top_k_indices"]
        ),
        bm25_top_k_indices=tuple(
            int(item) for item in value["bm25_top_k_indices"]
        ),
        rrf_top_k_indices=tuple(
            int(item) for item in value["rrf_top_k_indices"]
        ),
        retrieved_dense_scores=tuple(
            _finite(item, "retrieved_dense_scores")
            for item in value["retrieved_dense_scores"]
        ),
        retrieved_bm25_scores=tuple(
            _finite(item, "retrieved_bm25_scores")
            for item in value["retrieved_bm25_scores"]
        ),
        retrieved_rrf_scores=tuple(
            _finite(item, "retrieved_rrf_scores")
            for item in value["retrieved_rrf_scores"]
        ),
        retrieved_reranker_scores=tuple(
            _finite(item, "retrieved_reranker_scores")
            for item in value["retrieved_reranker_scores"]
        ),
    )


def parse_public_task(
    values: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
) -> list[QueryObservation]:
    if not isinstance(values, list) or not values:
        raise ValueError("task input must be a non-empty JSON list")
    required_methods = {
        str(model["dense_baseline"]),
        str(model["safe_default"]),
        *(str(method) for method in model["candidate_methods"]),
    }
    output: list[QueryObservation] = []
    seen_keys: set[str] = set()
    for position, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise ValueError(f"queries[{position}]: expected object")
        _check_forbidden_fields(value, f"queries[{position}]")
        _require_exact_fields(value, QUERY_FIELDS, f"queries[{position}]")
        query_key = str(value["query_key"])
        if not query_key or query_key in seen_keys:
            raise ValueError(f"queries[{position}]: invalid/duplicate query_key")
        seen_keys.add(query_key)
        question = value["question"]
        chunks = value["chunks"]
        methods = value["methods"]
        if not isinstance(question, str):
            raise ValueError(f"queries[{position}].question: expected string")
        if (
            not isinstance(chunks, list)
            or not chunks
            or any(not isinstance(chunk, str) for chunk in chunks)
        ):
            raise ValueError(
                f"queries[{position}].chunks: expected non-empty string list"
            )
        if not isinstance(methods, Mapping):
            raise ValueError(f"queries[{position}].methods: expected object")
        if set(methods) != required_methods:
            raise ValueError(
                f"queries[{position}].methods: coverage mismatch; "
                f"missing={sorted(required_methods - set(methods))}, "
                f"extra={sorted(set(methods) - required_methods)}"
            )
        parsed = {
            method: _method_observation_from_public(
                method=method,
                value=methods[method],
                chunks=chunks,
            )
            for method in sorted(required_methods)
        }
        dense_reference = parsed[str(model["dense_baseline"])]
        for method, observation in parsed.items():
            if observation.original_chunks != dense_reference.original_chunks:
                raise ValueError(f"{query_key}/{method}: chunk-count mismatch")
            if (
                method != str(model["dense_baseline"])
                and len(observation.kept_indices)
                >= observation.original_chunks
            ):
                raise ValueError(
                    f"{query_key}/{method}: non-positive chunk compression"
                )
            for field in (
                "dense_top_k_indices",
                "bm25_top_k_indices",
                "rrf_top_k_indices",
            ):
                if getattr(observation, field) != getattr(
                    dense_reference, field
                ):
                    raise ValueError(
                        f"{query_key}/{method}: inconsistent {field}"
                    )
        output.append(
            QueryObservation(
                query_key=query_key,
                question=question,
                methods=parsed,
            )
        )
    return output


def _jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _range(values: Sequence[float]) -> float:
    return max(values) - min(values) if values else 0.0


def base_feature_row(
    observation: MethodObservation,
    dense_observation: MethodObservation,
) -> dict[str, float]:
    original_chunks = observation.original_chunks
    if original_chunks <= 0:
        raise ValueError("original chunk count must be positive")
    if observation.original_text_chars <= 0:
        raise ValueError("original text must be non-empty")
    retrieved = observation.retrieved_indices
    dense_retrieved = dense_observation.retrieved_indices
    kept_count = len(observation.kept_indices)
    return {
        "log1p_original_chunks": math.log1p(original_chunks),
        "candidate_pool_fraction": kept_count / original_chunks,
        "chunk_compression": 1.0 - kept_count / original_chunks,
        "char_compression": (
            1.0
            - observation.candidate_text_chars
            / observation.original_text_chars
        ),
        "retrieved_order_same_dense": float(retrieved == dense_retrieved),
        "retrieved_set_same_dense": float(
            set(retrieved) == set(dense_retrieved)
        ),
        "retrieved_dense_set_jaccard": _jaccard(
            retrieved, dense_retrieved
        ),
        "dense_bm25_topk_jaccard": _jaccard(
            observation.dense_top_k_indices,
            observation.bm25_top_k_indices,
        ),
        "dense_rrf_topk_jaccard": _jaccard(
            observation.dense_top_k_indices,
            observation.rrf_top_k_indices,
        ),
        "bm25_rrf_topk_jaccard": _jaccard(
            observation.bm25_top_k_indices,
            observation.rrf_top_k_indices,
        ),
        "hybrid_replacement_count": float(
            len(
                set(retrieved)
                - set(observation.dense_top_k_indices)
            )
        ),
        "retrieved_dense_score_mean": _mean(
            observation.retrieved_dense_scores
        ),
        "retrieved_dense_score_range": _range(
            observation.retrieved_dense_scores
        ),
        "retrieved_bm25_score_mean": _mean(
            observation.retrieved_bm25_scores
        ),
        "retrieved_bm25_score_range": _range(
            observation.retrieved_bm25_scores
        ),
        "retrieved_rrf_score_mean": _mean(
            observation.retrieved_rrf_scores
        ),
        "retrieved_rrf_score_range": _range(
            observation.retrieved_rrf_scores
        ),
        "retrieved_reranker_score_mean": _mean(
            observation.retrieved_reranker_scores
        ),
        "retrieved_reranker_score_range": _range(
            observation.retrieved_reranker_scores
        ),
    }


def enhanced_feature_row(
    question: str,
    observation: MethodObservation,
) -> dict[str, float]:
    output: dict[str, float] = {
        "retrieved_context_to_original_ratio": (
            len(observation.retrieved_context)
            / max(observation.original_text_chars, EPS)
        ),
    }
    output.update(enhanced.question_features(question))
    output.update(
        enhanced.overlap_features(
            question,
            observation.retrieved_context,
            "context",
        )
    )
    output.update(
        enhanced.segment_overlap_features(
            question,
            observation.retrieved_context,
        )
    )
    output.update(
        enhanced.sequence_features(
            observation.retrieved_indices,
            observation.original_chunks,
            "retrieved_index",
        )
    )
    output.update(
        enhanced.sequence_features(
            observation.dense_top_k_indices,
            observation.original_chunks,
            "dense_index",
        )
    )
    output.update(
        enhanced.sequence_features(
            observation.bm25_top_k_indices,
            observation.original_chunks,
            "bm25_index",
        )
    )
    output.update(
        enhanced.sequence_features(
            observation.rrf_top_k_indices,
            observation.original_chunks,
            "rrf_index",
        )
    )
    output.update(
        enhanced.sequence_features(
            observation.kept_indices,
            observation.original_chunks,
            "kept_index",
        )
    )
    output.update(
        enhanced.permutation_features(
            observation.retrieved_indices,
            observation.dense_top_k_indices,
        )
    )
    output.update(
        enhanced.score_features(
            observation.retrieved_dense_scores,
            "dense_score",
        )
    )
    output.update(
        enhanced.score_features(
            observation.retrieved_bm25_scores,
            "bm25_score",
        )
    )
    output.update(
        enhanced.score_features(
            observation.retrieved_rrf_scores,
            "rrf_score",
        )
    )
    output.update(
        enhanced.score_features(
            observation.retrieved_reranker_scores,
            "reranker_score",
        )
    )
    return output


def _profile(
    rows: Sequence[Mapping[str, float]],
    fields: Sequence[str],
    statistics_names: Sequence[str],
) -> dict[tuple[str, str], float]:
    matrix = np.asarray(
        [[_finite(row[field], field) for field in fields] for row in rows],
        dtype=np.float64,
    )
    parts: dict[str, np.ndarray] = {
        "mean": matrix.mean(axis=0),
        "std": matrix.std(axis=0),
        "q25": np.quantile(matrix, 0.25, axis=0),
        "q50": np.quantile(matrix, 0.50, axis=0),
        "q75": np.quantile(matrix, 0.75, axis=0),
    }
    return {
        (statistic, field): float(parts[statistic][index])
        for statistic in statistics_names
        for index, field in enumerate(fields)
    }


def construct_feature_vectors(
    queries: Sequence[QueryObservation],
    model: Mapping[str, Any],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, Any],
    dict[tuple[str, str], dict[str, float]],
]:
    dense_method = str(model["dense_baseline"])
    candidate_methods = [str(method) for method in model["candidate_methods"]]
    base_rows: dict[tuple[str, str], dict[str, float]] = {}
    enhanced_rows: dict[tuple[str, str], dict[str, float]] = {}
    for query in queries:
        dense_observation = query.methods[dense_method]
        for method in [dense_method, str(model["safe_default"]), *candidate_methods]:
            key = (query.query_key, method)
            observation = query.methods[method]
            base_rows[key] = base_feature_row(
                observation,
                dense_observation,
            )
            enhanced_rows[key] = enhanced_feature_row(
                query.question,
                observation,
            )

    dense_base_rows = [
        base_rows[(query.query_key, dense_method)] for query in queries
    ]
    dense_enhanced_rows = [
        enhanced_rows[(query.query_key, dense_method)] for query in queries
    ]
    base_profile = _profile(
        dense_base_rows,
        selector.PROFILE_FEATURES,
        ("mean", "std", "q25", "q50", "q75"),
    )
    enhanced_profile = _profile(
        dense_enhanced_rows,
        selector.ENHANCED_PROFILE_FEATURES,
        ("mean", "std"),
    )

    vectors: dict[str, list[list[float]]] = {
        method: [] for method in candidate_methods
    }
    named_features: dict[tuple[str, str], dict[str, float]] = {}
    for query in queries:
        dense_base = base_rows[(query.query_key, dense_method)]
        dense_enhanced = enhanced_rows[(query.query_key, dense_method)]
        for method in candidate_methods:
            base = base_rows[(query.query_key, method)]
            extra = enhanced_rows[(query.query_key, method)]
            named: dict[str, float] = {}
            named.update(
                {
                    f"query.{field}": dense_base[field]
                    for field in selector.QUERY_FEATURES
                }
            )
            named.update(
                {
                    f"candidate.{field}": base[field]
                    for field in selector.CANDIDATE_FEATURES
                }
            )
            named.update(
                {
                    f"candidate_minus_dense.{field}": (
                        base[field] - dense_base[field]
                    )
                    for field in selector.CANDIDATE_FEATURES
                }
            )
            named.update(
                {
                    f"task_profile.{statistic}.{field}": base_profile[
                        (statistic, field)
                    ]
                    for statistic in ("mean", "std", "q25", "q50", "q75")
                    for field in selector.PROFILE_FEATURES
                }
            )
            named.update(
                {
                    f"enhanced_candidate.{field}": extra[field]
                    for field in selector.ENHANCED_MODEL_FEATURES
                }
            )
            named.update(
                {
                    f"enhanced_candidate_minus_dense.{field}": (
                        extra[field] - dense_enhanced[field]
                    )
                    for field in selector.ENHANCED_MODEL_FEATURES
                }
            )
            named.update(
                {
                    f"enhanced_task_profile.{statistic}.{field}": (
                        enhanced_profile[(statistic, field)]
                    )
                    for statistic in ("mean", "std")
                    for field in selector.ENHANCED_PROFILE_FEATURES
                }
            )
            schema = list(model["feature_schema"])
            if list(named) != schema:
                raise ValueError(
                    f"{method}: runtime feature schema/order mismatch"
                )
            vector = [_finite(named[field], field) for field in schema]
            vectors[method].append(vector)
            named_features[(query.query_key, method)] = named

    query_cjk_ratio = np.asarray(
        [row["question_cjk_ratio"] for row in dense_enhanced_rows],
        dtype=np.float64,
    )
    dense_top1 = np.asarray(
        [row["dense_score_top1"] for row in dense_enhanced_rows],
        dtype=np.float64,
    )
    dense_margin = np.asarray(
        [row["dense_score_top2_margin"] for row in dense_enhanced_rows],
        dtype=np.float64,
    )
    dense_normalized_margin = dense_margin / np.maximum(
        np.abs(dense_top1),
        EPS,
    )
    structural_profile = {
        "question_cjk_ratio_mean": float(query_cjk_ratio.mean()),
        "dense_normalized_top2_margin_mean": float(
            dense_normalized_margin.mean()
        ),
        "query_cjk_ratio": query_cjk_ratio,
    }
    return (
        {
            method: np.asarray(rows, dtype=np.float64)
            for method, rows in vectors.items()
        },
        structural_profile,
        named_features,
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
    quadratic_count = int(artifact["quadratic_feature_count"])
    design = np.concatenate(
        [
            np.ones((len(matrix), 1), dtype=np.float64),
            standardized,
            standardized[:, :quadratic_count] ** 2,
        ],
        axis=1,
    )
    if design.shape[1] != len(coefficients):
        raise ValueError("frozen coefficient dimension mismatch")
    result = design @ coefficients
    if not np.all(np.isfinite(result)):
        raise ValueError("selector produced non-finite predictions")
    return result


def load_frozen_model(path: Path) -> tuple[dict[str, Any], str]:
    observed_hash = sha256_file(path)
    if observed_hash != EXPECTED_MODEL_SHA256:
        raise ValueError(
            f"frozen model SHA256 {observed_hash} != "
            f"{EXPECTED_MODEL_SHA256}"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("frozen model must be a JSON object")
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
        raise ValueError("frozen model is missing required fields")
    if value["status"] != "FROZEN_AFTER_OPENED_DEVELOPMENT_NOT_CONFIRMED":
        raise ValueError("frozen model status mismatch")
    if value["safe_default"] != selector.SAFE_METHOD:
        raise ValueError("safe-default mismatch")
    if value["dense_baseline"] != selector.DENSE_METHOD:
        raise ValueError("Dense-baseline mismatch")
    if len(value["feature_schema"]) != 324:
        raise ValueError("frozen feature count mismatch")
    if set(value["models"]) != set(value["candidate_methods"]):
        raise ValueError("frozen candidate/model coverage mismatch")
    policy = value["policy"]
    expected_policy = {
        "activation_fraction": 1.0,
        "alpha": 1000.0,
        "portfolio": "no_rrforder",
        "routing": "query",
        "safe_default": selector.SAFE_METHOD,
        "target": "softsign0.05",
        "task_prior_weight": 0.0,
        "weight_scheme": "family",
    }
    for field, expected in expected_policy.items():
        if policy.get(field) != expected:
            raise ValueError(f"frozen policy {field} mismatch")
    return value, observed_hash


def route_observations(
    queries: Sequence[QueryObservation],
    model: Mapping[str, Any],
    model_sha256: str,
) -> dict[str, Any]:
    if not queries:
        raise ValueError("cannot route an empty task")
    vectors, profile, _ = construct_feature_vectors(queries, model)
    methods = [str(method) for method in model["candidate_methods"]]
    score_matrix = np.stack(
        [predict(model["models"][method], vectors[method]) for method in methods],
        axis=1,
    )
    chosen_indices = np.argmax(score_matrix, axis=1)
    chosen_methods = np.asarray(
        [methods[int(index)] for index in chosen_indices],
        dtype=object,
    )
    active = np.ones(len(queries), dtype=bool)
    cjk_route = selector.choose_cjk_structural_route(profile)
    if cjk_route is not None:
        cjk_queries = (
            profile["query_cjk_ratio"]
            > selector.CJK_QUERY_RATIO_THRESHOLD
        )
        active[:] = cjk_queries
        chosen_methods[cjk_queries] = cjk_route
    decisions: list[dict[str, Any]] = []
    selected = Counter()
    for index, query in enumerate(queries):
        method = (
            str(chosen_methods[index])
            if active[index]
            else str(model["safe_default"])
        )
        selected[method] += 1
        decisions.append(
            {
                "query_key": query.query_key,
                "selected_method": method,
                "selection_source": (
                    "cjk_structural_router"
                    if cjk_route is not None and active[index]
                    else (
                        "frozen_selector"
                        if active[index]
                        else "safe_default"
                    )
                ),
            }
        )
    return {
        "status": "COMPLETE_RETRIEVAL_ONLY_ROUTING",
        "protocol": PROTOCOL,
        "model_sha256": model_sha256,
        "queries": len(queries),
        "generation_started_before_routing_complete": False,
        "forbidden_quality_inputs_read": False,
        "cjk_structural_route": cjk_route,
        "selected_method_counts": dict(sorted(selected.items())),
        "decisions": decisions,
    }


def route_public_task(
    values: Sequence[Mapping[str, Any]],
    model_path: Path,
) -> dict[str, Any]:
    model, model_hash = load_frozen_model(model_path)
    queries = parse_public_task(values, model)
    return route_observations(queries, model, model_hash)


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


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
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


if __name__ == "__main__":
    main()
