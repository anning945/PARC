#!/usr/bin/env python3
"""Explore the identity-free Adaptive-v3 parc with a CJK router.

This is an opened-development diagnostic.  It intentionally reports strict
LOTO/LOFO task outcomes and never promotes a configuration to confirmation.
The safe default is PARC_CompressedDense_R45_k8, whose generated
answers are exactly Dense-equivalent on the verified development table while
retaining positive chunk compression.  A deterministic structural router
handles CJK-majority tasks using only question-script statistics and a
scale-normalized Dense retrieval margin; benchmark/task identity, gold
answers, predictions, and evaluation scores are not router inputs.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EPS = 1e-12
SAFE_METHOD = "PARC_CompressedDense_R45_k8"
DENSE_METHOD = "FullPool_Dense_k8"
DOC_ORDER_METHOD = "PARC_CompressedDense_DocOrder_R45_k8"
LANG_MARGIN_METHOD = (
    "PARC_Hybrid_RRFXE_LangMargin_D7A1_EN25_CJK20_R45_k8"
)

# Frozen opened-development structural rule.  The normalized margin is
# mean((top1 - top2) / abs(top1)) over Dense retrieval results for the task.
# Relative normalization avoids binding the rule to one model's score scale.
CJK_QUERY_RATIO_THRESHOLD = 0.5
CJK_TASK_MEAN_RATIO_THRESHOLD = 0.5
CJK_NORMALIZED_MARGIN_THRESHOLD = 0.105
CJK_ROUTER_MODE = "cjk_queries_else_safe"

QUERY_FEATURES = (
    "log1p_original_chunks",
    "dense_bm25_topk_jaccard",
    "dense_rrf_topk_jaccard",
    "bm25_rrf_topk_jaccard",
    "retrieved_dense_score_mean",
    "retrieved_dense_score_range",
    "retrieved_bm25_score_mean",
    "retrieved_bm25_score_range",
    "retrieved_rrf_score_mean",
    "retrieved_rrf_score_range",
)

CANDIDATE_FEATURES = (
    "candidate_pool_fraction",
    "chunk_compression",
    "char_compression",
    "retrieved_order_same_dense",
    "retrieved_set_same_dense",
    "retrieved_dense_set_jaccard",
    "hybrid_replacement_count",
    "retrieved_dense_score_mean",
    "retrieved_dense_score_range",
    "retrieved_bm25_score_mean",
    "retrieved_bm25_score_range",
    "retrieved_rrf_score_mean",
    "retrieved_rrf_score_range",
    "retrieved_reranker_score_mean",
    "retrieved_reranker_score_range",
)

ENHANCED_MODEL_FEATURES = (
    "question_log1p_chars",
    "question_english_words",
    "question_unique_word_ratio",
    "question_cjk_ratio",
    "question_digit_count",
    "question_digit_ratio",
    "question_number_token_count",
    "question_year_count",
    "question_punctuation_ratio",
    "question_unique_char_ratio",
    "question_mark_count",
    "question_quote_count",
    "question_negation_hits",
    "question_comparison_hits",
    "question_multihop_hits",
    "question_type_person",
    "question_type_time",
    "question_type_location",
    "question_type_reason",
    "question_type_manner",
    "question_type_quantity",
    "question_type_choice",
    "question_type_boolean",
    "question_type_definition",
    "question_type_count",
    "retrieved_context_to_original_ratio",
    "context_word_recall",
    "context_word_jaccard",
    "context_cjk_bigram_recall",
    "context_cjk_bigram_jaccard",
    "context_number_recall",
    "segment_count",
    "segment_overlap_mean",
    "segment_overlap_std",
    "segment_overlap_max",
    "segment_overlap_margin",
    "segment_overlap_best_rank_norm",
    "segment_overlap_entropy",
    "retrieved_index_mean_norm",
    "retrieved_index_std_norm",
    "retrieved_index_min_norm",
    "retrieved_index_max_norm",
    "retrieved_index_range_norm",
    "retrieved_index_first_norm",
    "retrieved_index_last_norm",
    "retrieved_index_mean_abs_gap_norm",
    "retrieved_index_adjacent_fraction",
    "retrieved_index_ascending_fraction",
    "retrieved_index_descending_fraction",
    "retrieved_index_early_fraction",
    "retrieved_index_late_fraction",
    "kept_index_mean_norm",
    "kept_index_std_norm",
    "kept_index_range_norm",
    "kept_index_mean_abs_gap_norm",
    "kept_index_adjacent_fraction",
    "kept_index_early_fraction",
    "kept_index_late_fraction",
    "candidate_dense_common_fraction",
    "candidate_dense_rank_displacement",
    "candidate_dense_rank_agreement",
    "candidate_dense_first_rank_norm",
    "dense_score_top1",
    "dense_score_top2_margin",
    "dense_score_std",
    "dense_score_cv_abs",
    "dense_score_entropy",
    "dense_score_slope",
    "bm25_score_top1",
    "bm25_score_top2_margin",
    "bm25_score_std",
    "bm25_score_cv_abs",
    "bm25_score_entropy",
    "bm25_score_slope",
    "rrf_score_top1",
    "rrf_score_top2_margin",
    "rrf_score_std",
    "rrf_score_cv_abs",
    "rrf_score_entropy",
    "rrf_score_slope",
    "reranker_score_top1",
    "reranker_score_top2_margin",
    "reranker_score_std",
    "reranker_score_cv_abs",
    "reranker_score_entropy",
    "reranker_score_slope",
)

ENHANCED_PROFILE_FEATURES = (
    "question_log1p_chars",
    "question_english_words",
    "question_cjk_ratio",
    "question_digit_ratio",
    "question_number_token_count",
    "question_year_count",
    "question_negation_hits",
    "question_comparison_hits",
    "question_multihop_hits",
    "question_type_person",
    "question_type_time",
    "question_type_location",
    "question_type_reason",
    "question_type_quantity",
    "question_type_boolean",
    "question_type_definition",
    "context_word_recall",
    "context_cjk_bigram_recall",
    "context_number_recall",
    "segment_overlap_mean",
    "segment_overlap_max",
    "segment_overlap_best_rank_norm",
    "retrieved_index_mean_norm",
    "retrieved_index_std_norm",
    "retrieved_index_range_norm",
    "retrieved_index_early_fraction",
    "retrieved_index_late_fraction",
    "dense_score_top2_margin",
    "dense_score_std",
    "dense_score_entropy",
    "bm25_score_top2_margin",
    "bm25_score_std",
    "bm25_score_entropy",
    "rrf_score_top2_margin",
    "rrf_score_std",
    "rrf_score_entropy",
)

PROFILE_FEATURES = (
    "log1p_original_chunks",
    "dense_bm25_topk_jaccard",
    "dense_rrf_topk_jaccard",
    "bm25_rrf_topk_jaccard",
    "retrieved_dense_score_mean",
    "retrieved_dense_score_range",
    "retrieved_bm25_score_mean",
    "retrieved_bm25_score_range",
)

PORTFOLIOS = {
    "all": None,
    "no_docorder": {
        "PARC_CompressedDense_RRFOrder_R45_k8",
        "PARC_CompressedDense_RerankOrder_R45_k8",
        "PARC_Hybrid_BM25_D7A1_R45_k8",
        "PARC_Hybrid_RRFXE_LangMargin_D7A1_EN25_CJK20_R35_k8",
        "PARC_Hybrid_RRFXE_LangMargin_D7A1_EN25_CJK20_R45_k8",
        "PARC_Hybrid_RRFXE_LangMargin_D7A1_EN25_CJK20_R55_k8",
        "PARC_Hybrid_RRF_D6A2_R45_k8",
        "PARC_Hybrid_Rerank_D7A1_M10_R45_k8",
    },
    "no_rrforder": {
        "PARC_CompressedDense_DocOrder_R45_k8",
        "PARC_CompressedDense_RerankOrder_R45_k8",
        "PARC_Hybrid_BM25_D7A1_R45_k8",
        "PARC_Hybrid_RRFXE_LangMargin_D7A1_EN25_CJK20_R35_k8",
        "PARC_Hybrid_RRFXE_LangMargin_D7A1_EN25_CJK20_R45_k8",
        "PARC_Hybrid_RRFXE_LangMargin_D7A1_EN25_CJK20_R55_k8",
        "PARC_Hybrid_RRF_D6A2_R45_k8",
        "PARC_Hybrid_Rerank_D7A1_M10_R45_k8",
    },
    "hybrid": {
        "PARC_Hybrid_BM25_D7A1_R45_k8",
        "PARC_Hybrid_RRFXE_LangMargin_D7A1_EN25_CJK20_R35_k8",
        "PARC_Hybrid_RRFXE_LangMargin_D7A1_EN25_CJK20_R45_k8",
        "PARC_Hybrid_RRFXE_LangMargin_D7A1_EN25_CJK20_R55_k8",
        "PARC_Hybrid_RRF_D6A2_R45_k8",
        "PARC_Hybrid_Rerank_D7A1_M10_R45_k8",
    },
    "robust_hybrid": {
        "PARC_Hybrid_RRFXE_LangMargin_D7A1_EN25_CJK20_R35_k8",
        "PARC_Hybrid_RRF_D6A2_R45_k8",
        "PARC_Hybrid_Rerank_D7A1_M10_R45_k8",
    },
    "order": {
        "PARC_CompressedDense_DocOrder_R45_k8",
        "PARC_CompressedDense_RRFOrder_R45_k8",
        "PARC_CompressedDense_RerankOrder_R45_k8",
    },
}


def f(row: Mapping[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"{key}: non-finite")
    return value


def choose_cjk_structural_route(
    profile: Mapping[str, float],
) -> str | None:
    """Return a method from label-free task structure, or no override."""
    if (
        profile["question_cjk_ratio_mean"]
        <= CJK_TASK_MEAN_RATIO_THRESHOLD
    ):
        return None
    if (
        profile["dense_normalized_top2_margin_mean"]
        <= CJK_NORMALIZED_MARGIN_THRESHOLD
    ):
        return DOC_ORDER_METHOD
    return LANG_MARGIN_METHOD


def load_enhanced_table(
    path: Path,
) -> dict[tuple[str, int, str], dict[str, str]]:
    output: dict[tuple[str, int, str], dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        required = {
            "dataset",
            "sample_index",
            "method",
            *ENHANCED_MODEL_FEATURES,
            *ENHANCED_PROFILE_FEATURES,
        }
        missing = sorted(required - columns)
        if missing:
            raise ValueError(f"enhanced features missing columns: {missing}")
        for row in reader:
            key = (row["dataset"], int(row["sample_index"]), row["method"])
            if key in output:
                raise ValueError(f"duplicate enhanced feature row: {key}")
            output[key] = row
    return output


def load_table(
    path: Path,
    enhanced_path: Path | None = None,
) -> dict[str, Any]:
    grouped: dict[tuple[str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    task_family: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            task = row["dataset"]
            sample = int(row["sample_index"])
            method = row["method"]
            if method in grouped[(task, sample)]:
                raise ValueError(f"duplicate {task}/{sample}/{method}")
            grouped[(task, sample)][method] = row
            task_family[task] = row["family"]
    methods = sorted({method for rows in grouped.values() for method in rows})
    if DENSE_METHOD not in methods or SAFE_METHOD not in methods:
        raise ValueError("Dense or safe compressed method missing")
    if any(set(rows) != set(methods) for rows in grouped.values()):
        raise ValueError("incomplete method coverage")
    keys = sorted(grouped)
    enhanced = (
        load_enhanced_table(enhanced_path)
        if enhanced_path is not None
        else None
    )
    if enhanced is not None:
        expected = {
            (task_name, sample_index, method)
            for task_name, sample_index in keys
            for method in methods
        }
        if set(enhanced) != expected:
            raise ValueError("enhanced feature coverage mismatch")
    task = np.asarray([key[0] for key in keys], dtype=object)
    sample = np.asarray([key[1] for key in keys], dtype=np.int64)
    family = np.asarray([task_family[key[0]] for key in keys], dtype=object)
    task_indices = {
        name: np.flatnonzero(task == name)
        for name in sorted({str(value) for value in task})
    }
    family_indices = {
        name: np.flatnonzero(family == name)
        for name in sorted({str(value) for value in family})
    }

    profiles: dict[str, np.ndarray] = {}
    enhanced_profiles: dict[str, np.ndarray] = {}
    for task_name, indices in task_indices.items():
        dense = [grouped[keys[index]][DENSE_METHOD] for index in indices]
        values = np.asarray(
            [[f(row, field) for field in PROFILE_FEATURES] for row in dense],
            dtype=np.float64,
        )
        parts = (
            values.mean(axis=0),
            values.std(axis=0),
            np.quantile(values, 0.25, axis=0),
            np.quantile(values, 0.50, axis=0),
            np.quantile(values, 0.75, axis=0),
        )
        profiles[task_name] = np.concatenate(parts)
        if enhanced is not None:
            enhanced_values = np.asarray(
                [
                    [
                        f(
                            enhanced[
                                (
                                    task_name,
                                    int(sample[index]),
                                    DENSE_METHOD,
                                )
                            ],
                            field,
                        )
                        for field in ENHANCED_PROFILE_FEATURES
                    ]
                    for index in indices
                ],
                dtype=np.float64,
            )
            enhanced_profiles[task_name] = np.concatenate(
                [
                    enhanced_values.mean(axis=0),
                    enhanced_values.std(axis=0),
                ]
            )

    method_data: dict[str, dict[str, np.ndarray]] = {}
    dense_rows = [grouped[key][DENSE_METHOD] for key in keys]
    dense_candidate = np.asarray(
        [[f(row, field) for field in CANDIDATE_FEATURES] for row in dense_rows],
        dtype=np.float64,
    )
    query_matrix = np.asarray(
        [[f(row, field) for field in QUERY_FEATURES] for row in dense_rows],
        dtype=np.float64,
    )
    profile_matrix = np.asarray(
        [profiles[str(task_name)] for task_name in task],
        dtype=np.float64,
    )
    enhanced_profile_matrix = (
        np.asarray(
            [enhanced_profiles[str(task_name)] for task_name in task],
            dtype=np.float64,
        )
        if enhanced is not None
        else np.empty((len(keys), 0), dtype=np.float64)
    )
    for method in methods:
        rows = [grouped[key][method] for key in keys]
        candidate = np.asarray(
            [[f(row, field) for field in CANDIDATE_FEATURES] for row in rows],
            dtype=np.float64,
        )
        relative = candidate - dense_candidate
        components = [query_matrix, candidate, relative, profile_matrix]
        if enhanced is not None:
            enhanced_candidate = np.asarray(
                [
                    [
                        f(
                            enhanced[
                                (
                                    str(task[index]),
                                    int(sample[index]),
                                    method,
                                )
                            ],
                            field,
                        )
                        for field in ENHANCED_MODEL_FEATURES
                    ]
                    for index in range(len(keys))
                ],
                dtype=np.float64,
            )
            enhanced_dense = np.asarray(
                [
                    [
                        f(
                            enhanced[
                                (
                                    str(task[index]),
                                    int(sample[index]),
                                    DENSE_METHOD,
                                )
                            ],
                            field,
                        )
                        for field in ENHANCED_MODEL_FEATURES
                    ]
                    for index in range(len(keys))
                ],
                dtype=np.float64,
            )
            components.extend(
                [
                    enhanced_candidate,
                    enhanced_candidate - enhanced_dense,
                    enhanced_profile_matrix,
                ]
            )
        base = np.concatenate(components, axis=1)
        target = np.asarray(
            [f(row, "dense_delta_score") for row in rows],
            dtype=np.float64,
        )
        compression = np.asarray(
            [f(row, "chunk_compression") for row in rows],
            dtype=np.float64,
        )
        method_data[method] = {
            "x": base,
            "y": target,
            "compression": compression,
        }
    if np.any(np.abs(method_data[SAFE_METHOD]["y"]) > EPS):
        raise ValueError("safe compressed method is not Dense-equivalent")
    if np.any(method_data[SAFE_METHOD]["compression"] <= 0.0):
        raise ValueError("safe compressed method has non-positive compression")
    if enhanced is None:
        raise ValueError("parc v2 requires enhanced structural features")
    dense_enhanced_rows = [
        enhanced[(str(task[index]), int(sample[index]), DENSE_METHOD)]
        for index in range(len(keys))
    ]
    query_cjk_ratio = np.asarray(
        [f(row, "question_cjk_ratio") for row in dense_enhanced_rows],
        dtype=np.float64,
    )
    dense_score_top1 = np.asarray(
        [f(row, "dense_score_top1") for row in dense_enhanced_rows],
        dtype=np.float64,
    )
    dense_score_top2_margin = np.asarray(
        [f(row, "dense_score_top2_margin") for row in dense_enhanced_rows],
        dtype=np.float64,
    )
    dense_normalized_margin = dense_score_top2_margin / np.maximum(
        np.abs(dense_score_top1),
        EPS,
    )
    task_structural_profiles = {
        task_name: {
            "question_cjk_ratio_mean": float(
                query_cjk_ratio[indices].mean()
            ),
            "dense_normalized_top2_margin_mean": float(
                dense_normalized_margin[indices].mean()
            ),
        }
        for task_name, indices in task_indices.items()
    }
    return {
        "keys": keys,
        "task": task,
        "sample": sample,
        "family": family,
        "methods": methods,
        "task_indices": task_indices,
        "family_indices": family_indices,
        "method_data": method_data,
        "feature_count": method_data[SAFE_METHOD]["x"].shape[1],
        "quadratic_feature_count": (
            len(QUERY_FEATURES)
            + 2 * len(CANDIDATE_FEATURES)
            + 5 * len(PROFILE_FEATURES)
        ),
        "enhanced_features_used": enhanced is not None,
        "query_cjk_ratio": query_cjk_ratio,
        "dense_normalized_margin": dense_normalized_margin,
        "task_structural_profiles": task_structural_profiles,
    }


def weighted_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
    quadratic_feature_count: int,
) -> np.ndarray:
    means = np.average(x_train, axis=0, weights=weights)
    centered = x_train - means
    scales = np.sqrt(np.average(centered * centered, axis=0, weights=weights))
    scales[scales <= EPS] = 1.0
    train = centered / scales
    test = (x_test - means) / scales
    # A modest nonlinear map captures score-margin/profile regimes while
    # retaining deterministic closed-form fitting.
    quadratic_feature_count = min(quadratic_feature_count, train.shape[1])
    train = np.concatenate(
        [train, train[:, :quadratic_feature_count] ** 2],
        axis=1,
    )
    test = np.concatenate(
        [test, test[:, :quadratic_feature_count] ** 2],
        axis=1,
    )
    train = np.concatenate(
        [np.ones((len(train), 1), dtype=np.float64), train],
        axis=1,
    )
    test = np.concatenate(
        [np.ones((len(test), 1), dtype=np.float64), test],
        axis=1,
    )
    sqrt_w = np.sqrt(weights)
    wx = train * sqrt_w[:, None]
    wy = y_train * sqrt_w
    penalty = np.eye(train.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    lhs = wx.T @ wx + penalty
    rhs = wx.T @ wy
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(lhs) @ rhs
    return test @ beta


def fold_weights(
    table: Mapping[str, Any],
    train_indices: np.ndarray,
    scheme: str,
) -> np.ndarray:
    task = table["task"]
    family = table["family"]
    weights = np.zeros(len(train_indices), dtype=np.float64)
    train_tasks = sorted({str(task[index]) for index in train_indices})
    task_counts = Counter(str(task[index]) for index in train_indices)
    if scheme == "query":
        weights.fill(1.0)
    elif scheme == "task":
        for local, index in enumerate(train_indices):
            weights[local] = 1.0 / task_counts[str(task[index])]
    elif scheme == "family":
        tasks_per_family = Counter(
            str(family[index])
            for name in train_tasks
            for index in [train_indices[np.flatnonzero(task[train_indices] == name)[0]]]
        )
        for local, index in enumerate(train_indices):
            weights[local] = 1.0 / (
                task_counts[str(task[index])]
                * tasks_per_family[str(family[index])]
            )
    else:
        raise ValueError(scheme)
    weights *= len(weights) / weights.sum()
    return weights


def target_values(y: np.ndarray, target: str) -> np.ndarray:
    if target.startswith("clip"):
        bound = float(target.removeprefix("clip"))
        return np.clip(y, -bound, bound)
    if target == "sign":
        return np.sign(y)
    if target.startswith("utility"):
        loss_weight = float(target.removeprefix("utility"))
        return (y > EPS).astype(np.float64) - loss_weight * (
            y < -EPS
        ).astype(np.float64)
    if target.startswith("softsign"):
        scale = float(target.removeprefix("softsign"))
        return np.tanh(y / scale)
    if target == "rank":
        raise AssertionError("rank target is assembled per query elsewhere")
    raise ValueError(target)


def crossfit_predictions(
    table: Mapping[str, Any],
    *,
    heldout_unit: str,
    alpha: float,
    weight_scheme: str,
    target: str,
) -> dict[str, np.ndarray]:
    unit_values = table[heldout_unit]
    all_indices = np.arange(len(unit_values), dtype=np.int64)
    methods = [
        method
        for method in table["methods"]
        if method not in {DENSE_METHOD, SAFE_METHOD}
    ]
    output = {
        method: np.empty(len(unit_values), dtype=np.float64)
        for method in methods
    }
    for held in sorted({str(value) for value in unit_values}):
        test_indices = np.flatnonzero(unit_values == held)
        train_indices = all_indices[unit_values != held]
        weights = fold_weights(table, train_indices, weight_scheme)
        for method in methods:
            data = table["method_data"][method]
            y = target_values(data["y"][train_indices], target)
            output[method][test_indices] = weighted_ridge(
                data["x"][train_indices],
                y,
                weights,
                data["x"][test_indices],
                alpha,
                int(table["quadratic_feature_count"]),
            )
    return output


def evaluate_policy(
    table: Mapping[str, Any],
    predictions: Mapping[str, np.ndarray],
    *,
    activation_fraction: float,
    routing: str,
    portfolio: str,
    task_prior_weight: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    allowed = PORTFOLIOS[portfolio]
    methods = sorted(
        method
        for method in predictions
        if allowed is None or method in allowed
    )
    if not methods:
        raise ValueError(f"{portfolio}: empty portfolio")
    safe = table["method_data"][SAFE_METHOD]
    task_rows: list[dict[str, Any]] = []
    for task_name, indices in table["task_indices"].items():
        score_matrix = np.stack(
            [predictions[method][indices] for method in methods],
            axis=1,
        )
        task_prior = np.quantile(score_matrix, 0.90, axis=0)
        adjusted = score_matrix + task_prior_weight * task_prior[None, :]
        if routing == "query":
            chosen_method_index = np.argmax(adjusted, axis=1)
            confidence = adjusted[
                np.arange(len(indices)), chosen_method_index
            ]
        elif routing == "task":
            task_method = int(np.argmax(task_prior))
            chosen_method_index = np.full(
                len(indices), task_method, dtype=np.int64
            )
            confidence = score_matrix[:, task_method]
        else:
            raise ValueError(routing)
        active_count = max(
            1,
            min(
                len(indices),
                int(math.ceil(len(indices) * activation_fraction)),
            ),
        )
        active_local = np.argsort(-confidence, kind="stable")[:active_count]
        active = np.zeros(len(indices), dtype=bool)
        active[active_local] = True
        chosen_methods = np.asarray(
            [methods[int(index)] for index in chosen_method_index],
            dtype=object,
        )
        profile = table["task_structural_profiles"][task_name]
        cjk_route = choose_cjk_structural_route(profile)
        if cjk_route is not None:
            # Conservative structural override: route only CJK queries and
            # return minority non-CJK queries to the verified-safe default.
            cjk_queries = (
                table["query_cjk_ratio"][indices]
                > CJK_QUERY_RATIO_THRESHOLD
            )
            active[:] = cjk_queries
            chosen_methods[cjk_queries] = cjk_route
        deltas = np.zeros(len(indices), dtype=np.float64)
        compression = safe["compression"][indices].copy()
        selected = Counter({SAFE_METHOD: int((~active).sum())})
        for local in np.flatnonzero(active):
            method = str(chosen_methods[local])
            global_index = int(indices[local])
            deltas[local] = table["method_data"][method]["y"][global_index]
            compression[local] = table["method_data"][method]["compression"][
                global_index
            ]
            selected[method] += 1
        task_rows.append(
            {
                "family": str(table["family"][indices[0]]),
                "dataset": task_name,
                "rows": len(indices),
                "delta_f1": 100.0 * float(deltas.mean()),
                "chunk_compression": float(compression.mean()),
                "active_fraction": float(active.mean()),
                "structural_profile": profile,
                "cjk_structural_route": cjk_route,
                "selected_methods": dict(sorted(selected.items())),
            }
        )
    deltas = [row["delta_f1"] for row in task_rows]
    summary = {
        "tasks": len(task_rows),
        "wins": sum(value > EPS for value in deltas),
        "ties": sum(abs(value) <= EPS for value in deltas),
        "losses": sum(value < -EPS for value in deltas),
        "minimum_task_delta_f1": min(deltas),
        "task_macro_delta_f1": statistics.fmean(deltas),
        "minimum_task_chunk_compression": min(
            row["chunk_compression"] for row in task_rows
        ),
    }
    return summary, task_rows


def objective(
    lofo: Mapping[str, Any],
    loto: Mapping[str, Any],
) -> tuple[Any, ...]:
    return (
        min(lofo["wins"], loto["wins"]),
        lofo["wins"],
        lofo["minimum_task_delta_f1"],
        loto["wins"],
        loto["minimum_task_delta_f1"],
        lofo["task_macro_delta_f1"],
        loto["task_macro_delta_f1"],
        min(
            lofo["minimum_task_chunk_compression"],
            loto["minimum_task_chunk_compression"],
        ),
    )


def run(
    path: Path,
    output: Path,
    enhanced_path: Path | None = None,
) -> dict[str, Any]:
    if enhanced_path is None:
        raise ValueError("--enhanced-features is required for parc v2")
    table = load_table(path, enhanced_path)
    # Convenience mapping used by family-balanced weights.
    table["task_family"] = {
        task: str(table["family"][indices[0]])
        for task, indices in table["task_indices"].items()
    }
    grid = {
        "alpha": (
            [100.0, 1000.0]
            if enhanced_path is not None
            else [1.0, 10.0, 100.0, 1000.0]
        ),
        "weight_scheme": (
            ["family"]
            if enhanced_path is not None
            else ["task", "family"]
        ),
        "target": (
            ["clip0.05", "clip0.1", "softsign0.05"]
            if enhanced_path is not None
            else [
                "clip0.05",
                "clip0.1",
                "clip0.25",
                "sign",
                "utility1.0",
                "utility1.5",
                "utility2.0",
                "softsign0.02",
                "softsign0.05",
                "softsign0.1",
            ]
        ),
        "activation_fraction": [
            0.002,
            0.005,
            0.01,
            0.02,
            0.05,
            0.10,
            0.20,
            0.40,
            1.0,
        ],
        "routing": ["query", "task"],
        "portfolio": list(PORTFOLIOS),
        "task_prior_weight": [0.0, 0.25, 0.5, 1.0],
    }
    best: dict[str, Any] | None = None
    best_key: tuple[Any, ...] | None = None
    model_count = 0
    policy_count = 0
    for alpha, weight_scheme, target in itertools.product(
        grid["alpha"], grid["weight_scheme"], grid["target"]
    ):
        loto_predictions = crossfit_predictions(
            table,
            heldout_unit="task",
            alpha=alpha,
            weight_scheme=weight_scheme,
            target=target,
        )
        lofo_predictions = crossfit_predictions(
            table,
            heldout_unit="family",
            alpha=alpha,
            weight_scheme=weight_scheme,
            target=target,
        )
        model_count += 1
        for (
            activation_fraction,
            routing,
            portfolio,
            task_prior_weight,
        ) in itertools.product(
            grid["activation_fraction"],
            grid["routing"],
            grid["portfolio"],
            grid["task_prior_weight"],
        ):
            loto_summary, loto_tasks = evaluate_policy(
                table,
                loto_predictions,
                activation_fraction=activation_fraction,
                routing=routing,
                portfolio=portfolio,
                task_prior_weight=task_prior_weight,
            )
            lofo_summary, lofo_tasks = evaluate_policy(
                table,
                lofo_predictions,
                activation_fraction=activation_fraction,
                routing=routing,
                portfolio=portfolio,
                task_prior_weight=task_prior_weight,
            )
            policy_count += 1
            key = objective(lofo_summary, loto_summary)
            if best_key is None or key > best_key:
                best_key = key
                best = {
                    "policy": {
                        "alpha": alpha,
                        "weight_scheme": weight_scheme,
                        "target": target,
                        "activation_fraction": activation_fraction,
                        "routing": routing,
                        "portfolio": portfolio,
                        "task_prior_weight": task_prior_weight,
                        "safe_default": SAFE_METHOD,
                        "structural_router": {
                            "mode": CJK_ROUTER_MODE,
                            "query_cjk_ratio_threshold": (
                                CJK_QUERY_RATIO_THRESHOLD
                            ),
                            "task_mean_cjk_ratio_threshold": (
                                CJK_TASK_MEAN_RATIO_THRESHOLD
                            ),
                            "normalized_dense_margin_threshold": (
                                CJK_NORMALIZED_MARGIN_THRESHOLD
                            ),
                            "low_margin_method": DOC_ORDER_METHOD,
                            "high_margin_method": LANG_MARGIN_METHOD,
                        },
                    },
                    "leave_one_task_out": loto_summary,
                    "leave_one_family_out": lofo_summary,
                    "loto_tasks": loto_tasks,
                    "lofo_tasks": lofo_tasks,
                }
                print(
                    json.dumps(
                        {
                            "policy": best["policy"],
                            "loto": loto_summary,
                            "lofo": lofo_summary,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
    assert best is not None
    result = {
        "status": "COMPLETE_OPENED_DEVELOPMENT_EXPLORATION_V2",
        "safe_default_verified_dense_equivalent": True,
        "router_identity_free_by_construction": True,
        "router_label_free_by_construction": True,
        "feature_count": table["feature_count"],
        "model_configurations": model_count,
        "policy_configurations": policy_count,
        "grid": grid,
        **best,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-features", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--enhanced-features", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    result = run(
        args.safe_features,
        args.output,
        args.enhanced_features,
    )
    print(json.dumps(result["policy"], ensure_ascii=False, sort_keys=True))
    print(
        json.dumps(
            {
                "loto": result["leave_one_task_out"],
                "lofo": result["leave_one_family_out"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
