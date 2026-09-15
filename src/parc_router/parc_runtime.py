#!/usr/bin/env python3
"""Prediction-free deployment runtime for the frozen PARC selector.

The runtime receives one complete opaque task group containing only questions,
chunk stores, and the terminally verified eleven-method retrieval inventory.
It reconstructs the frozen 221-dimensional raw and 431-dimensional relative
views, applies the deterministic dual-view sklearn bundle, and routes every
inactive query to the predeclared maximum-compression Dense-order-equivalent
anchor (or the safe compressed-Dense fallback in the 399 information-theoretic
impossibility cases).

No benchmark, family, task, generator, model-identity, answer, prediction, F1,
quality-score, or partial-confirmation field is accepted by the public parser.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

import parc_compat_runtime as legacy
import parc_runtime_v3 as deployment_v3


PROTOCOL = "Successor-v9-dualview-selector-runtime-20260720-v2"
FROZEN_MODEL_PROTOCOL = (
    "Successor-v9-dualview-final-model-training-freeze-20260720-v2"
)
EXPECTED_MODEL_SHA256 = (
    "612f952f3046ebb13d4c087f41f9666a9d33cd8e15f050f7b262cc558a15e1ab"
)
EXPECTED_MANIFEST_SHA256 = (
    "5eff9281c28e703a0b96ad9abb641b8404093605c9bdb73fbc8b4e36b3029b0f"
)
MODEL_MANIFEST_NAME = "parc_frozen_selector_manifest.json"

DENSE_METHOD = "FullPool_Dense_k8"
SAFE_METHOD = "SmartMemory_CompressedDense_R45_k8"
QUALITY_CANDIDATES = (
    "SmartMemory_CompressedDense_DocOrder_R45_k8",
    "SmartMemory_CompressedDense_RRFOrder_R45_k8",
    "SmartMemory_CompressedDense_RerankOrder_R45_k8",
    "SmartMemory_Hybrid_BM25_D7A1_R45_k8",
    "SmartMemory_Hybrid_RRFXE_LangMargin_D7A1_EN25_CJK20_R35_k8",
    "SmartMemory_Hybrid_RRFXE_LangMargin_D7A1_EN25_CJK20_R45_k8",
    "SmartMemory_Hybrid_RRFXE_LangMargin_D7A1_EN25_CJK20_R55_k8",
    "SmartMemory_Hybrid_RRF_D6A2_R45_k8",
    "SmartMemory_Hybrid_Rerank_D7A1_M10_R45_k8",
)
ANCHOR_CANDIDATES = (
    "SmartMemory_CompressedDense_DocOrder_R45_k8",
    "SmartMemory_CompressedDense_R45_k8",
    "SmartMemory_CompressedDense_RRFOrder_R45_k8",
    "SmartMemory_CompressedDense_RerankOrder_R45_k8",
    "SmartMemory_Hybrid_BM25_D7A1_R45_k8",
    "SmartMemory_Hybrid_RRFXE_LangMargin_D7A1_EN25_CJK20_R35_k8",
    "SmartMemory_Hybrid_RRFXE_LangMargin_D7A1_EN25_CJK20_R45_k8",
    "SmartMemory_Hybrid_RRFXE_LangMargin_D7A1_EN25_CJK20_R55_k8",
    "SmartMemory_Hybrid_RRF_D6A2_R45_k8",
    "SmartMemory_Hybrid_Rerank_D7A1_M10_R45_k8",
)

EXPECTED_POLICY = {
    "maximum_negative_probability": 0.4,
    "minimum_positive_probability": 0.2,
    "negative_penalty": 0.5,
    "regression_weight": 2.0,
    "route_margin": 0.02,
    "score_threshold": 0.2,
}
RAW_WEIGHT = 0.5
RELATIVE_WEIGHT = 0.5
DISAGREEMENT_PENALTY = 0.0
MINIMUM_ACTIVE_QUERY_COMPRESSION = 0.40
EXPECTED_BASE_FEATURES = 212
EXPECTED_RAW_FEATURES = 221
EXPECTED_RELATIVE_FEATURES = 431
EXPECTED_CANDIDATES = 9

EXPECTED_FORMAL_QUERY_ORDER_SHA256 = (
    "11864eac123980f69a805edffaec6a1d8506b066b2fc372adf9bce7e66552d7e"
)
EXPECTED_FORMAL_RAW_TENSOR_SHA256 = (
    "cb23ccd14e808458df6e9b324efd27f75b11b32346da4b867a3ae5c34824ee7a"
)
EXPECTED_FORMAL_RELATIVE_TENSOR_SHA256 = (
    "0e1d9ed3eb8296a7f975a92f9f347b8c0ae1b259f3728ac9301b6b6bf3e950cc"
)
EXPECTED_FORMAL_ANCHOR_DECISIONS_SHA256 = (
    "e67645dd875d7c28ef79a78d8ea290ca2061c345caccab56bb2dc0b9742277e5"
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    canonical = np.ascontiguousarray(value)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_array(value: np.ndarray, label: str) -> np.ndarray:
    observed = np.asarray(value)
    if not np.all(np.isfinite(observed)):
        raise ValueError(f"{label}: non-finite values")
    return observed


def relative_feature_indices(schema: Sequence[str]) -> list[int]:
    """Reconstruct the frozen set of 70 query-internal relative fields."""
    selected: list[int] = []
    for index, name in enumerate(schema):
        if name.startswith("candidate_minus_dense."):
            selected.append(index)
            continue
        if not name.startswith("enhanced_candidate_minus_dense."):
            continue
        suffix = name.split(".", 1)[1]
        if suffix.startswith("question_") or suffix.startswith("dense_score_"):
            continue
        selected.append(index)
    _require(len(selected) == 70, "relative feature field count mismatch")
    return selected


def append_candidate_identity(base_tensor: np.ndarray) -> np.ndarray:
    """Append the frozen nine-position candidate identity one-hot."""
    _require(base_tensor.ndim == 3, "base tensor must be rank three")
    queries, candidates, dimensions = base_tensor.shape
    _require(candidates == EXPECTED_CANDIDATES, "candidate count mismatch")
    _require(dimensions == EXPECTED_BASE_FEATURES, "base feature count mismatch")
    identity = np.broadcast_to(
        np.eye(candidates, dtype=np.float32)[None, :, :],
        (queries, candidates, candidates),
    )
    raw = np.concatenate(
        (np.asarray(base_tensor, dtype=np.float32), identity),
        axis=2,
    )
    _require(
        raw.shape == (queries, EXPECTED_CANDIDATES, EXPECTED_RAW_FEATURES),
        "raw feature shape mismatch",
    )
    return _finite_array(raw, "raw features")


def augment_relative_features(
    raw_features: np.ndarray,
    base_schema: Sequence[str],
) -> np.ndarray:
    """Add centered, robust-IQR, and percentile views exactly as frozen."""
    _require(raw_features.ndim == 3, "raw tensor must be rank three")
    selected = relative_feature_indices(base_schema)
    values = np.asarray(raw_features[:, :, selected], dtype=np.float64)
    median = np.median(values, axis=1, keepdims=True)
    centered = values - median
    q25 = np.quantile(values, 0.25, axis=1, keepdims=True)
    q75 = np.quantile(values, 0.75, axis=1, keepdims=True)
    scale = np.maximum(q75 - q25, 1e-6)
    robust = np.clip(centered / scale, -8.0, 8.0)

    queries, candidates, dimensions = values.shape
    percentile = np.empty_like(values)
    denominator = float(max(candidates - 1, 1))
    for dimension in range(dimensions):
        column = values[:, :, dimension]
        less = np.sum(column[:, :, None] > column[:, None, :], axis=2)
        equal = np.sum(column[:, :, None] == column[:, None, :], axis=2)
        percentile[:, :, dimension] = (
            less + 0.5 * np.maximum(equal - 1, 0)
        ) / denominator

    relative = np.concatenate(
        (
            np.asarray(raw_features, dtype=np.float32),
            centered.astype(np.float32),
            robust.astype(np.float32),
            percentile.astype(np.float32),
        ),
        axis=2,
    )
    _require(
        relative.shape
        == (queries, EXPECTED_CANDIDATES, EXPECTED_RELATIVE_FEATURES),
        "relative feature shape mismatch",
    )
    return _finite_array(relative, "relative features")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


@functools.lru_cache(maxsize=4)
def _load_frozen_cached(
    model_path_text: str,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    model_path = Path(model_path_text)
    model_sha256 = sha256_file(model_path)
    _require(
        model_sha256 == EXPECTED_MODEL_SHA256,
        f"frozen model SHA256 mismatch: {model_sha256}",
    )
    manifest_path = model_path.parent / MODEL_MANIFEST_NAME
    manifest_sha256 = sha256_file(manifest_path)
    _require(
        manifest_sha256 == EXPECTED_MANIFEST_SHA256,
        f"frozen manifest SHA256 mismatch: {manifest_sha256}",
    )
    manifest = _load_json(manifest_path)
    _require(
        manifest.get("protocol") == FROZEN_MODEL_PROTOCOL,
        "frozen model protocol mismatch",
    )
    _require(
        manifest.get("status") == "FROZEN_FINAL_MODEL_NOT_FORMALLY_CONFIRMED",
        "frozen manifest status mismatch",
    )
    _require(
        manifest.get("formal_predictions_answers_f1_or_scores_used") is False,
        "frozen manifest used forbidden formal quality inputs",
    )
    _require(
        manifest.get("formal_generation_started") is False,
        "formal generation preceded model freeze",
    )
    method = manifest.get("method")
    _require(isinstance(method, dict), "frozen method block missing")
    _require(
        tuple(method.get("candidate_methods", ())) == QUALITY_CANDIDATES,
        "quality candidate order mismatch",
    )
    _require(
        tuple(method.get("anchor_methods", ())) == ANCHOR_CANDIDATES,
        "anchor candidate order mismatch",
    )
    _require(
        method.get("raw_weight") == RAW_WEIGHT
        and method.get("relative_weight") == RELATIVE_WEIGHT
        and method.get("disagreement_penalty") == DISAGREEMENT_PENALTY,
        "dual-view combination mismatch",
    )
    _require(
        method.get("minimum_active_query_compression")
        == MINIMUM_ACTIVE_QUERY_COMPRESSION,
        "minimum active compression mismatch",
    )
    selected_policy = method.get("selected_final_policy")
    _require(isinstance(selected_policy, dict), "selected policy block missing")
    _require(
        selected_policy.get("policy_index") == 116,
        "selected policy index mismatch",
    )
    _require(
        selected_policy.get("policy") == EXPECTED_POLICY,
        "selected policy mismatch",
    )

    contract = manifest.get("feature_contract")
    _require(isinstance(contract, dict), "feature contract missing")
    for key, expected in (
        ("base_feature_count", EXPECTED_BASE_FEATURES),
        ("candidate_identity_feature_count", EXPECTED_CANDIDATES),
        ("raw_feature_count", EXPECTED_RAW_FEATURES),
        ("relative_feature_count", EXPECTED_RELATIVE_FEATURES),
    ):
        _require(contract.get(key) == expected, f"{key} mismatch")
    base_schema = contract.get("base_schema")
    _require(
        isinstance(base_schema, list)
        and base_schema == deployment_v3.feature_schema(),
        "base feature schema mismatch",
    )
    formal = manifest.get("verified_formal_feature_anchor_parity")
    _require(isinstance(formal, dict), "formal parity block missing")
    for key, expected in (
        ("formal_query_order_sha256", EXPECTED_FORMAL_QUERY_ORDER_SHA256),
        ("formal_raw_feature_tensor_sha256", EXPECTED_FORMAL_RAW_TENSOR_SHA256),
        (
            "formal_relative_feature_tensor_sha256",
            EXPECTED_FORMAL_RELATIVE_TENSOR_SHA256,
        ),
        ("anchor_decisions_sha256", EXPECTED_FORMAL_ANCHOR_DECISIONS_SHA256),
    ):
        _require(formal.get(key) == expected, f"formal {key} mismatch")
    _require(
        manifest.get("model_artifact", {}).get("sha256") == model_sha256,
        "manifest/model artifact digest mismatch",
    )

    bundle = joblib.load(model_path)
    _require(isinstance(bundle, dict), "joblib bundle is not an object")
    _require(
        bundle.get("protocol") == FROZEN_MODEL_PROTOCOL,
        "joblib protocol mismatch",
    )
    _require(set(bundle) == {"protocol", "raw", "relative"}, "bundle keys mismatch")
    for view, expected_features in (
        ("raw", EXPECTED_RAW_FEATURES),
        ("relative", EXPECTED_RELATIVE_FEATURES),
    ):
        artifact = bundle.get(view)
        _require(isinstance(artifact, dict), f"{view} model missing")
        _require(
            int(artifact.get("features", -1)) == expected_features,
            f"{view} feature count mismatch",
        )
        _require(
            int(artifact.get("candidates", -1)) == EXPECTED_CANDIDATES,
            f"{view} candidate count mismatch",
        )
        _require(
            list(artifact["classifier"].classes_) == [0, 1, 2],
            f"{view} classifier classes mismatch",
        )
        _require(
            artifact["classifier"].n_jobs == 1,
            f"{view} classifier is not deterministic single-worker",
        )
    return bundle, manifest, model_sha256, manifest_sha256


def load_frozen_model(
    model_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    resolved = str(model_path.resolve())
    return _load_frozen_cached(resolved)


def _predict_view(
    artifact: Mapping[str, Any],
    features: np.ndarray,
) -> dict[str, np.ndarray]:
    queries, candidates, dimensions = features.shape
    matrix = features.reshape(-1, dimensions)
    regression = artifact["regressor"].predict(matrix).reshape(
        queries,
        candidates,
    )
    raw_probabilities = artifact["classifier"].predict_proba(matrix)
    probabilities = np.zeros((queries * candidates, 3), dtype=np.float64)
    for column, label in enumerate(artifact["classifier"].classes_.tolist()):
        probabilities[:, int(label)] = raw_probabilities[:, column]
    probabilities = probabilities.reshape(queries, candidates, 3)
    output = {
        "regression": np.asarray(regression, dtype=np.float64),
        "p_positive": probabilities[:, :, 2],
        "p_negative": probabilities[:, :, 0],
    }
    for key, value in output.items():
        _finite_array(value, f"{key} prediction")
    return output


def _mean_predictions(
    raw: Mapping[str, np.ndarray],
    relative: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for key in ("regression", "p_positive", "p_negative"):
        _require(raw[key].shape == relative[key].shape, f"{key} shape mismatch")
        value = RAW_WEIGHT * raw[key] + RELATIVE_WEIGHT * relative[key]
        if key.startswith("p_"):
            value = np.clip(value, 0.0, 1.0)
        output[key] = _finite_array(
            np.asarray(value, dtype=np.float64),
            f"dual-view {key}",
        )
    return output


def _compression_matrix(
    queries: Sequence[legacy.QueryObservation],
) -> np.ndarray:
    matrix = np.asarray(
        [
            [
                1.0
                - len(query.methods[method].kept_indices)
                / query.methods[method].original_chunks
                for method in QUALITY_CANDIDATES
            ]
            for query in queries
        ],
        dtype=np.float64,
    )
    _finite_array(matrix, "candidate compression")
    _require(np.all(matrix > 0.0), "non-positive quality-candidate compression")
    return matrix


def apply_frozen_policy(
    predictions: Mapping[str, np.ndarray],
    compression: np.ndarray,
) -> dict[str, np.ndarray]:
    """Apply the preregistered policy without changing its numerical rules."""
    required = {"regression", "p_positive", "p_negative"}
    _require(set(predictions) == required, "prediction field mismatch")
    regression = _finite_array(
        np.asarray(predictions["regression"], dtype=np.float64),
        "policy regression",
    )
    positive = _finite_array(
        np.asarray(predictions["p_positive"], dtype=np.float64),
        "policy positive probability",
    )
    negative = _finite_array(
        np.asarray(predictions["p_negative"], dtype=np.float64),
        "policy negative probability",
    )
    compression = _finite_array(
        np.asarray(compression, dtype=np.float64),
        "policy compression",
    )
    _require(
        regression.shape == positive.shape == negative.shape == compression.shape,
        "policy tensor shape mismatch",
    )
    _require(
        regression.ndim == 2
        and regression.shape[1] == EXPECTED_CANDIDATES,
        "policy tensors must be query-by-candidate matrices",
    )

    score = (
        EXPECTED_POLICY["regression_weight"] * regression
        + positive
        - EXPECTED_POLICY["negative_penalty"] * negative
    )
    eligible = compression >= MINIMUM_ACTIVE_QUERY_COMPRESSION
    has_eligible = np.any(eligible, axis=1)
    masked_score = np.where(eligible, score, -np.inf)
    order = np.argsort(-masked_score, axis=1, kind="stable")
    rows = np.arange(regression.shape[0], dtype=np.int64)
    best = order[:, 0]
    second = order[:, 1]
    best_score = masked_score[rows, best]
    second_score = masked_score[rows, second]
    margin = np.full(regression.shape[0], -np.inf, dtype=np.float64)
    margin[has_eligible] = (
        best_score[has_eligible] - second_score[has_eligible]
    )
    active = (
        has_eligible
        & (best_score > EXPECTED_POLICY["score_threshold"])
        & (
            margin > EXPECTED_POLICY["route_margin"]
        )
        & (
            negative[rows, best]
            <= EXPECTED_POLICY["maximum_negative_probability"]
        )
        & (
            positive[rows, best]
            >= EXPECTED_POLICY["minimum_positive_probability"]
        )
    )

    # A formal query can have zero eligible candidates when integer rounding
    # in a very small chunk pool makes every nominal R45/R35/R55 candidate
    # compress by less than 40%.  Such a query is deterministically inactive
    # and uses the separately frozen positive-compression anchor.  A query
    # with exactly one eligible candidate retains the development behavior:
    # the stable second position is -inf and the route margin is +inf.
    _require(
        not np.any(np.isnan(best_score))
        and not np.any(np.isposinf(best_score)),
        "best score contains invalid values",
    )
    _require(
        not np.any(np.isnan(second_score))
        and not np.any(np.isposinf(second_score)),
        "second score contains invalid values",
    )
    _require(
        not np.any(np.isnan(margin)),
        "route margin contains invalid values",
    )
    return {
        "score": score,
        "eligible": eligible,
        "masked_score": masked_score,
        "order": order,
        "best_candidate": best,
        "second_candidate": second,
        "best_score": best_score,
        "second_score": second_score,
        "margin": margin,
        "has_eligible": has_eligible,
        "active": active,
    }


def _select_anchor(
    query: legacy.QueryObservation,
) -> dict[str, Any]:
    dense_retrieved = query.methods[DENSE_METHOD].retrieved_indices
    eligible: list[tuple[float, int, str]] = []
    for index, method in enumerate(ANCHOR_CANDIDATES):
        observation = query.methods[method]
        compression = (
            1.0
            - len(observation.kept_indices) / observation.original_chunks
        )
        if observation.retrieved_indices == dense_retrieved and compression > 0.0:
            eligible.append((float(compression), -index, method))
    if eligible:
        compression, negative_index, method = max(eligible)
        return {
            "anchor_method": method,
            "anchor_candidate_index": -negative_index,
            "anchor_selection_type": "existing_dense_order_equivalent",
            "dense_order_equivalent": True,
            "exact_equivalence_impossible_under_strict_compression": False,
            "chunk_compression": float(compression),
        }

    dense_unique = tuple(dict.fromkeys(dense_retrieved))
    _require(
        len(dense_retrieved) == query.methods[DENSE_METHOD].original_chunks
        and len(dense_unique) == query.methods[DENSE_METHOD].original_chunks,
        f"{query.query_key}: missing exact anchor outside impossibility case",
    )
    safe = query.methods[SAFE_METHOD]
    compression = 1.0 - len(safe.kept_indices) / safe.original_chunks
    _require(compression > 0.0, f"{query.query_key}: safe fallback not compressed")
    return {
        "anchor_method": SAFE_METHOD,
        "anchor_candidate_index": ANCHOR_CANDIDATES.index(SAFE_METHOD),
        "anchor_selection_type": (
            "safe_default_due_to_dense_full_pool_impossibility"
        ),
        "dense_order_equivalent": False,
        "exact_equivalence_impossible_under_strict_compression": True,
        "chunk_compression": float(compression),
    }


def reconstruct_public_task(
    values: Sequence[Mapping[str, Any]],
    model_path: Path,
) -> dict[str, Any]:
    """Reconstruct all prediction-free deployment state for one opaque group."""
    bundle, manifest, model_sha256, manifest_sha256 = load_frozen_model(model_path)
    base_schema = [str(value) for value in manifest["feature_contract"]["base_schema"]]
    contract_model = {
        "dense_baseline": DENSE_METHOD,
        "safe_default": SAFE_METHOD,
        "candidate_methods": list(QUALITY_CANDIDATES),
        "feature_schema": base_schema,
    }
    queries = legacy.parse_public_task(values, contract_model)
    vectors, _profile = deployment_v3.construct_feature_vectors(
        queries,
        contract_model,
    )
    _require(
        tuple(vectors) == QUALITY_CANDIDATES,
        "runtime candidate vector order mismatch",
    )
    base = np.stack(
        [
            np.asarray(vectors[method], dtype=np.float32)
            for method in QUALITY_CANDIDATES
        ],
        axis=1,
    )
    _require(
        base.shape
        == (len(queries), EXPECTED_CANDIDATES, EXPECTED_BASE_FEATURES),
        "runtime base feature shape mismatch",
    )
    raw_features = append_candidate_identity(base)
    relative_features = augment_relative_features(raw_features, base_schema)
    raw_prediction = _predict_view(bundle["raw"], raw_features)
    relative_prediction = _predict_view(bundle["relative"], relative_features)
    predictions = _mean_predictions(raw_prediction, relative_prediction)
    compression = _compression_matrix(queries)
    anchors = [_select_anchor(query) for query in queries]
    policy_state = apply_frozen_policy(predictions, compression)
    score = policy_state["score"]
    best = policy_state["best_candidate"]
    second = policy_state["second_candidate"]
    active = policy_state["active"]

    decisions: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for index, query in enumerate(queries):
        if bool(active[index]):
            method = QUALITY_CANDIDATES[int(best[index])]
            selection_source = "frozen_dualview_quality_selector"
            selected_compression = float(compression[index, int(best[index])])
        else:
            anchor = anchors[index]
            method = str(anchor["anchor_method"])
            selection_source = str(anchor["anchor_selection_type"])
            selected_compression = float(anchor["chunk_compression"])
        _require(
            selected_compression > 0.0,
            f"{query.query_key}: selected route is not compressed",
        )
        counts[method] += 1
        source_counts[selection_source] += 1
        decisions.append(
            {
                "query_key": query.query_key,
                "selected_method": method,
                "selection_source": selection_source,
            }
        )

    return {
        "queries": queries,
        "base_features": base,
        "raw_features": raw_features,
        "relative_features": relative_features,
        "compression": compression,
        "anchors": anchors,
        "raw_predictions": raw_prediction,
        "relative_predictions": relative_prediction,
        "predictions": predictions,
        "score": score,
        "best_candidate": best,
        "second_candidate": second,
        "best_score": policy_state["best_score"],
        "second_score": policy_state["second_score"],
        "eligible": policy_state["eligible"],
        "has_eligible": policy_state["has_eligible"],
        "active": active,
        "decisions": decisions,
        "selected_method_counts": dict(sorted(counts.items())),
        "selection_source_counts": dict(sorted(source_counts.items())),
        "model_sha256": model_sha256,
        "manifest_sha256": manifest_sha256,
    }


def route_public_task(
    values: Sequence[Mapping[str, Any]],
    model_path: Path,
) -> dict[str, Any]:
    state = reconstruct_public_task(values, model_path)
    active = np.asarray(state["active"], dtype=bool)
    has_eligible = np.asarray(state["has_eligible"], dtype=bool)
    anchors = state["anchors"]
    anchor_counts = Counter(str(row["anchor_method"]) for row in anchors)
    anchor_source_counts = Counter(
        str(row["anchor_selection_type"]) for row in anchors
    )
    return {
        "status": "COMPLETE_RETRIEVAL_ONLY_ROUTING_V9",
        "protocol": PROTOCOL,
        "model_sha256": state["model_sha256"],
        "model_manifest_sha256": state["manifest_sha256"],
        "queries": len(state["queries"]),
        "generation_started_before_routing_complete": False,
        "forbidden_quality_inputs_read": False,
        "task_or_benchmark_identity_read": False,
        "model_or_generator_identity_read": False,
        "learned_group_profile_features_used": False,
        "source_family_used_at_inference": False,
        "partial_results_used": False,
        # The audited v2 routing container persists this legacy field verbatim.
        # R9 uses it only as a prediction-free per-group runtime attestation.
        "cjk_structural_route": {
            "legacy_attestation_slot": True,
            "protocol": PROTOCOL,
            "queries": len(state["queries"]),
            "base_feature_tensor_sha256": array_sha256(state["base_features"]),
            "raw_feature_tensor_sha256": array_sha256(state["raw_features"]),
            "relative_feature_tensor_sha256": array_sha256(
                state["relative_features"]
            ),
            "active_queries": int(np.sum(active)),
            "inactive_anchor_queries": int(np.sum(~active)),
            "quality_eligible_queries": int(np.sum(has_eligible)),
            "zero_eligible_quality_queries": int(np.sum(~has_eligible)),
            "anchor_method_counts": dict(sorted(anchor_counts.items())),
            "anchor_selection_type_counts": dict(
                sorted(anchor_source_counts.items())
            ),
            "candidate_order_sha256": canonical_sha256(
                list(QUALITY_CANDIDATES)
            ),
            "policy_sha256": canonical_sha256(EXPECTED_POLICY),
            "task_or_benchmark_identity_read": False,
            "model_or_generator_identity_read": False,
            "source_family_used_at_inference": False,
            "partial_results_used": False,
        },
        "selected_method_counts": state["selected_method_counts"],
        "decisions": state["decisions"],
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def self_test() -> None:
    base = np.zeros((2, EXPECTED_CANDIDATES, EXPECTED_BASE_FEATURES), dtype=np.float32)
    for candidate in range(EXPECTED_CANDIDATES):
        base[:, candidate, 25] = float(candidate)
    raw = append_candidate_identity(base)
    relative = augment_relative_features(raw, deployment_v3.feature_schema())
    assert raw.shape == (2, 9, 221)
    assert relative.shape == (2, 9, 431)
    assert np.array_equal(raw[0, :, -9:], np.eye(9, dtype=np.float32))
    assert math.isfinite(float(relative.sum()))
    assert canonical_sha256(EXPECTED_POLICY) == canonical_sha256(
        dict(reversed(list(EXPECTED_POLICY.items())))
    )
    print("PASS parc_runtime self-test")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--task-input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.model is None or args.task_input is None or args.output is None:
        raise SystemExit("--model, --task-input, and --output are required")
    values = json.loads(args.task_input.read_text(encoding="utf-8"))
    result = route_public_task(values, args.model)
    atomic_json(args.output, result)
    print(
        canonical_json(
            {
                "status": result["status"],
                "queries": result["queries"],
                "model_sha256": result["model_sha256"],
                "model_manifest_sha256": result["model_manifest_sha256"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
