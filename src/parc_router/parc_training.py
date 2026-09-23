"""Train PARC selectors from development-only retrieval features and labels.

The published paper artifact remains immutable. New fits have a separate
artifact protocol and never inherit the paper's evaluation status.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

import parc_runtime as runtime


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "PARC_DEVELOPMENT_SELECTOR_V1"
DATA_PROTOCOL = "PARC_DEVELOPMENT_FEATURES_V1"
SEED = 20260719
EPS = 1e-12
PREDICTION_FIELDS = ("regression", "p_positive", "p_negative")
DATA_FIELDS = {
    "protocol", "raw_features", "target", "compression", "anchor_delta",
    "anchor_compression", "task", "family", "query_key",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def protocol_config():
    return json.loads((ROOT / "configs/parc_selection_protocol.json").read_text())


def retrieval_features(values):
    """Use the same label-rejecting parser and feature extractor as deployment."""
    schema = runtime.deployment_v3.feature_schema()
    contract = {
        "dense_baseline": runtime.DENSE_METHOD,
        "safe_default": runtime.SAFE_METHOD,
        "candidate_methods": list(runtime.QUALITY_CANDIDATES),
        "feature_schema": schema,
    }
    queries = runtime.legacy.parse_public_task(values, contract)
    vectors, _ = runtime.deployment_v3.construct_feature_vectors(queries, contract)
    base = np.stack([vectors[name] for name in runtime.QUALITY_CANDIDATES], axis=1)
    raw = runtime.append_candidate_identity(np.asarray(base, dtype=np.float32))
    compression = runtime._compression_matrix(queries)
    anchors = [runtime._select_anchor(query) for query in queries]
    return queries, raw, compression, anchors


def validate_data(data):
    require(set(data) == DATA_FIELDS, "training archive fields mismatch")
    require(np.asarray(data["protocol"]).shape == (), "protocol must be a scalar")
    require(str(data["protocol"]) == DATA_PROTOCOL, "unsupported training protocol")
    raw = np.asarray(data["raw_features"])
    require(raw.ndim == 3 and raw.shape[1:] == (9, 221) and len(raw) > 0,
            "raw_features must have shape (queries, 9, 221)")
    require(raw.dtype == np.float32, "raw_features must use float32")
    n = len(raw)
    for name, shape in (("raw_features", (n, 9, 221)), ("target", (n, 9)),
                        ("compression", (n, 9)), ("anchor_delta", (n,)),
                        ("anchor_compression", (n,))):
        value = np.asarray(data[name])
        require(value.shape == shape and value.dtype.kind == "f", f"invalid {name} shape/dtype")
        require(np.all(np.isfinite(value)), f"non-finite {name}")
    require(np.array_equal(raw[:, :, -9:], np.broadcast_to(np.eye(9), (n, 9, 9))),
            "candidate indicators/order mismatch")
    for name in ("target", "anchor_delta"):
        require(np.all(np.abs(data[name]) <= 1), f"{name} must be F1 differences on a 0-1 scale")
    for name in ("compression", "anchor_compression"):
        require(np.all((data[name] > 0) & (data[name] < 1)), f"{name} must be strictly between 0 and 1")
    for name in ("task", "family", "query_key"):
        value = np.asarray(data[name])
        require(value.shape == (n,) and value.dtype.kind == "U", f"{name} must be a Unicode string vector")
        require(all(item.strip() for item in value.tolist()), f"empty {name}")
    keys = list(zip(data["task"].tolist(), data["query_key"].tolist()))
    require(len(set(keys)) == n, "duplicate task/query_key")
    for task in np.unique(data["task"]):
        require(len(np.unique(data["family"][data["task"] == task])) == 1,
                "each task must belong to exactly one family")
    inventory = json.loads((ROOT / "configs/parc_data_inventory.json").read_text())
    heldout = {row["family"].casefold() for row in inventory["evaluation_tasks"]}
    require(not heldout.intersection(value.casefold() for value in data["family"]),
            "paper evaluation families must not be used for training")
    return data


def load_data(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    return validate_data(data)


def verify_paper_inputs(data):
    """Require exact reference feature/label hashes, not just matching counts."""
    manifest = json.loads((ROOT / "artifacts/parc_selector/parc_frozen_selector_manifest.json").read_text())
    contract, training = manifest["feature_contract"], manifest["training"]
    relative = runtime.augment_relative_features(data["raw_features"], contract["base_schema"])
    expected = {
        "raw_features": contract["development_raw_tensor_sha256"],
        "relative_features": contract["development_relative_tensor_sha256"],
        "target": training["target_tensor_sha256"],
        "compression": training["compression_tensor_sha256"],
        "anchor_delta": training["anchor_target_sha256"],
        "anchor_compression": training["anchor_compression_sha256"],
    }
    arrays = {**data, "relative_features": relative}
    for name, digest in expected.items():
        require(runtime.array_sha256(arrays[name]) == digest, f"paper input hash mismatch: {name}")
    for scheme, field in (("loto", "task"), ("lofo", "family")):
        folds = manifest["oof_policy_evidence"][scheme]["folds"]
        require(set(data[field]) == {fold["outer_unit"] for fold in folds}, f"paper {field} inventory mismatch")
        for fold in folds:
            indices = np.flatnonzero(data[field] == fold["outer_unit"]).astype(np.int64)
            require(runtime.array_sha256(indices) == fold["target_indices_sha256"], f"paper {field} order mismatch")
    return expected


def prepare_data(groups_path, output_path):
    groups_path, output_path = Path(groups_path), Path(output_path)
    spec = json.loads(groups_path.read_text(encoding="utf-8"))
    require(set(spec) == {"role", "groups"} and spec["role"] == "development",
            "groups file must explicitly declare role=development")
    require(isinstance(spec["groups"], list) and spec["groups"], "empty development groups")
    pieces = {key: [] for key in DATA_FIELDS - {"protocol"}}
    required_methods = set(runtime.ANCHOR_CANDIDATES) | {runtime.DENSE_METHOD}
    seen_tasks = set()
    for group in spec["groups"]:
        require(set(group) == {"task", "family", "retrieval", "labels"}, "group fields mismatch")
        require(all(isinstance(value, str) and value.strip() for value in group.values()), "invalid group value")
        require(group["task"] not in seen_tasks, "provide one complete group per development task")
        seen_tasks.add(group["task"])
        values = json.loads((groups_path.parent / group["retrieval"]).read_text(encoding="utf-8"))
        labels = json.loads((groups_path.parent / group["labels"]).read_text(encoding="utf-8"))
        require(isinstance(labels, dict), "labels must map query_key to method F1 scores")
        queries, raw, compression, anchors = retrieval_features(values)
        require(set(labels) == {query.query_key for query in queries}, "label/query coverage mismatch")
        target, anchor_delta = [], []
        for query, anchor in zip(queries, anchors):
            scores = labels[query.query_key]
            require(isinstance(scores, dict) and set(scores) == required_methods, "label method coverage mismatch")
            require(all(type(value) in (int, float) and np.isfinite(value) and 0 <= value <= 1
                        for value in scores.values()), "F1 labels must be finite numbers in [0, 1]")
            dense = scores[runtime.DENSE_METHOD]
            target.append([scores[name] - dense for name in runtime.QUALITY_CANDIDATES])
            anchor_delta.append(scores[anchor["anchor_method"]] - dense)
        pieces["raw_features"].append(raw)
        pieces["target"].append(np.asarray(target, dtype=np.float64))
        pieces["compression"].append(compression)
        pieces["anchor_delta"].append(np.asarray(anchor_delta, dtype=np.float64))
        pieces["anchor_compression"].append(np.asarray([a["chunk_compression"] for a in anchors]))
        for key in ("task", "family"):
            pieces[key].append(np.asarray([group[key]] * len(queries)))
        pieces["query_key"].append(np.asarray([query.query_key for query in queries]))
    data = {key: np.concatenate(value, axis=0) for key, value in pieces.items()}
    data["protocol"] = np.asarray(DATA_PROTOCOL)
    validate_data(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        np.savez_compressed(stream, **data)
    return {"status": "PARC_DEVELOPMENT_FEATURES_READY", "queries": len(data["task"]),
            "sha256": runtime.sha256_file(output_path)}


def task_weights(tasks):
    counts = Counter(tasks.tolist())
    weights = np.asarray([1.0 / counts[task] for task in tasks], dtype=np.float64)
    return weights / weights.mean()


def fit_view(features, target, tasks, seed=SEED):
    """Match the original final-fit parameters, weighting, and label transform."""
    queries, candidates, dimensions = features.shape
    matrix, y = features.reshape(-1, dimensions), target.reshape(-1)
    weights = np.repeat(task_weights(tasks), candidates)
    regressor = HistGradientBoostingRegressor(
        max_iter=220, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=30, l2_regularization=10.0, random_state=seed,
    )
    labels = np.where(y > EPS, 2, np.where(y < -EPS, 0, 1))
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    factors = len(labels) / np.maximum(3.0 * counts, 1.0)
    classifier = ExtraTreesClassifier(
        n_estimators=160, max_depth=14, min_samples_leaf=10,
        max_features=0.65, bootstrap=False, random_state=seed + 97, n_jobs=1,
    )
    with threadpool_limits(limits=1):
        regressor.fit(matrix, y.astype(np.float64) / (np.abs(y.astype(np.float64)) + 0.05), sample_weight=weights)
        classifier.fit(matrix, labels, sample_weight=weights * factors[labels])
    return {"regressor": regressor, "classifier": classifier, "seed": int(seed),
            "queries": queries, "candidates": candidates, "features": dimensions,
            "training_rows": len(y), "class_counts": counts.astype(int).tolist()}


def predict_views(bundle, raw):
    relative = runtime.augment_relative_features(raw, runtime.deployment_v3.feature_schema())
    with threadpool_limits(limits=1):
        return runtime._mean_predictions(runtime._predict_view(bundle["raw"], raw),
                                         runtime._predict_view(bundle["relative"], relative))


def apply_policy(predictions, compression, policy):
    require(policy in protocol_config()["grid"], "policy must be in the published grid")
    shape = compression.shape
    require(len(shape) == 2 and shape[1] == 9, "invalid policy matrix shape")
    require(set(predictions) == set(PREDICTION_FIELDS), "prediction fields mismatch")
    for key, value in predictions.items():
        require(value.shape == shape and np.all(np.isfinite(value)), f"invalid {key}")
        if key.startswith("p_"):
            require(np.all((value >= 0) & (value <= 1)), "invalid probability")
    require(np.all(np.isfinite(compression)), "invalid compression")
    score = (policy["regression_weight"] * predictions["regression"]
             + predictions["p_positive"] - policy["negative_penalty"] * predictions["p_negative"])
    eligible = compression >= runtime.MINIMUM_ACTIVE_QUERY_COMPRESSION
    masked = np.where(eligible, score, -np.inf)
    order = np.argsort(-masked, axis=1, kind="stable")
    rows, best, second = np.arange(len(score)), order[:, 0], order[:, 1]
    has = eligible.any(axis=1)
    margin = np.full(len(score), -np.inf)
    margin[has] = masked[rows[has], best[has]] - masked[rows[has], second[has]]
    active = (has & (masked[rows, best] > policy["score_threshold"])
              & (margin > policy["route_margin"])
              & (predictions["p_negative"][rows, best] <= policy["maximum_negative_probability"])
              & (predictions["p_positive"][rows, best] >= policy["minimum_positive_probability"]))
    return active, best


def policy_summary(data, predictions, policy):
    active, best = apply_policy(predictions, data["compression"], policy)
    rows = np.arange(len(best))
    delta = np.where(active, data["target"][rows, best], data["anchor_delta"])
    compression = np.where(active, data["compression"][rows, best], data["anchor_compression"])
    task_groups = [data["task"] == name for name in np.unique(data["task"])]
    family_groups = [data["family"] == name for name in np.unique(data["family"])]
    task_delta = np.asarray([100 * delta[group].mean() for group in task_groups])
    family_delta = np.asarray([100 * delta[group].mean() for group in family_groups])
    activation = [float(active[group].mean()) for group in task_groups]
    return {
        "tasks": len(task_groups), "families": len(family_groups),
        "task_wins": int(np.sum(task_delta > EPS)), "task_losses": int(np.sum(task_delta < -EPS)),
        "family_wins": int(np.sum(family_delta > EPS)),
        "minimum_task_delta_f1": float(task_delta.min()),
        "minimum_family_delta_f1": float(family_delta.min()),
        "task_macro_delta_f1": float(task_delta.mean()),
        "query_weighted_delta_f1": float(100 * delta.mean()),
        "clipped_query_weighted_delta_f1": float(100 * np.clip(delta, -.05, .05).mean()),
        "active_positive_precision": float(np.mean(delta[active] > EPS)) if active.any() else 0.,
        "active_negative_rate": float(np.mean(delta[active] < -EPS)) if active.any() else 0.,
        "query_weighted_chunk_compression": float(compression.mean()),
        "minimum_task_chunk_compression": float(min(compression[group].mean() for group in task_groups)),
        "mean_task_active_fraction": float(np.mean(activation)),
        "maximum_task_active_fraction": max(activation),
        "all_queries_strictly_compressed": bool(np.all(compression > 0)),
    }


def objective(summary):
    thresholds = protocol_config()["development_thresholds"]
    s, t = summary, thresholds
    feasible = (s["all_queries_strictly_compressed"]
                and s["query_weighted_chunk_compression"] >= t["minimum_query_weighted_compression"]
                and s["minimum_task_chunk_compression"] >= t["minimum_task_mean_compression"]
                and s["mean_task_active_fraction"] <= t["maximum_task_macro_activation"]
                and s["maximum_task_active_fraction"] <= t["maximum_single_task_activation"])
    risk = (s["minimum_task_delta_f1"] >= t["minimum_task_delta_f1_pp"]
            and s["active_negative_rate"] <= t["maximum_active_negative_rate"]
            and s["active_positive_precision"] >= t["minimum_active_positive_precision"])
    return (int(feasible), int(risk), s["family_wins"] / s["families"],
            s["minimum_family_delta_f1"], s["task_wins"] / s["tasks"],
            s["minimum_task_delta_f1"], -s["task_losses"], s["active_positive_precision"],
            -s["active_negative_rate"], s["clipped_query_weighted_delta_f1"],
            s["task_macro_delta_f1"], s["query_weighted_delta_f1"], s["query_weighted_chunk_compression"])


def cross_validated_predictions(data, field):
    require(field in {"task", "family"}, "invalid split field")
    units = np.unique(data[field])
    require(len(units) >= 2, f"need at least two {field} groups")
    raw = data["raw_features"]
    relative = runtime.augment_relative_features(raw, runtime.deployment_v3.feature_schema())
    output = {key: np.full(data["target"].shape, np.nan) for key in PREDICTION_FIELDS}
    folds = []
    for index, unit in enumerate(units):
        train = np.flatnonzero(data[field] != unit).astype(np.int64)
        test = np.flatnonzero(data[field] == unit).astype(np.int64)
        require(not np.intersect1d(train, test).size, "train/held-out overlap")
        seed = SEED + 900_000 + 100_000 * (field == "family") + index
        predictions = []
        for features in (raw, relative):
            model = fit_view(features[train], data["target"][train], data["task"][train], seed)
            with threadpool_limits(limits=1):
                predictions.append(runtime._predict_view(model, features[test]))
        combined = runtime._mean_predictions(*predictions)
        for key in output:
            output[key][test] = combined[key]
        folds.append({"held_out": str(unit), "seed": int(seed), "train_queries": len(train),
                      "held_out_queries": len(test), "train_indices_sha256": runtime.array_sha256(train),
                      "held_out_indices_sha256": runtime.array_sha256(test)})
        print(json.dumps({"stage": "development_oof", "split": field, **folds[-1]}), flush=True)
    require(all(np.all(np.isfinite(value)) for value in output.values()), "incomplete OOF predictions")
    return output, folds


def select_policy(data, loto, lofo):
    best = None
    for index, policy in enumerate(protocol_config()["grid"]):
        summaries = {"loto": policy_summary(data, loto, policy), "lofo": policy_summary(data, lofo, policy)}
        left, right = objective(summaries["loto"]), objective(summaries["lofo"])
        key = tuple(min(a, b) for a, b in zip(left, right)) + tuple((a + b) / 2 for a, b in zip(left, right))
        if best is None or key > tuple(best["objective"]):
            best = {"policy_index": index, "policy": policy, "objective": list(key), **summaries}
    best["feasibility_passed"] = all(objective(best[name])[:2] == (1, 1) for name in ("loto", "lofo"))
    best["scope"] = "development policy selection; not an unbiased evaluation of the selected policy"
    return best


def train_selector(data_path, output_dir, selection="cross-validated", verify_paper=False):
    data_path, output_dir = Path(data_path), Path(output_dir)
    require(not output_dir.exists(), "output directory exists; choose a new directory")
    data = load_data(data_path)
    paper_hashes = verify_paper_inputs(data) if verify_paper else None
    config = protocol_config()
    folds = {}
    if selection == "cross-validated":
        loto, folds["loto"] = cross_validated_predictions(data, "task")
        lofo, folds["lofo"] = cross_validated_predictions(data, "family")
        selected = select_policy(data, loto, lofo)
    else:
        require(selection == "published", "invalid policy selection mode")
        selected = {"policy_index": config["selected_policy_index_zero_based"],
                    "policy": config["selected_policy"], "scope": "published policy reused; no new selection"}
    raw = data["raw_features"]
    relative = runtime.augment_relative_features(raw, runtime.deployment_v3.feature_schema())
    bundle = {"protocol": PROTOCOL,
              "raw": fit_view(raw, data["target"], data["task"]),
              "relative": fit_view(relative, data["target"], data["task"])}
    output_dir.mkdir(parents=True, exist_ok=False)
    model_path = output_dir / "parc_trained_selector.joblib"
    joblib.dump(bundle, model_path, compress=3, protocol=5)
    manifest = {
        "protocol": PROTOCOL, "status": "TRAINED_DEVELOPMENT_SELECTOR_NOT_EVALUATED",
        "model_sha256": runtime.sha256_file(model_path),
        "data_sha256": runtime.sha256_file(data_path),
        "training_source_sha256": runtime.sha256_file(Path(__file__)),
        "base_schema": runtime.deployment_v3.feature_schema(),
        "candidate_methods": list(runtime.QUALITY_CANDIDATES),
        "training_queries": len(raw), "task_counts": dict(Counter(data["task"].tolist())),
        "selection_mode": selection, "selection": selected, "oof_folds": folds,
        "paper_input_hashes_verified": paper_hashes,
        "historical_artifact_reproduced": False,
        "views": {name: {key: value for key, value in bundle[name].items()
                         if key not in {"regressor", "classifier"}} for name in ("raw", "relative")},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
    }
    runtime.atomic_json(output_dir / "parc_trained_selector_manifest.json", manifest)
    return manifest


def route_trained(values, model_path, expected_sha256):
    """Explicit opt-in for a locally trained artifact; frozen loading is untouched."""
    model_path = Path(model_path)
    digest = runtime.sha256_file(model_path)
    require(digest == expected_sha256, "trained model SHA256 mismatch")
    manifest = json.loads(model_path.with_name("parc_trained_selector_manifest.json").read_text())
    require(manifest["protocol"] == PROTOCOL and manifest["model_sha256"] == digest, "trained manifest mismatch")
    require(manifest["candidate_methods"] == list(runtime.QUALITY_CANDIDATES), "candidate order mismatch")
    require(manifest["base_schema"] == runtime.deployment_v3.feature_schema(), "feature schema mismatch")
    # A digest checks integrity, not trust: joblib can execute code when loaded.
    bundle = joblib.load(model_path)
    require(bundle.get("protocol") == PROTOCOL, "trained bundle protocol mismatch")
    queries, raw, compression, anchors = retrieval_features(values)
    predictions = predict_views(bundle, raw)
    active, best = apply_policy(predictions, compression, manifest["selection"]["policy"])
    decisions = [{"query_key": query.query_key,
                  "selected_method": runtime.QUALITY_CANDIDATES[int(best[i])] if active[i] else anchors[i]["anchor_method"],
                  "selection_source": "trained_selector" if active[i] else "compressed_anchor"}
                 for i, query in enumerate(queries)]
    return {"status": "PARC_TRAINED_SELECTOR_ROUTING_COMPLETE", "model_sha256": digest,
            "paper_frozen_model": False, "queries": len(queries), "decisions": decisions}
