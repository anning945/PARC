"""Check training contracts, reference parameters, and isolated artifact loading."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/parc_router"))
sys.path.insert(0, str(ROOT / "scripts"))
import parc_runtime as runtime
import parc_training as training
from parc_make_training_example import make_example


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.input_dir = cls.root / "input"
        make_example(cls.input_dir)
        cls.data_path = cls.input_dir / "parc_development.npz"
        cls.data = training.load_data(cls.data_path)
        cls.output = cls.root / "trained"
        cls.manifest = training.train_selector(cls.data_path, cls.output, "published")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_prepared_features_match_released_runtime(self):
        values = json.loads((self.input_dir / "parc_synthetic_retrieval_0.json").read_text())
        frozen = runtime.reconstruct_public_task(values, ROOT / "artifacts/parc_selector/parc_frozen_selector.joblib")
        np.testing.assert_array_equal(self.data["raw_features"][:8], frozen["raw_features"])
        np.testing.assert_array_equal(self.data["compression"][:8], frozen["compression"])
        self.assertEqual(self.manifest["training_queries"], 32)
        self.assertFalse(self.manifest["historical_artifact_reproduced"])

    def test_parameters_match_original_frozen_models(self):
        bundle, _, _, _ = runtime.load_frozen_model(ROOT / "artifacts/parc_selector/parc_frozen_selector.joblib")
        trained = training.joblib.load(self.output / "parc_trained_selector.joblib")
        for view in ("raw", "relative"):
            for model in ("regressor", "classifier"):
                self.assertEqual(bundle[view][model].get_params(), trained[view][model].get_params())
            self.assertEqual(bundle[view]["seed"], trained[view]["seed"])

    def test_equal_task_mass_and_class_counts(self):
        tasks = np.asarray(["a", "a", "b"])
        weights = training.task_weights(tasks)
        self.assertAlmostEqual(weights[tasks == "a"].sum(), weights[tasks == "b"].sum())
        expected = np.bincount(np.where(self.data["target"] > 1e-12, 2,
                                       np.where(self.data["target"] < -1e-12, 0, 1)).ravel(), minlength=3)
        self.assertEqual(expected.tolist(), self.manifest["views"]["raw"]["class_counts"])

    def test_invalid_training_inputs_are_rejected(self):
        changes = [
            ("target", np.full((32, 9), np.nan)),
            ("target", np.full((32, 9), 1.1)),
            ("compression", np.zeros((32, 9))),
            ("family", np.asarray(["BAMBOO"] * 32)),
            ("raw_features", self.data["raw_features"].astype(np.float64)),
            ("query_key", np.asarray(["duplicate"] * 32)),
        ]
        for key, value in changes:
            with self.subTest(key=key):
                altered = {**self.data, key: value}
                with self.assertRaises(ValueError):
                    training.validate_data(altered)
        altered = copy.deepcopy(self.data)
        altered["raw_features"][0, 0, -1] = 2
        with self.assertRaisesRegex(ValueError, "indicators"):
            training.validate_data(altered)

    def test_no_evaluation_labels_in_retrieval_parser(self):
        values = json.loads((self.input_dir / "parc_synthetic_retrieval_0.json").read_text())
        values[0]["f1"] = 0.7
        with self.assertRaises(ValueError):
            training.retrieval_features(values)

    def test_group_and_label_coverage_validation(self):
        spec_path = self.input_dir / "parc_training_groups.json"
        spec = json.loads(spec_path.read_text())
        with tempfile.TemporaryDirectory(dir=self.input_dir) as tmp:
            path = Path(tmp) / "groups.json"
            path.write_text(json.dumps({**spec, "role": "evaluation"}))
            with self.assertRaisesRegex(ValueError, "role=development"):
                training.prepare_data(path, Path(tmp) / "bad.npz")
        labels_path = self.input_dir / "parc_synthetic_labels_0.json"
        original = labels_path.read_text()
        try:
            labels = json.loads(original)
            del labels[next(iter(labels))]
            labels_path.write_text(json.dumps(labels))
            with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                training.prepare_data(spec_path, self.root / "bad.npz")
        finally:
            labels_path.write_text(original)

    def test_new_models_cannot_masquerade_as_frozen_artifact(self):
        model = self.output / "parc_trained_selector.joblib"
        values = json.loads((self.input_dir / "parc_synthetic_retrieval_0.json").read_text())
        with self.assertRaisesRegex(ValueError, "frozen model SHA256"):
            runtime.load_frozen_model(model)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            training.route_trained(values, model, "0" * 64)
        first = training.route_trained(values, model, self.manifest["model_sha256"])
        second = training.route_trained(values, model, self.manifest["model_sha256"])
        self.assertEqual(first, second)
        self.assertFalse(first["paper_frozen_model"])
        self.assertEqual(first["queries"], 8)

    def test_no_overwrite_or_false_paper_identity(self):
        with self.assertRaisesRegex(ValueError, "output directory exists"):
            training.train_selector(self.data_path, self.output, "published")
        with self.assertRaisesRegex(ValueError, "paper input hash mismatch"):
            training.verify_paper_inputs(self.data)

    def test_policy_matches_frozen_runtime_including_zero_eligible(self):
        rng = np.random.default_rng(4)
        predictions = {"regression": rng.normal(size=(8, 9)),
                       "p_positive": rng.uniform(size=(8, 9)), "p_negative": rng.uniform(size=(8, 9))}
        compression = rng.uniform(.2, .7, (8, 9))
        compression[0] = .2
        compression[1] = .2
        compression[1, 3] = .6
        active, best = training.apply_policy(predictions, compression, runtime.EXPECTED_POLICY)
        expected = runtime.apply_frozen_policy(predictions, compression)
        np.testing.assert_array_equal(active, expected["active"])
        np.testing.assert_array_equal(best, expected["best_candidate"])

    def test_oof_excludes_entire_groups_and_selection_preserves_ties(self):
        predictions, folds = training.cross_validated_predictions(self.data, "family")
        self.assertEqual(len(folds), 2)
        for fold in folds:
            self.assertEqual(fold["train_queries"], 16)
            self.assertEqual(fold["held_out_queries"], 16)
            self.assertNotEqual(fold["train_indices_sha256"], fold["held_out_indices_sha256"])
        selected = training.select_policy(self.data, predictions, predictions)
        self.assertIn(selected["policy"], training.protocol_config()["grid"])
        tied = {key: np.zeros_like(self.data["target"]) for key in training.PREDICTION_FIELDS}
        self.assertEqual(training.select_policy(self.data, tied, tied)["policy_index"], 0)


if __name__ == "__main__":
    unittest.main()
