import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/parc_router"))

import parc_runtime as runtime  # noqa: E402


class RoutingContractTest(unittest.TestCase):
    def setUp(self):
        self.values = json.loads(
            (ROOT / "examples/parc_synthetic_task_input.json").read_text()
        )
        self.contract = {
            "dense_baseline": runtime.DENSE_METHOD,
            "safe_default": runtime.SAFE_METHOD,
            "candidate_methods": list(runtime.QUALITY_CANDIDATES),
        }

    def parse(self):
        return runtime.legacy.parse_public_task(self.values, self.contract)

    def test_fixture_has_all_eleven_methods(self):
        queries = self.parse()
        self.assertEqual(len(queries), 1)
        self.assertEqual(len(queries[0].methods), 11)

    def test_test_answers_and_identity_fields_are_rejected(self):
        for field in ("answer", "prediction", "f1", "benchmark", "model"):
            with self.subTest(field=field):
                values = copy.deepcopy(self.values)
                values[0][field] = "forbidden"
                with self.assertRaisesRegex(ValueError, "forbidden inference field|schema mismatch"):
                    runtime.legacy.parse_public_task(values, self.contract)

    def test_missing_candidate_is_rejected(self):
        del self.values[0]["methods"][runtime.QUALITY_CANDIDATES[0]]
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            self.parse()

    def test_noncompressed_candidate_is_rejected(self):
        method = self.values[0]["methods"][runtime.SAFE_METHOD]
        method["kept_indices"] = list(range(len(self.values[0]["chunks"])))
        with self.assertRaisesRegex(ValueError, "non-positive chunk compression"):
            self.parse()

    def test_duplicate_query_is_rejected(self):
        self.values.append(copy.deepcopy(self.values[0]))
        with self.assertRaisesRegex(ValueError, "duplicate query_key"):
            self.parse()

    def test_frozen_policy_matches_public_configuration(self):
        config = json.loads((ROOT / "configs/parc_policy.json").read_text())
        self.assertEqual(config["candidate_methods"], list(runtime.QUALITY_CANDIDATES))
        self.assertEqual(config["dense_baseline"], runtime.DENSE_METHOD)
        self.assertEqual(config["safe_default"], runtime.SAFE_METHOD)
        self.assertEqual(
            config["policy"],
            {**runtime.EXPECTED_POLICY,
             "minimum_active_query_compression": runtime.MINIMUM_ACTIVE_QUERY_COMPRESSION},
        )

    def test_frozen_routing_is_deterministic_and_compressed(self):
        model = ROOT / "artifacts/parc_selector/parc_frozen_selector.joblib"
        first = runtime.route_public_task(self.values, model)
        second = runtime.route_public_task(self.values, model)
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "COMPLETE_RETRIEVAL_ONLY_ROUTING_V9")
        self.assertEqual(first["queries"], 1)
        selected = first["decisions"][0]["selected_method"]
        self.assertEqual(selected, "PARC_CompressedDense_DocOrder_R45_k8")
        self.assertLess(
            len(self.values[0]["methods"][selected]["kept_indices"]),
            len(self.values[0]["chunks"]),
        )
        self.assertFalse(first["forbidden_quality_inputs_read"])


if __name__ == "__main__":
    unittest.main()
