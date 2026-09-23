"""Check public protocol metadata against the already released freeze."""

import csv
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProtocolMetadataTests(unittest.TestCase):
    def setUp(self):
        self.manifest_path = ROOT / "artifacts/parc_selector/parc_frozen_selector_manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text())
        self.inventory = json.loads((ROOT / "configs/parc_data_inventory.json").read_text())
        self.protocol = json.loads((ROOT / "configs/parc_selection_protocol.json").read_text())

    def test_development_matches_frozen_folds(self):
        tasks = self.inventory["development_tasks"]
        self.assertEqual(len(tasks), 12)
        self.assertEqual(sum(t["queries"] for t in tasks), 5763)
        folds = {f["outer_unit"]: f for f in self.manifest["oof_policy_evidence"]["loto"]["folds"]}
        self.assertEqual({t["task"] for t in tasks}, set(folds))
        for task in tasks:
            fold = folds[task["task"]]
            self.assertEqual(task["queries"], fold["target_queries"])
            self.assertEqual(task["loto_target_indices_sha256"], fold["target_indices_sha256"])
        for fold in self.manifest["oof_policy_evidence"]["lofo"]["folds"]:
            self.assertEqual(sum(t["queries"] for t in tasks if t["family"] == fold["outer_unit"]), fold["target_queries"])

    def test_evaluation_is_complete_unfiltered_inventory(self):
        rows = self.inventory["evaluation_tasks"]
        with (ROOT / "results/parc_task_quality_verified.csv").open(newline="") as stream:
            expected = {r["task_key"]: r for r in csv.DictReader(stream)}
        self.assertEqual(len(rows), 50)
        self.assertEqual({r["task_key"] for r in rows}, set(expected))
        self.assertEqual(sum(r["scored_units"] for r in rows), 18035)
        self.assertEqual(sum(r["evaluations"] for r in rows), 12309)
        self.assertFalse({r["family"] for r in rows} & {r["family"] for r in self.inventory["development_tasks"]})
        for row in rows:
            self.assertEqual(set(row), {"task_key", "family", "task", "evaluations", "scored_units"})
            for key in ("evaluations", "scored_units"):
                self.assertEqual(row[key], int(expected[row["task_key"]][key]))

    def test_grid_and_selected_policy_match_frozen_model(self):
        method = self.manifest["method"]
        self.assertEqual(self.protocol["grid"], method["policy_grid"])
        self.assertEqual(len(self.protocol["grid"]), 144)
        self.assertEqual(self.protocol["selected_policy_index_zero_based"], 116)
        self.assertEqual(self.protocol["grid"][116], self.protocol["selected_policy"])
        self.assertEqual(self.protocol["selected_policy"], method["selected_final_policy"]["policy"])
        self.assertEqual(len(self.protocol["objective_order_maximize"]), 13)
        self.assertEqual(self.protocol["development_thresholds"], {
            "minimum_query_weighted_compression": .55, "minimum_task_mean_compression": .55,
            "maximum_task_macro_activation": .70, "maximum_single_task_activation": .90,
            "minimum_task_delta_f1_pp": -1.5, "maximum_active_negative_rate": .20,
            "minimum_active_positive_precision": .12, "query_delta_clip_fraction": .05,
        })

    def test_provenance_and_metadata_boundary(self):
        digest = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        for value in (self.inventory, self.protocol):
            self.assertEqual(value["frozen_model_manifest_sha256"], digest)
            serialized = json.dumps(value)
            for forbidden in ("/mnt/", "/Users/", "id_rsa", "BEGIN PRIVATE", "SmartMemory", "successor_"):
                self.assertNotIn(forbidden, serialized)


if __name__ == "__main__":
    unittest.main()
