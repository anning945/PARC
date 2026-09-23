"""Check paper-scope aggregates against the complete execution inventory."""

import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "parc_paper35_summary", ROOT / "scripts/parc_summarize_paper35.py"
)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


class Paper35ResultsTests(unittest.TestCase):
    def setUp(self):
        self.inventory = summary.read_json(summary.INVENTORY)
        self.scores = summary.read_json(summary.SCORES)

    def test_current_scope_matches_execution_inventory(self):
        execution = summary.read_json(ROOT / "configs/parc_data_inventory.json")
        indexed = {r["task_key"]: r for r in execution["evaluation_tasks"]}
        self.assertEqual(len(indexed), 50)
        self.assertEqual(len(self.inventory["tasks"]), 35)
        self.assertEqual(self.inventory["task_count"], 35)
        self.assertEqual(sum(r["scored_units"] for r in self.inventory["tasks"]), 11997)
        self.assertEqual(self.inventory["scored_units"], 11997)
        for row in self.inventory["tasks"]:
            self.assertEqual(row, indexed[row["task_key"]])
        families = {f: sum(r["family"] == f for r in self.inventory["tasks"])
                    for f in ("ETHIC", "HELMET", "LongTableBench")}
        self.assertEqual(families, {"ETHIC": 1, "HELMET": 1, "LongTableBench": 33})

    def test_published_aggregates_and_private_data_boundary(self):
        result = summary.summarize_published(self.scores, self.inventory)
        self.assertEqual(result, summary.read_json(summary.SUMMARY))
        self.assertEqual(result["compressed_units"], 11997)
        qwen = result["models"]["qwen"]["overall"]
        llama = result["models"]["llama"]["overall"]
        self.assertAlmostEqual(qwen["parc_macro_f1"], .4410611501430988, places=14)
        self.assertAlmostEqual(llama["parc_macro_f1"], .28381469103541695, places=14)
        self.assertAlmostEqual(qwen["mean_chunk_compression"], .5642904189684251, places=14)
        self.assertAlmostEqual(result["qwen_controls"]["fixed_compressed_dense"]["method_macro_f1"],
                               .4273917861832410, places=14)
        for value in (self.inventory, self.scores, result):
            text = json.dumps(value)
            for forbidden in ("/Users/", "/mnt/", "id_rsa", "BEGIN PRIVATE", "successor_"):
                self.assertNotIn(forbidden, text)

    def test_task_macro_and_unit_weighted_compression_are_distinct(self):
        rows = [dict(scored_units=1, dense_f1=.2, parc_f1=.1, mean_chunk_compression=.2),
                dict(scored_units=9, dense_f1=.4, parc_f1=.3, mean_chunk_compression=.6)]
        result = summary.aggregate(rows)
        self.assertAlmostEqual(result["dense_macro_f1"], .3)
        self.assertAlmostEqual(result["parc_macro_f1"], .2)
        self.assertAlmostEqual(result["mean_chunk_compression"], .56)

    def test_rejects_missing_duplicate_miscounted_and_invalid_scores(self):
        for mutation in ("missing", "duplicate", "count", "nan", "delta", "coverage"):
            scores = copy.deepcopy(self.scores)
            rows = scores["tasks"]
            if mutation == "missing":
                rows.pop()
            elif mutation == "duplicate":
                rows.append(copy.deepcopy(rows[0]))
            elif mutation == "count":
                rows[0]["scored_units"] -= 1
            elif mutation == "nan":
                rows[0]["qwen_dense_f1"] = float("nan")
            elif mutation == "delta":
                rows[0]["qwen_delta"] = .5
            else:
                rows[0]["parc_compressed_units"] -= 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                summary.summarize_published(scores, self.inventory)

    def make_replay(self, root):
        rows = summary.read_json(ROOT / "configs/parc_data_inventory.json")["evaluation_tasks"]
        path = root / "parc_task_scores.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "family", "task", "scored_units", "dense_f1", "parc_f1", "mean_chunk_compression"
            ])
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(family=row["family"], task=row["task"],
                                     scored_units=row["scored_units"], dense_f1=.5,
                                     parc_f1=.4, mean_chunk_compression=.6))
        artifact = dict(status="COMPLETE", protocol="PARC-official-scoring-v1",
                        task_scores_csv=path.name,
                        task_scores_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        output = root / "parc_score_summary.json"
        output.write_text(json.dumps(artifact))
        return output, artifact, path

    def test_new_replay_keeps_fixed_tasks_even_when_all_deltas_are_negative(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _, _ = self.make_replay(Path(folder))
            result = summary.summarize_replay(path, self.inventory)
            self.assertEqual(result["overall"]["tasks"], 35)
            self.assertEqual(result["overall"]["scored_units"], 11997)
            self.assertAlmostEqual(result["overall"]["delta_f1_pp"], -10)

    def test_replay_rejects_partial_tampered_and_incomplete_inputs(self):
        for mutation in ("partial", "tampered", "missing"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                path, artifact, csv_path = self.make_replay(Path(folder))
                if mutation == "partial":
                    artifact["status"] = "PARTIAL_EXPLICIT"
                elif mutation == "tampered":
                    csv_path.write_text(csv_path.read_text() + "\n")
                else:
                    csv_path.write_text("\n".join(csv_path.read_text().splitlines()[:-1]) + "\n")
                    artifact["task_scores_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
                path.write_text(json.dumps(artifact))
                with self.assertRaises(ValueError):
                    summary.summarize_replay(path, self.inventory)


if __name__ == "__main__":
    unittest.main()
