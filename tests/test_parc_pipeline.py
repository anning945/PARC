import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/parc_pipeline"))
sys.path.insert(0, str(ROOT / "src/parc_router"))

import parc_compat_runtime as compatibility_runtime  # noqa: E402
from parc_contracts import (
    begin_run,
    check_record,
    finish_run,
    read_completed,
    seal_record,
    validate_inventory,
)
from parc_generation import PromptMessage, PromptSegment, pack_prompt  # noqa: E402
from parc_io import append_jsonl, load_unique_jsonl
from parc_retrieval import (  # noqa: E402
    DENSE_METHOD,
    METHODS,
    SAFE_METHOD,
    materialize_methods,
)
from parc_scoring import (
    GoldUnit,
    OfficialScorers,
    bamboo_prediction,  # noqa: E402
    summarize,
)


class StubReranker:
    def score(self, question, chunks):
        return np.asarray(
            [len(chunk) + index for index, chunk in enumerate(chunks)], dtype=np.float32
        )


class FakeEncoding(dict):
    def __init__(self, values):
        super().__init__(input_ids=np.asarray([values], dtype=np.int64))


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        suffix = "assistant:" if add_generation_prompt else ""
        return "".join(f"{row['role']}:{row['content']}\n" for row in messages) + suffix

    def __call__(self, text, **kwargs):
        return FakeEncoding(self.encode(text))

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def decode(self, values, **kwargs):
        return "".join(chr(value) for value in values)


class RetrievalPipelineTest(unittest.TestCase):
    def test_reference_index_regression(self):
        # Golden index trace checked against the archived formal retriever;
        # no benchmark text, labels, predictions, or scores are in this fixture.
        class ReferenceReranker:
            def score(self, question, chunks):
                return np.asarray(
                    [sum(map(ord, chunk)) % 101 for chunk in chunks], dtype=np.float32
                )

        rng = np.random.default_rng(42)
        digest = hashlib.sha256()
        for count in (2, 3, 7, 8, 9, 10, 15, 17, 20, 31, 64, 100):
            for _ in range(10):
                embeddings = rng.normal(size=(count, 8)).astype(np.float32)
                query = rng.normal(size=8).astype(np.float32)
                chunks = [
                    f"evidence {index} entity {index % 7}" for index in range(count)
                ]
                for question in (
                    "which evidence entity 5",
                    "\u54ea\u4e2a\u5b9e\u4f53 5",
                    "",
                ):
                    observed = materialize_methods(
                        question, chunks, embeddings, query, ReferenceReranker()
                    )
                    indices = {
                        method: {
                            field: value
                            for field, value in row.items()
                            if field.endswith("indices")
                        }
                        for method, row in observed.items()
                    }
                    digest.update(
                        json.dumps(
                            indices, sort_keys=True, separators=(",", ":")
                        ).encode()
                    )
        self.assertEqual(
            digest.hexdigest(),
            "392faf06c5f5fd66fe340cd4b1970f0f5b9f473368eb22d0beee154094c000d7",
        )

    def test_short_pools_keep_strict_compression(self):
        for count in range(2, 21):
            methods = materialize_methods(
                "x",
                [str(i) for i in range(count)],
                np.eye(count, dtype=np.float32),
                np.arange(count, dtype=np.float32),
                StubReranker(),
            )
            for name, row in methods.items():
                if name != DENSE_METHOD:
                    self.assertLess(len(row["kept_indices"]), count)
                    self.assertEqual(len(row["retrieved_indices"]), min(8, count - 1))

    def test_single_chunk_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            materialize_methods("x", ["x"], np.ones((1, 1)), np.ones(1), StubReranker())

    def test_nonfinite_embeddings_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            materialize_methods(
                "x", ["a", "b"], np.full((2, 1), np.nan), np.ones(1), StubReranker()
            )

    def test_all_frozen_methods_are_materialized_and_compressed(self):
        chunks = [f"chunk {index} evidence" for index in range(20)]
        embeddings = np.eye(20, dtype=np.float32)
        query = np.linspace(0.0, 1.0, 20, dtype=np.float32)
        methods = materialize_methods(
            "evidence 19", chunks, embeddings, query, StubReranker()
        )
        self.assertEqual(set(methods), {method.name for method in METHODS})
        self.assertEqual(len(methods[DENSE_METHOD]["kept_indices"]), 20)
        self.assertLess(len(methods[SAFE_METHOD]["kept_indices"]), 20)
        for name, observation in methods.items():
            self.assertTrue(
                set(observation["retrieved_indices"]).issubset(
                    observation["kept_indices"]
                ),
                name,
            )
        contract = {
            "dense_baseline": DENSE_METHOD,
            "safe_default": SAFE_METHOD,
            "candidate_methods": [
                method.name
                for method in METHODS
                if method.name not in {DENSE_METHOD, SAFE_METHOD}
            ],
        }
        parsed = compatibility_runtime.parse_public_task(
            [
                {
                    "query_key": "q",
                    "question": "evidence 19",
                    "chunks": chunks,
                    "methods": methods,
                }
            ],
            contract,
        )
        self.assertEqual(len(parsed), 1)

    def test_bamboo_answer_parser_matches_paper_contract(self):
        self.assertTrue(bamboo_prediction("Yes"))
        self.assertFalse(bamboo_prediction("No"))


class PromptPackingTest(unittest.TestCase):
    def test_fitting_prompt_is_unchanged(self):
        plan = [PromptMessage("user", (PromptSegment("context", "short", "context"),))]
        config = {
            "decoding": {"max_input_tokens": 100, "target_input_tokens": 90},
            "packing": {
                "priority": ["history", "context", "instruction"],
                "omission_marker": "<CUT>",
            },
        }
        packed = pack_prompt(CharacterTokenizer(), plan, config)
        self.assertFalse(packed["packing_applied"])
        self.assertEqual(packed["prompt"], "user:short\nassistant:")

    def test_protected_overflow_is_rejected(self):
        config = {
            "decoding": {"max_input_tokens": 100, "target_input_tokens": 90},
            "packing": {
                "priority": ["history", "context", "instruction"],
                "omission_marker": "<CUT>",
            },
        }
        plan = [
            PromptMessage("user", (PromptSegment("protected", "x" * 200, "question"),))
        ]
        with self.assertRaisesRegex(ValueError, "protected"):
            pack_prompt(CharacterTokenizer(), plan, config)

    def test_overflow_packing_preserves_protected_segments(self):
        tokenizer = CharacterTokenizer()
        plan = (
            PromptMessage("system", (PromptSegment("protected", "RULE", "system"),)),
            PromptMessage(
                "user",
                (
                    PromptSegment("context", "x" * 200, "context"),
                    PromptSegment("protected", "QUESTION", "question"),
                ),
            ),
        )
        config = {
            "decoding": {"max_input_tokens": 100, "target_input_tokens": 90},
            "packing": {
                "priority": ["history", "context", "instruction"],
                "omission_marker": "<CUT>",
            },
        }
        packed = pack_prompt(tokenizer, plan, config)
        self.assertTrue(packed["packing_applied"])
        self.assertLessEqual(packed["input_tokens"], 90)
        self.assertIn("RULE", packed["prompt"])
        self.assertIn("QUESTION", packed["prompt"])


class PipelineIntegrityTest(unittest.TestCase):
    def test_complete_artifact_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.jsonl"
            config, scope = {"model": "synthetic"}, {"shard_count": 1}
            config_hash = begin_run(path, config, scope)
            row = seal_record({"config_sha256": config_hash, "query_key": "q"})
            append_jsonl(path, row)
            finish_run(path, config, scope, 1)
            self.assertEqual(read_completed(path)[0], [row])
            self.assertEqual(begin_run(path, config, scope), config_hash)
            with self.assertRaisesRegex(ValueError, "not complete"):
                read_completed(path)
            check_record(row, config_hash)

    def test_changed_config_and_record_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.jsonl"
            config_hash = begin_run(path, {"model": "first"}, {})
            row = seal_record({"config_sha256": config_hash, "query_key": "q"})
            append_jsonl(path, row)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                begin_run(path, {"model": "second"}, {})
            with self.assertRaisesRegex(ValueError, "SHA256"):
                check_record({**row, "query_key": "modified"}, config_hash)

    def test_output_without_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.jsonl"
            append_jsonl(path, {"query_key": "q"})
            with self.assertRaisesRegex(ValueError, "without"):
                begin_run(path, {}, {})

    def test_duplicates_and_incomplete_inventory_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.jsonl"
            for _ in range(2):
                append_jsonl(path, {"query_key": "q"})
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_unique_jsonl(path, key_fields=("query_key",))
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            validate_inventory([], ROOT / "configs/parc_data_inventory.json")

    def test_unit_weighted_compression_and_task_macro_f1(self):
        gold = {}
        records = {}
        for index in range(4):
            unit = GoldUnit(
                "HELMET", "a" if index == 0 else "b", str(index), 0, ["yes"]
            )
            gold[unit.key] = unit
            records[unit.key] = {
                "dense": {"prediction": "no", "chunk_compression": 0.0},
                "parc": {
                    "prediction": "yes",
                    "chunk_compression": 0.2 if index == 0 else 0.6,
                },
            }
        scorers = OfficialScorers(
            None,
            None,
            lambda pred, answer: (float(pred == answer), 0, 0),
            lambda metric, pred, answers: max(
                metric(pred, answer) for answer in answers
            ),
            lambda pred, cue: None,
            None,
            {},
        )
        rows, summary = summarize(records, gold, scorers)
        self.assertAlmostEqual(summary["mean_chunk_compression"], 0.5)
        self.assertEqual(summary["parc_macro_f1"], 1.0)
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
