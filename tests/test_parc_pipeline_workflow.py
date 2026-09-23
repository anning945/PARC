"""CPU integration tests with synthetic data and injected model backends."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "scripts"),
    str(ROOT / "src/parc_benchmarks"),
    str(ROOT / "tests"),
]

import parc_benchmark_adapters as adapters
import parc_generate as generation_cli
import parc_materialize_retrieval as retrieval_cli
import parc_merge_shards as merge_cli
import parc_score as scoring_cli
from parc_contracts import begin_run, finish_run, read_completed, seal_record
from parc_generation import build_prompt_plan, messages_from_plan
from parc_io import append_jsonl, stable_shard
from parc_scoring import GoldUnit, OfficialScorers
from test_parc_pipeline import CharacterTokenizer, StubReranker


class FakeEmbedder:
    def encode(self, texts):
        return np.asarray(
            [[1, len(text), sum(map(ord, text)) % 101] for text in texts],
            dtype=np.float32,
        )


class PipelineWorkflowTest(unittest.TestCase):
    def test_verified_shard_merge_and_missing_shard_rejection(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            root = Path(folder)
            paths = [root / f"part{index}.jsonl" for index in range(2)]
            config = {"protocol": "PARC-retrieval-materialization-v1"}
            for index, path in enumerate(paths):
                scope = {"shard_count": 2, "shard_index": index, "limit": None}
                config_hash = begin_run(path, config, scope)
                evaluation_key = next(
                    str(value)
                    for value in range(100)
                    if stable_shard(str(value), 2) == index
                )
                append_jsonl(
                    path,
                    seal_record(
                        {
                            "config_sha256": config_hash,
                            "evaluation_key": evaluation_key,
                            "query_key": f"q_{index}",
                        }
                    ),
                )
                finish_run(path, config, scope, 1)
            with self.assertRaisesRegex(ValueError, "shard index"):
                merge_cli.main(
                    [
                        "--inputs",
                        str(paths[0]),
                        "--output",
                        str(root / "incomplete.jsonl"),
                    ]
                )
            merged = root / "merged.jsonl"
            with patch.object(merge_cli, "validate_inventory"):
                self.assertEqual(
                    merge_cli.main(
                        ["--inputs", *map(str, paths), "--output", str(merged)]
                    ),
                    0,
                )
            self.assertEqual(len(read_completed(merged)[0]), 2)

    def test_retrieval_generation_scoring_and_resume(self):
        evaluations = [
            adapters.AdapterEvaluation(
                family="BAMBOO",
                task="synthetic",
                evaluation_key=f"e{index}",
                chunks=tuple(f"evidence {chunk}" for chunk in range(20)),
                rounds=(
                    adapters.AdapterRound(
                        adapters.opaque_key("synthetic", index), "hypothesis", 0
                    ),
                ),
                generation={
                    "kind": "bamboo_hallucination",
                    "system": "Reply yes or no.",
                    "hypothesis": "hypothesis",
                    "final_answer_template": "{content}\n{hypothesis}",
                },
            )
            for index in range(2)
        ]

        def generate(model, tokenizer, packed, device, max_new_tokens, use_cache):
            return {
                "prediction": "Yes",
                "input_tokens": packed["input_tokens"],
                "output_tokens": 1,
            }

        scorers = OfficialScorers(
            lambda labels, predictions: (1.0, 1.0, 1.0),
            None,
            None,
            None,
            None,
            None,
            {},
        )
        gold = [
            GoldUnit("BAMBOO", "synthetic", f"e{index}", 0, True) for index in range(2)
        ]
        with (
            tempfile.TemporaryDirectory() as folder,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            root = Path(folder)
            retrieval, generated = root / "retrieval.jsonl", root / "generation.jsonl"
            data_args = [
                "--bamboo-root",
                folder,
                "--ethic-root",
                folder,
                "--helmet-root",
                folder,
                "--longtable-root",
                folder,
            ]
            retrieval_args = [
                *data_args,
                "--embedder",
                "synthetic",
                "--reranker",
                "synthetic",
                "--output",
                str(retrieval),
                "--cache-dir",
                str(root / "cache"),
                "--device",
                "cpu",
                "--limit",
                "2",
            ]
            generation_args = [
                *data_args,
                "--retrieval",
                str(retrieval),
                "--output",
                str(generated),
                "--model",
                "synthetic",
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--limit-evaluations",
                "2",
            ]
            with (
                patch.object(
                    adapters,
                    "iter_all_evaluations",
                    side_effect=lambda **kwargs: iter(evaluations),
                ),
                patch.object(
                    retrieval_cli, "validate_formatter", return_value="synthetic"
                ),
                patch.object(
                    generation_cli, "validate_formatter", return_value="synthetic"
                ),
                patch.object(
                    scoring_cli, "validate_formatter", return_value="synthetic"
                ),
                patch.object(
                    retrieval_cli,
                    "resolve_model",
                    return_value=(root, {"sha256": "synthetic"}),
                ),
                patch.object(
                    retrieval_cli,
                    "SentenceTransformerEmbedder",
                    return_value=FakeEmbedder(),
                ),
                patch.object(
                    retrieval_cli, "TransformersReranker", return_value=StubReranker()
                ),
                patch.dict("os.environ", {"PYTHONHASHSEED": "0"}),
            ):
                self.assertEqual(retrieval_cli.main(retrieval_args), 0)
                original = retrieval.read_bytes()
                self.assertEqual(retrieval_cli.main(retrieval_args), 0)
                self.assertEqual(retrieval.read_bytes(), original)
                with (
                    patch.object(
                        generation_cli,
                        "resolve_model",
                        return_value=(root, {"sha256": "synthetic"}),
                    ),
                    patch.object(
                        generation_cli,
                        "load_generator",
                        return_value=(object(), CharacterTokenizer()),
                    ),
                    patch.object(
                        generation_cli, "generate_one", side_effect=generate
                    ) as mock_generate,
                ):
                    self.assertEqual(generation_cli.main(generation_args), 0)
                    self.assertEqual(mock_generate.call_count, 4)
                    self.assertEqual(generation_cli.main(generation_args), 0)
                    self.assertEqual(mock_generate.call_count, 4)
                self.assertEqual(len(read_completed(generated)[0]), 4)
                with (
                    patch.object(scoring_cli, "iter_gold", return_value=iter(gold)),
                    patch.object(
                        scoring_cli, "load_official_scorers", return_value=scorers
                    ),
                    patch.object(scoring_cli, "validate_inventory"),
                ):
                    self.assertEqual(
                        scoring_cli.main(
                            [
                                *data_args,
                                "--generation",
                                str(generated),
                                "--output-dir",
                                str(root / "scores"),
                                "--repos-root",
                                folder,
                                "--allow-partial",
                            ]
                        ),
                        0,
                    )
                summary = json.loads(
                    (root / "scores/parc_score_summary.json").read_text()
                )
                self.assertEqual(summary["status"], "PARTIAL_EXPLICIT")
                self.assertEqual(summary["summary"]["scored_units"], 2)
                self.assertFalse(summary["contains_predictions"])

    def test_prompt_parity_all_families_and_multiturn_history(self):
        common = dict(
            task="synthetic",
            evaluation_key="e",
            chunks=("alpha", "beta"),
            rounds=(adapters.AdapterRound("q", "question", 0),),
        )
        bamboo = adapters.AdapterEvaluation(
            family="BAMBOO",
            **common,
            generation={
                "kind": "bamboo_hallucination",
                "system": "system",
                "final_answer_template": "context {content} question {hypothesis}",
                "hypothesis": "question",
            },
        )
        ethic = adapters.AdapterEvaluation(
            family="ETHIC",
            **common,
            generation={
                "kind": "ethic_context_slot",
                "system_msg": "system",
                "prefix": "prefix ",
                "suffix": " suffix",
            },
        )
        helmet = adapters.AdapterEvaluation(
            family="HELMET",
            **common,
            generation={
                "kind": "helmet_rag",
                "user_template": adapters.HELMET_USER_TEMPLATE,
                "system_template": adapters.HELMET_SYSTEM_TEMPLATE,
                "demos": "demo text",
                "question": "question",
            },
        )
        expected = [
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": adapters.render_bamboo_prompt(bamboo, [0])},
            ],
            adapters.render_ethic_messages(ethic, [0]),
            [{"role": "user", "content": adapters.render_helmet_prompt(helmet, [0])}],
        ]
        for evaluation, messages in zip((bamboo, ethic, helmet), expected):
            self.assertEqual(
                messages_from_plan(
                    build_prompt_plan(evaluation, [0], 0, [], ROOT, adapters)
                ),
                messages,
            )
        table = adapters.AdapterEvaluation(
            family="LongTableBench",
            **{
                **common,
                "rounds": (
                    adapters.AdapterRound("q0", "first", 0),
                    adapters.AdapterRound("q1", "second", 1),
                ),
            },
            generation={
                "kind": "longtablebench",
                "round_questions": ("first", "second"),
                "is_multi_turn": True,
            },
        )
        with patch.object(adapters, "render_longtable_context", return_value="table"):
            messages = messages_from_plan(
                build_prompt_plan(
                    table, [0], 1, ["real previous output"], ROOT, adapters
                )
            )
            self.assertEqual(
                messages,
                adapters.build_longtable_messages(
                    table,
                    [0],
                    ROOT,
                    round_index=1,
                    previous_outputs=["real previous output"],
                ),
            )
            self.assertEqual(messages[2]["role"], "system")

    def test_embedding_cache_tracks_text_and_checks_values(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = retrieval_cli.cached_encode(root, FakeEmbedder(), ["a"])
            second = retrieval_cli.cached_encode(root, FakeEmbedder(), ["longer"])
            self.assertFalse(np.array_equal(first, second))
            self.assertEqual(len(list(root.glob("*.npz"))), 2)
