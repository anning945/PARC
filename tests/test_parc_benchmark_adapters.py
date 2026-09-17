import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parc_benchmarks.parc_benchmark_adapters import (  # noqa: E402
    AdapterEvaluation,
    AdapterRound,
    canonical_json,
    split_context,
    validate_selector_stub,
)


class PublicAdapterContractTest(unittest.TestCase):
    def test_split_context_runs_with_pinned_adapter_dependencies(self) -> None:
        chunks = split_context("Synthetic retrieval context with repeated words. " * 40)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.strip() and len(chunk) <= 300 for chunk in chunks))

    def test_empty_context_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no chunks"):
            split_context("   ")

    def test_canonical_json_is_stable(self) -> None:
        self.assertEqual(canonical_json({"b": 2, "a": 1}), '{"a":1,"b":2}')

    def test_selector_stub_rejects_quality_fields(self) -> None:
        valid = {
            "query_key": "q_opaque",
            "question": "What is the answer?",
            "chunks": ["context"],
        }
        validate_selector_stub(valid)
        with self.assertRaisesRegex(ValueError, "selector schema mismatch"):
            validate_selector_stub({**valid, "f1": 0.5})

    def test_runtime_query_adds_methods_only_after_stub_validation(self) -> None:
        evaluation = AdapterEvaluation(
            family="BAMBOO",
            task="synthetic",
            evaluation_key="q_eval",
            chunks=("context",),
            rounds=(
                AdapterRound(
                    query_key="q_query",
                    question="What is the answer?",
                    round_index=0,
                ),
            ),
            generation={"kind": "synthetic"},
        )
        query = evaluation.runtime_query(0, {"PARC_CompressedDense_R45_k8": {}})
        self.assertEqual(set(query), {"query_key", "question", "chunks", "methods"})


if __name__ == "__main__":
    unittest.main()
