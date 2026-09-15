"""Create a tiny retrieval-only input for schema validation."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "parc_router"))

from successor_selector_v9_dualview_runtime_20260720 import (  # noqa: E402
    ANCHOR_CANDIDATES, DENSE_METHOD, QUALITY_CANDIDATES, SAFE_METHOD
)


def method_observation(compressed: bool) -> dict:
    chunks = 10
    kept = list(range(9 if compressed else chunks))
    retrieved = list(range(min(8, len(kept))))
    reference = list(range(8))
    scores = [1.0 - 0.01 * i for i in range(len(retrieved))]
    return {
        "kept_indices": kept,
        "retrieved_indices": retrieved,
        "dense_top_k_indices": reference,
        "bm25_top_k_indices": reference,
        "rrf_top_k_indices": reference,
        "retrieved_dense_scores": scores,
        "retrieved_bm25_scores": scores,
        "retrieved_rrf_scores": scores,
        "retrieved_reranker_scores": scores,
    }


def main() -> None:
    names = set(QUALITY_CANDIDATES) | set(ANCHOR_CANDIDATES) | {DENSE_METHOD, SAFE_METHOD}
    payload = [{
        "query_key": "synthetic-0001",
        "question": "Which synthetic item is listed first?",
        "chunks": [f"synthetic chunk {i}" for i in range(10)],
        "methods": {
            name: method_observation(compressed=name != DENSE_METHOD)
            for name in sorted(names)
        },
    }]
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "examples" / "synthetic_task_input.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
