#!/usr/bin/env python3
"""Create synthetic development inputs and artificial labels for a CPU smoke test."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/parc_router"))
import parc_runtime as runtime
from parc_training import prepare_data


def make_example(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    template = json.loads((ROOT / "examples/parc_synthetic_task_input.json").read_text())[0]
    groups = []
    for task_index in range(4):
        queries, labels = [], {}
        for index in range(8):
            query = copy.deepcopy(template)
            query["query_key"] = f"synthetic-{task_index}-{index}"
            query["question"] = f"Which synthetic item {index} is listed first?"
            query["chunks"] = [f"synthetic chunk {i}" for i in range(20)]
            for name, observation in query["methods"].items():
                observation["kept_indices"] = list(range(20 if name == runtime.DENSE_METHOD else 9))
            scores = {name: 0.5 for name in query["methods"]}
            for candidate, name in enumerate(runtime.QUALITY_CANDIDATES[1:], 1):
                query["methods"][name]["retrieved_indices"] = list(reversed(range(8)))
                scores[name] = 0.5 + 0.1 * ((candidate + index) % 3 - 1)
            queries.append(query)
            labels[query["query_key"]] = scores
        retrieval_name, label_name = f"parc_synthetic_retrieval_{task_index}.json", f"parc_synthetic_labels_{task_index}.json"
        runtime.atomic_json(output_dir / retrieval_name, queries)
        runtime.atomic_json(output_dir / label_name, labels)
        groups.append({"task": f"synthetic_task_{task_index}", "family": f"synthetic_family_{task_index // 2}",
                       "retrieval": retrieval_name, "labels": label_name})
    group_path = output_dir / "parc_training_groups.json"
    runtime.atomic_json(group_path, {"role": "development", "groups": groups})
    result = prepare_data(group_path, output_dir / "parc_development.npz")
    result["scope"] = "synthetic software smoke test; not benchmark evidence"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(make_example(args.output_dir), indent=2))
