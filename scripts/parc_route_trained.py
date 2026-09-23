#!/usr/bin/env python3
"""Route with a trusted locally trained PARC artifact, not the paper artifact."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/parc_router"))
from parc_training import route_trained
from parc_runtime import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True, help="Integrity check; load only artifacts you trust")
    parser.add_argument("--task-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = json.loads(args.task_input.read_text(encoding="utf-8"))
    result = route_trained(values, args.model, args.model_sha256)
    atomic_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("status", "queries", "model_sha256")}))


if __name__ == "__main__":
    main()
