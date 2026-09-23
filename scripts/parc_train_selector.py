#!/usr/bin/env python3
"""Train a PARC selector without changing the released paper artifact."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/parc_router"))
from parc_training import train_selector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection", choices=("cross-validated", "published"), default="cross-validated")
    parser.add_argument("--verify-paper-inputs", action="store_true")
    args = parser.parse_args()
    result = train_selector(args.data, args.output_dir, args.selection, args.verify_paper_inputs)
    print(json.dumps({key: result[key] for key in ("status", "training_queries", "model_sha256", "selection_mode")}, indent=2))


if __name__ == "__main__":
    main()
