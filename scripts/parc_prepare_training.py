#!/usr/bin/env python3
"""Prepare PARC development features from retrieval inputs and separate F1 labels."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/parc_router"))
from parc_training import prepare_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_data(args.groups, args.output), indent=2))


if __name__ == "__main__":
    main()
