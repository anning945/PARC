# PARC Reproduction Workflow

This document describes the complete public workflow and its boundaries. The
repository contains the prediction-free adapter and routing stages. Benchmark
data, retriever/reranker checkpoints, generator checkpoints, and official
evaluation data remain upstream dependencies.

## 1. Create the environment

For routing only:

```bash
python -m pip install -r requirements-lock.txt
```

For benchmark parsing and prompt reconstruction:

```bash
python -m pip install -r requirements-benchmark.txt
```

The conda equivalent is `environment.yml`.

## 2. Obtain the four benchmark families

Download or clone the pinned upstream revisions listed in
`docs/benchmarks.md`. Do not commit the downloaded datasets to this
repository. Arrange local files as follows:

```text
data/benchmarks/
  BAMBOO/                 # prompt.json and datasets/*.jsonl
  ETHIC/                  # recalling.zip and attributing.zip
  HELMET/data/kilt/       # the original v1 KILT files and demos
  LongTableBench/         # datasets/ and eval/ from the pinned release
```

The paths can be overridden with `PARC_BAMBOO_ROOT`, `PARC_ETHIC_ROOT`,
`PARC_HELMET_ROOT`, `PARC_LONGTABLE_ROOT`, or `PARC_DATA_ROOT`. The example
configuration is `configs/benchmark_paths.example.json`.

## 3. Run the prediction-free adapter check

This stage reads only the fields needed to reconstruct prompts and selector
inputs. It does not read untouched test answers, predictions, F1 values, or
official scores. It verifies all four family counts, opaque key uniqueness,
chunk construction, and the public selector schema.

```bash
PYTHON=/path/to/python bash scripts/run_parc_schema_check.sh
```

The output is written under `outputs/parc_schema_check/` and is intentionally
ignored by Git. A successful run emits `"status": "PASS"` and a
`pipeline_status.json` attestation.

To also materialize selector-only JSONL for an external retriever, set an
output path under the ignored `outputs/` directory:

```bash
PARC_SELECTOR_STUB_OUTPUT=outputs/parc_schema_check/selector_stubs.jsonl \
  PYTHON=/path/to/python bash scripts/run_parc_schema_check.sh
```

The JSONL deliberately contains no family name, task name, answer, label,
prediction, score, or model identity.

## 4. Materialize retrieval observations

The public adapter emits selector stubs with `query_key`, `question`, and
`chunks`. A retrieval implementation must add the `methods` object described
in `docs/public_input_schema.md`. For every query, materialize Dense, safe,
and all nine compressed candidates before invoking PARC. Retrieval code is
not bundled because embedding and reranker checkpoints are model- and
hardware-specific.

## 5. Run PARC routing

```bash
python src/parc_router/parc_runtime.py \
  --model artifacts/parc_selector/parc_frozen_selector.joblib \
  --task-input /path/to/query_group_with_methods.json \
  --output routed.json
```

The runtime rejects benchmark identity, model identity, answers, labels,
predictions, F1, and quality scores. It returns one selected route per query
and an auditable routing summary.

## 6. Generate and evaluate externally

Use the selected route to reconstruct the family-specific prompt with the
adapter's renderer, then run the chosen generator with the benchmark's
official decoding contract. Score generated answers only with the official
upstream evaluator listed in `docs/benchmarks.md`. Keep generated answers and
per-example scores outside this repository.

## 7. Verify the release

```bash
PYTHON=/path/to/python bash scripts/verify_parc_release.sh
PYTHONPATH=src python -m unittest discover -s tests -v
```

The current repository supports a complete routing-only replay and a complete
prediction-free schema replay. It does not claim one-command reproduction of
the paper's generator-level F1 tables until upstream data, retrieval assets,
generator checkpoints, and independent end-to-end evidence are supplied.
