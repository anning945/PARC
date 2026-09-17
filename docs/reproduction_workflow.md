# PARC Reproduction Workflow

This document describes the available public stages and their boundaries. The
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

Use the direct download links and pinned upstream revisions in
[the dataset guide](benchmarks.md). Do not commit the downloaded datasets to this
repository. Arrange local files as follows:

```text
data/benchmarks/
  BAMBOO/                 # prompt.json and datasets/*.jsonl
  ETHIC/                  # recalling.zip and attributing.zip
  HELMET/data/kilt/       # the original v1 KILT files and demos
  LongTableBench/         # datasets/, eval/ and category/ from pinned release
```

The paths can be overridden with `PARC_BAMBOO_ROOT`, `PARC_ETHIC_ROOT`,
`PARC_HELMET_ROOT`, `PARC_LONGTABLE_ROOT`, or `PARC_DATA_ROOT`. The example
configuration is `configs/benchmark_paths.example.json`; it documents the
layout and is not read automatically by the shell script. For example:

```bash
PARC_DATA_ROOT=/path/to/benchmarks bash scripts/run_parc_schema_check.sh
```

`PARC_HELMET_ROOT` points directly to `HELMET/data/kilt/`, not the archive
or repository root. Extract the pinned HELMET v1 archive so the four required
JSONL files are directly inside that directory. Keep ETHIC archives zipped.

## 3. Run the prediction-free adapter check

This stage reads only the fields needed to reconstruct prompts and selector
inputs. It does not read untouched test answers, predictions, F1 values, or
official scores. HELMET's released demonstration answers are read only to
reconstruct the official prompt. It verifies all four family counts, opaque
key uniqueness, chunk construction, and the public selector schema. The
splitter uses a character-based size of 300 and overlap of 50, not token
budgets; its separators and behavior are frozen in the adapter code.

```bash
bash scripts/run_parc_schema_check.sh
```

The output is written under `outputs/parc_schema_check/` and is intentionally
ignored by Git. A successful run emits `"status": "PASS"` and a
`pipeline_status.json` attestation. The expected totals are 50 tasks, 12,309
record/conversation evaluations, and 18,035 scored units/selector queries.
The count/schema check is not a cryptographic proof of dataset identity and
does not generate answers or score F1; retain the upstream revisions too.

To also materialize selector-only JSONL for an external retriever, set an
output path under the ignored `outputs/` directory:

```bash
PARC_SELECTOR_STUB_OUTPUT=outputs/parc_schema_check/selector_stubs.jsonl \
  bash scripts/run_parc_schema_check.sh
```

The JSONL deliberately contains no family name, task name, answer, label,
prediction, score, or model identity.

## 4. Materialize retrieval observations

The public adapter emits selector stubs with `query_key`, `question`, and
`chunks`. A retrieval implementation must add the `methods` object described
in `docs/public_input_schema.md`. For every query, materialize Dense, safe,
and all nine compressed candidates before invoking PARC. **This retrieval
materializer is not yet included in the public release.** Downloading model
weights alone does not fill that code gap. The synthetic fixture illustrates
the interface, not a substitute retrieval algorithm for paper reproduction.

## 5. Run PARC routing

```bash
python src/parc_router/parc_runtime.py \
  --model artifacts/parc_selector/parc_frozen_selector.joblib \
  --task-input /path/to/query_group_with_methods.json \
  --output outputs/routed.json
```

The runtime rejects benchmark identity, model identity, answers, labels,
predictions, F1, and quality scores. It returns one selected route per query
and an auditable routing summary.

## 6. Generate and evaluate externally (not included)

Use the selected route to reconstruct the family-specific prompt with the
adapter's renderer, then run the chosen generator with the benchmark's
official decoding contract. Score generated answers only with the official
upstream evaluator listed in `docs/benchmarks.md`. Keep generated answers and
per-example scores outside this repository. The adapter includes prompt
renderers, but this release does not provide a generator runner or a wrapper
for the official scorers. Test labels belong only in this scoring stage.

## 7. Verify the release

```bash
bash scripts/verify_parc_release.sh
```

Set `PYTHON=/path/to/python` when using a non-default interpreter. Verification
checks source/artifact integrity, the self-test, unit tests and synthetic
routing. It does not download datasets or run the full schema check. Expected
final output: `PASS PARC release verification`.

The runtime supports routing from supplied retrieval observations, and the
adapters support a full paper-inventory schema check from upstream data.
Independent reproduction of the paper's generator-level F1 tables remains
pending retrieval and evaluation code plus a clean end-to-end replay.
