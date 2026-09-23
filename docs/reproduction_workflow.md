# PARC Reproduction Workflow

This document describes the public end-to-end stages and their boundaries. The
repository contains prediction-free adapters and routing, selector training,
retrieval materialization, prompt packing, generation, and official scoring.
Benchmark data, retriever/reranker checkpoints, generator checkpoints, and
official evaluation repositories remain upstream dependencies.

The commands below use `configs/parc_data_inventory.json` as the execution
inventory. After scoring, aggregate the paper's 35 tasks and 11,997 scored
units with the fixed-scope command in
[current paper results](current_paper_results.md#recompute-and-replay).

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

## 2. Prepare the runner inputs

Use the direct download links and pinned upstream revisions in
[the dataset guide](benchmarks.md#runner-data-layout). Arrange local files
under the ignored data directory as follows:

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
reconstruct the official prompt. It verifies the execution-inventory counts, opaque
key uniqueness, chunk construction, and the public selector schema. The
splitter uses a character-based size of 300 and overlap of 50, not token
budgets; its separators and behavior are frozen in the adapter code.

```bash
bash scripts/run_parc_schema_check.sh
```

The output is written under `outputs/parc_schema_check/` and is intentionally
ignored by Git. A successful run emits `"status": "PASS"` and a
`pipeline_status.json` attestation. Expected task, evaluation, and scored-unit
counts come from the execution inventory. The report checks data structure
and coverage; generated answers and F1 scores are produced in later stages.

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
`chunks`. The complete materializer adds the `methods` object described in
`docs/public_input_schema.md`. For every query, it materializes Dense, safe,
and all nine compressed candidates before invoking PARC. See the
[end-to-end guide](end_to_end_pipeline.md) and run
`python scripts/parc_materialize_retrieval.py --help` for the required dataset,
model, cache, output, and sharding arguments. The synthetic fixture remains
an interface test, not a substitute retrieval algorithm for paper evidence.

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

## 6. Generate and score

Use `scripts/parc_generate.py` to reconstruct family-specific prompts and run
paired Dense/PARC generation with prediction-blind packing, greedy decoding,
sharding, and resume. Use `scripts/parc_score.py` afterward to validate and load
the pinned official scorers. Test labels are read only in the scoring stage.
The [end-to-end guide](end_to_end_pipeline.md) explains configuration and the
complete pipeline command. Generated answers and per-example scores stay in
ignored output directories, not the public repository.

## 7. Verify the release

```bash
bash scripts/verify_parc_release.sh
```

Set `PYTHON=/path/to/python` when using a non-default interpreter. Verification
checks source/artifact integrity, the self-test, unit tests and synthetic
routing. It does not download datasets or run the full schema check. Expected
final output: `PASS PARC release verification`.

The runtime supports routing from supplied retrieval observations, and the
adapters support execution-inventory schema checks from upstream data.
The repository now contains the code needed for generator-level F1 replay;
independent reproduction still requires a clean run with pinned upstream data,
model assets, and evaluator revisions.
