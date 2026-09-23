# Contributing

Use Python 3.10.20 and install `requirements-benchmark.txt`. Changes should
preserve the frozen input contract, model hashes and route semantics unless
they deliberately define a separately evaluated method version.

## Tests

```bash
python -m unittest discover -s tests -v
bash scripts/run_parc_self_test.sh
bash scripts/run_parc_example.sh
```

All these checks use synthetic inputs or released metadata. They do not
download benchmarks, call an LLM, or establish a new F1 result. Full dataset
schema checks are separate; see [the workflow](docs/reproduction_workflow.md).
CI runs the release checks on Linux and macOS, including a source archive
without Git metadata. Linux/macOS here describe the CI targets, not a claim
that all possible accelerator environments have been tested.

## Release checksums

Root `SHA256SUMS` covers versioned release files, excluding itself. It must
never include `.git`, local data, environments, caches or generated outputs.
The separate artifact checksum manifest records the unchanged frozen model.
Integrity hashes detect file changes; they are not a cryptographic signature
or proof of scientific reproducibility.

After reviewing changes, stage the intended files explicitly so new paths
are visible to Git. Regenerate and verify the source manifest:

```bash
git add <reviewed-files>
python scripts/parc_release_checksums.py update
git add SHA256SUMS
bash scripts/verify_parc_release.sh
git diff --cached --check
```

The update command reads `git ls-files`, not a recursive filesystem scan.
It checks the artifact manifest before writing, so source maintenance cannot
silently bless changed model weights. Verification does not require `.git`
for a downloaded source ZIP. In a Git checkout it also checks that all
tracked release files appear in the manifest.

Do not regenerate checksums just to suppress an unexpected download or model
integrity failure. Compare against the published commit first.

## Issues and pull requests

Include the commit ID, operating system, Python/package versions, failing
command and expected behavior. Prefer a minimal synthetic input. Do not post
access tokens, private server paths, benchmark test answers, generations or
licensed raw data. Preserve upstream dataset attribution and license terms.

Retrieval materialization, generation, scoring, selector fitting and
final-policy selection are included. The
[current paper scope](docs/current_paper_results.md) uses a fixed manifest of
35 tasks. New runs have separate output files and retain this manifest for
aggregation. New components need documented parameters and tests.
