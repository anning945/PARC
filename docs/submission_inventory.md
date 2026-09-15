# GitHub Submission Inventory / GitHub 提交清单

## Upload now / 当前可以上传

| Path | Purpose |
| --- | --- |
| `README.md` | Bilingual project description and reproducibility boundary |
| `CITATION.cff` | Software citation metadata |
| `LICENSE` | MIT software license |
| `requirements.txt` | Runtime dependencies with artifact-compatible pins |
| `requirements-lock.txt` | Exact pip lock used for release verification |
| `environment.yml` | Reproducible Python 3.10 conda environment |
| `artifacts/parc_selector/` | Frozen routing model, manifest, and checksums |
| `src/parc_router/*.py` | Frozen v9 routing runtime and feature construction dependencies |
| `configs/parc_policy.json` | Route portfolio, policy constants, feature dimensions |
| `scripts/run_parc_self_test.sh` | Deterministic runtime self-test |
| `scripts/make_synthetic_fixture.py` | Synthetic retrieval-only schema fixture generator |
| `docs/*.md` | Input schema, benchmark provenance, release boundary, and submission inventory |
| `results/parc_quality_compression_pareto_verified.csv` | Score-only operating-point summary |
| `results/parc_task_quality_verified.csv` | Score-only task summary |
| `results/parc_release_summary.json` | Release audit collection summary |
| `examples/parc_synthetic_task_input.json` | Synthetic input for schema inspection only |
| `SHA256SUMS` | File integrity record |
| `scripts/verify_parc_release.sh` | Checksum, dependency, self-test, and example verification |

## Do not upload / 不要上传

- `id_rsa`, SSH commands, server addresses, absolute server paths, or tokens;
- raw benchmark files, prompt/question exports, answers, labels, or chunk text;
- formal prediction files, generated answers, per-example scores, and logs;
- model checkpoints, embedding/reranker caches, and GPU/runtime caches;
- the full private experiment directory;
- historical exploratory scripts, backups, temporary files, and Python caches;
- author contact metadata unless the authors intentionally publish it.

## Required before public release / 正式公开前必须补齐

1. Public benchmark download and preprocessing instructions.
2. Clean benchmark adapters and downstream generation/scoring code.
3. A clean-environment end-to-end routing run using the released model bundle.
4. A final credential/private-path scan and a fixed commit hash linked to the
   paper version.

The current directory intentionally stops before these gates. It is ready for
internal review and packaging, not yet a complete public reproducibility claim.
