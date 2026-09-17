# Benchmark Sources and Provenance

PARC does not redistribute benchmark data. Users should obtain each dataset
from its upstream project, respect the upstream terms, and keep the pinned
revision and file hashes recorded below when reproducing the paper setup.

## Download links

Use the pinned data revisions below. No dataset download is triggered by the
quick start or CI. The hub links were checked against the official authors'
repositories and Hugging Face metadata.

| Family | Data source | Required files |
| --- | --- | --- |
| BAMBOO | [Official data directory](https://github.com/RUCAIBox/BAMBOO/tree/f230f206148396c12774ef3a632df291d96dc6ab/datasets) | `prompt.json` from the repository root; four `datasets/{abshallu,senhallu}_{4k,16k}.jsonl` files |
| ETHIC | [Hugging Face](https://huggingface.co/datasets/dmis-lab/ETHIC/tree/34d9a5c6ffd9b06b1eca75d2e7a437bc96d628c4) | `recalling.zip`, `attributing.zip`; keep both archives zipped |
| HELMET | [Hugging Face original v1](https://huggingface.co/datasets/princeton-nlp/HELMET/tree/c5b85e7f0d954ffe71b3fe5b6d1da17a11094b1b) | Extract `data.tar.gz`; retain the four KILT test/demo files listed below |
| LongTableBench | [Official data directory](https://github.com/liyaooi/LongTableBench/tree/8db45ac102632cba3bcc1393e1023f0689da7d1c/datasets) | Clone the pinned repository, including `datasets/`, `eval/` and `category/` |

For BAMBOO and LongTableBench, no author-confirmed Hugging Face or ModelScope
mirror has been verified for these revisions. Use the official GitHub sources;
a same-name hub dataset is not evidence of identical data or licensing.

Direct archive links:

- ETHIC [recalling.zip](https://huggingface.co/datasets/dmis-lab/ETHIC/resolve/34d9a5c6ffd9b06b1eca75d2e7a437bc96d628c4/recalling.zip)
- ETHIC [attributing.zip](https://huggingface.co/datasets/dmis-lab/ETHIC/resolve/34d9a5c6ffd9b06b1eca75d2e7a437bc96d628c4/attributing.zip)
- HELMET [v1 data.tar.gz](https://huggingface.co/datasets/princeton-nlp/HELMET/resolve/c5b85e7f0d954ffe71b3fe5b6d1da17a11094b1b/data.tar.gz)

The original HELMET archive is 11,271,916,108 bytes (about 11.27 GB), before
extraction. Its recorded SHA256 is
`9d693981aa3c065b8b2ff82ddf946141cdc4ece4524f18bff6f3fbd2a86982d9`.
HELMET's newer v2 release is not interchangeable with this archive. Do not
download it by an unpinned `main` URL when matching the paper setup.

## Pinned upstream code and evaluators

| Family | Upstream source | Pinned code/data reference | Official evaluation reference |
| --- | --- | --- | --- |
| BAMBOO | <https://github.com/RUCAIBox/BAMBOO> | code commit `f230f206148396c12774ef3a632df291d96dc6ab` | `evaluate.py`, binary F1 |
| ETHIC | <https://github.com/dmis-lab/ETHIC> | code commit `b26c399e494f9a44c97115a2b37503eeba87f1af`; Hugging Face revision `34d9a5c6ffd9b06b1eca75d2e7a437bc96d628c4` | `utils.py`, sample-level set F1 and official mean |
| HELMET | <https://github.com/princeton-nlp/HELMET> | code commit `af609c4d51b97fc35012099380aa889da961c42d`; original v1 data revision `c5b85e7f0d954ffe71b3fe5b6d1da17a11094b1b` | official HELMET evaluator; QA F1 is the primary quality metric |
| LongTableBench | <https://github.com/liyaooi/LongTableBench> | code commit `8db45ac102632cba3bcc1393e1023f0689da7d1c` | `eval/result_process.py`, structured-answer macro F1 |

## Exact paper evaluation inventory

| Family | Task configurations | Record/conversation evaluations | Scored units |
| --- | --- | ---: | ---: |
| BAMBOO | `abshallu_4k`, `abshallu_16k`, `senhallu_4k`, `senhallu_16k` (200 each) | 800 | 800 |
| ETHIC | `Recalling` (662), `Attributing` (333) | 995 | 995 |
| HELMET | `kilt_hotpotqa` (2,961), `kilt_popqa_3` (882) | 3,843 | 3,843 |
| LongTableBench | 7 formats x 3 length settings x 2 turn types = 42 | 6,671 | 12,397 |
| Total | 50 | 12,309 | 18,035 |

LongTableBench formats are `markdown`, `html`, `json`, `latex`, `sql`, `xml`,
and `csv`; length directories are `32k`, `8k`, and `inf`; turn types are
`multi` and `single`. Question files follow
`datasets/questions/{32k,8k,inf}/longtablebench_{multi,single}.json`.
Counts include format-specific evaluations and separate scored turns for
multi-turn cases. These are not counts of distinct source questions.

HELMET requires these files under the configured KILT directory:

```text
hotpotqa-dev-multikilt_1000_k1000_dep3.jsonl
hotpotqa-train-multikilt_1000_k3_dep3.jsonl
popqa_test_1000_k1000_dep6.jsonl
popqa_test_1000_k3_dep6.jsonl
```

The `k3` files supply released demonstrations. Their answers are part of the
official input prompt, not untouched test labels. Adapters do not consume
test-answer values or expose demonstration answers as selector metadata.

"Complete" means all eligible records in this paper inventory, not every
task offered by the upstream projects. Never replace it with successful
samples or the 35-task descriptive cross-model subset. Follow the
[workflow](reproduction_workflow.md) to validate the inventory and file layout.

## Licensing notes

- LongTableBench documents its dataset terms in the upstream README (CC BY
  4.0 was recorded for the pinned release).
- BAMBOO's pinned README claims MIT, but the pinned tree did not contain a
  separate license file; users should verify the upstream repository before
  redistribution.
- ETHIC's Hugging Face metadata records CC BY 4.0 for the dataset; the pinned
  code tree did not contain a separate code license file.
- HELMET's code and data terms follow the upstream repository and its original
  v1 release. The later v2 archive is not the paper's pinned data revision.

The PARC software license is independent from the licenses of these datasets.
The PARC code is released under the MIT License in the repository root; that
license does not grant rights to redistribute the benchmark data.

## Reproduction boundary

This release records source provenance and the retrieval-only PARC routing
interface. It intentionally does not include benchmark questions, answers,
labels, raw contexts, model checkpoints, generated predictions, or per-example
scores. The benchmark adapters are included. Complete end-to-end reproduction
still requires the unreleased retrieval materializer and generation/scoring
orchestration, upstream data and model assets, and an independent full replay.

## 中文说明

PARC 不在仓库中重新分发 benchmark 原始数据。复现实验时应从上游项目
下载数据，遵守数据许可证，并使用表中固定的代码提交、数据版本和评测脚本。
四个 benchmark family 为 BAMBOO、ETHIC、HELMET 和 LongTableBench。
