# Benchmark Sources and Provenance

PARC does not redistribute benchmark data. Users should obtain each dataset
from its upstream project, respect the upstream terms, and keep the pinned
revision and file hashes recorded below when reproducing the paper setup.

The current manuscript reports 35 tasks and 11,997 scored units from ETHIC,
HELMET, and LongTableBench. Its exact manifest, aggregate definitions, and
provenance are in [current paper results](current_paper_results.md).
The separate [development inventory](development_inventory.md) identifies
the 12 training tasks. The [runner data layout](#runner-data-layout) lists
the input dependencies for executing the supplied pipeline.

## Download links

Use the pinned data revisions below. No dataset download is triggered by the
quick start or CI. The hub links were checked against the official authors'
repositories and Hugging Face metadata.

| Family | Data source | Required files |
| --- | --- | --- |
| ETHIC | [Hugging Face](https://huggingface.co/datasets/dmis-lab/ETHIC/tree/34d9a5c6ffd9b06b1eca75d2e7a437bc96d628c4) | `attributing.zip`; keep the archive zipped |
| HELMET | [Hugging Face original v1](https://huggingface.co/datasets/princeton-nlp/HELMET/tree/c5b85e7f0d954ffe71b3fe5b6d1da17a11094b1b) | Extract `data.tar.gz`; HotpotQA test/demo files are under `data/kilt/` |
| LongTableBench | [Official data directory](https://github.com/liyaooi/LongTableBench/tree/8db45ac102632cba3bcc1393e1023f0689da7d1c/datasets) | Clone the pinned repository, including `datasets/`, `eval/` and `category/` |

LongTableBench is distributed through its official GitHub dataset directory.

Direct archive links:

- ETHIC [attributing.zip](https://huggingface.co/datasets/dmis-lab/ETHIC/resolve/34d9a5c6ffd9b06b1eca75d2e7a437bc96d628c4/attributing.zip)
- HELMET [v1 data.tar.gz](https://huggingface.co/datasets/princeton-nlp/HELMET/resolve/c5b85e7f0d954ffe71b3fe5b6d1da17a11094b1b/data.tar.gz)

The original HELMET archive is 11,271,916,108 bytes (about 11.27 GB), before
extraction. Its recorded SHA256 is
`9d693981aa3c065b8b2ff82ddf946141cdc4ece4524f18bff6f3fbd2a86982d9`.
The paper setup uses this pinned original v1 archive.

## Pinned upstream code and evaluators

| Family | Upstream source | Pinned code/data reference | Official evaluation reference |
| --- | --- | --- | --- |
| ETHIC | <https://github.com/dmis-lab/ETHIC> | code commit `b26c399e494f9a44c97115a2b37503eeba87f1af`; Hugging Face revision `34d9a5c6ffd9b06b1eca75d2e7a437bc96d628c4` | `utils.py`, sample-level set F1 and official mean |
| HELMET | <https://github.com/princeton-nlp/HELMET> | code commit `af609c4d51b97fc35012099380aa889da961c42d`; original v1 data revision `c5b85e7f0d954ffe71b3fe5b6d1da17a11094b1b` | official HELMET evaluator; QA F1 is the primary quality metric |
| LongTableBench | <https://github.com/liyaooi/LongTableBench> | code commit `8db45ac102632cba3bcc1393e1023f0689da7d1c` | `eval/result_process.py`, structured-answer macro F1 |

## Paper evaluation inventory

| Family | Task configurations | Tasks | Scored units |
| --- | --- | ---: | ---: |
| ETHIC | `Attributing` | 1 | 333 |
| HELMET | `kilt_hotpotqa` | 1 | 2,961 |
| LongTableBench | Named format/length/turn configurations in the paper manifest | 33 | 8,703 |
| Total | | 35 | 11,997 |

LongTableBench formats are `markdown`, `html`, `json`, `latex`, `sql`, `xml`,
and `csv`; length directories are `32k`, `8k`, and `inf`; turn types are
`multi` and `single`. Question files follow
`datasets/questions/{32k,8k,inf}/longtablebench_{multi,single}.json`.
The exact combinations are listed in
[`parc_paper35_inventory.json`](../configs/parc_paper35_inventory.json).
Counts include format-specific evaluations and separate scored turns for
multi-turn cases.

## Runner data layout

The supplied runner executes the inventory in
[`parc_data_inventory.json`](../configs/parc_data_inventory.json), then
[`parc_summarize_paper35.py`](../scripts/parc_summarize_paper35.py) aggregates
the paper's fixed task IDs. Its execution inventory includes additional input
configurations beyond the paper report. Prepare these dependencies as well:

- BAMBOO: [pinned official repository](https://github.com/RUCAIBox/BAMBOO/tree/f230f206148396c12774ef3a632df291d96dc6ab), including `prompt.json`, `datasets/{abshallu,senhallu}_{4k,16k}.jsonl`, and `evaluate.py`.
- ETHIC: [recalling.zip](https://huggingface.co/datasets/dmis-lab/ETHIC/resolve/34d9a5c6ffd9b06b1eca75d2e7a437bc96d628c4/recalling.zip), alongside `attributing.zip`.
- HELMET: the PopQA test/demo files, alongside the HotpotQA files.
- LongTableBench: the complete pinned dataset directory used by the adapters.

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

The runner checks completion against its execution inventory; the paper
aggregation uses the 35-task manifest. Follow the
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

This release records source provenance and includes retrieval, routing,
generation, and official-scoring orchestration. It intentionally does not
include benchmark questions, answers, labels, raw contexts, model checkpoints,
generated predictions, or per-example scores. Complete reproduction therefore
still requires the pinned upstream data and model assets plus an independent
full replay, but no final pipeline stage is intentionally withheld.

## 中文说明

PARC 不在仓库中重新分发 benchmark 原始数据。复现实验时应从上游项目
下载数据，遵守数据许可证，并使用表中固定的代码提交、数据版本和评测脚本。
当前论文使用 ETHIC、HELMET 和 LongTableBench 的 35 个任务，共 11,997 个
评分单元，清单与指标见[当前论文结果](current_paper_results.md)。默认运行器
按执行清单完成检索、生成和评分，再按论文清单汇总结果；所需额外输入配置见
[运行数据布局](#runner-data-layout)。
