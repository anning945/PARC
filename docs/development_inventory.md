# Development and Evaluation Inventory

This document identifies the data used to fit the frozen PARC selector. The
machine-readable [development inventory](../configs/parc_data_inventory.json)
records task metadata, source revisions, and input hashes. The
[paper evaluation inventory](../configs/parc_paper35_inventory.json) defines
the 35 tasks used in the current report.

## Development tasks

All 5,763 queries in these named subsets were used for selector fitting and
policy selection. The paper evaluation uses
ETHIC, HELMET, and LongTableBench, separate from these development families.

| Family | Task identifier | Queries | Source subset |
| --- | --- | ---: | --- |
| LongBench | `longbench_triviaqa` | 200 | Every official `triviaqa` test row |
| InfiniteBench | `infinitebench_longbook_qa_eng` | 351 | Every nonblank `longbook_qa_eng.jsonl` row |
| LV-Eval | `hotpotwikiqa_mixup` | 620 | All five length levels, 124 rows each |
| LV-Eval | `multifieldqa_en_mixup` | 505 | All five length levels, 101 rows each |
| LV-Eval | `multifieldqa_zh_mixup` | 665 | All five length levels, 133 rows each |
| LV-Eval | `cmrc_mixup` | 1,000 | All five length levels, 200 rows each |
| LV-Eval | `factrecall_en` | 1,000 | All five length levels, 200 rows each |
| LV-Eval | `factrecall_zh` | 1,000 | All five length levels, 200 rows each |
| L-Eval | `leval_financial_qa` | 52 | All instruction/output pairs from six official F1 documents |
| L-Eval | `leval_legal_contract_qa` | 130 | All pairs from 20 official F1 documents |
| L-Eval | `leval_multidoc_qa` | 136 | All pairs from 20 official F1 documents |
| L-Eval | `leval_natural_question` | 104 | All pairs from 20 official F1 documents |
| Total | 12 tasks | 5,763 | All eligible rows in the declared subsets |

LV-Eval length levels are `16k`, `32k`, `64k`, `128k`, and `256k`, in that
order; rows retain source order within each level. Each task ZIP contains
`{task}/{task}_{length}.jsonl`. L-Eval files are under
`LEval-data/Open-ended-tasks/`: retain documents with `evaluation='f1'`,
then expand every aligned `instructions`/`outputs` pair in source order.
The L-Eval counts above are expanded queries, not document counts.

## Pinned upstream sources

| Family | Data source | Recorded evaluator/code revision |
| --- | --- | --- |
| LongBench | [Hugging Face, pinned](https://huggingface.co/datasets/THUDM/LongBench/tree/5e628be450b7e67fb7ae6e201bd6d8f7056f7672), `data.zip`, member `data/triviaqa.jsonl` | `2e00731f8d0bff23dc4325161044d0ed8af94c1e` |
| InfiniteBench | [Hugging Face, pinned](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/tree/90f0394333616266d9fe85824ceaf505093cbaa5), `longbook_qa_eng.jsonl` | `51d9b37b0f1790ead936df2243abbf7f0420e439` |
| LV-Eval | [Hugging Face, pinned](https://huggingface.co/datasets/Infinigence/LVEval/tree/86a3b0e6f2266281d481bcb46a0bda5b511cffb0), the six task ZIPs above | `63e7ae939bd347c76b2cdb1cebfb216faca26897` |
| L-Eval | [Official GitHub data, pinned](https://github.com/OpenLMLab/LEval/tree/cd34b050269148aed75acbbe4a599873ad0f37e9/LEval-data/Open-ended-tasks) | `cd34b050269148aed75acbbe4a599873ad0f37e9` |

The JSON inventory includes the recorded archive/member hashes, normalized
development-file hashes, sample-order hashes, and LOTO target-index hashes.
Source hashes help identify exact inputs; hashes alone do not reconstruct a
missing preprocessing implementation or demonstrate the absence of semantic
overlap between different benchmarks. Upstream data licenses still apply.

## Split and training use

LOTO holds out one of the 12 tasks; LOFO holds out one of the four families.
The [frozen model manifest](../artifacts/parc_selector/parc_frozen_selector_manifest.json)
records each fold's training/target counts and index hashes. The final policy
uses the complete out-of-fold predictions from both schemes, then fits the
two feature views on all 5,763 development queries. See the exact
[selection protocol](selection_protocol.md).

The current manuscript reports 35 tasks and 11,997 scored units from ETHIC,
HELMET, and LongTableBench. The
[manifest](../configs/parc_paper35_inventory.json) and
[results guide](current_paper_results.md) identify all task IDs, source
provenance, and aggregation rules. The development data and frozen selector
are unchanged.

## Reproduction boundary

These are metadata and provenance documents, not new benchmark results or a
new data split. The frozen model and runtime are unchanged. The repository's
[release boundary](release_boundary.md) still applies. The
[training entry point](selector_training.md) now prepares features from
materialized retrieval inputs and development labels, selects a policy, and
fits both views. Upstream retrieval construction and generation/scoring are
still needed for an independent end-to-end F1 reproduction.

## 中文说明

开发集包括上述四个 family 的 12 个任务，共 5,763 个查询，用于选择器训练
与策略选择。当前论文评测覆盖 ETHIC、HELMET 和 LongTableBench 的 35 个
任务，共 11,997 个评分单元。开发清单与论文评测清单分别记录任务、数量、
来源规则、版本和哈希；各任务结果及汇总方式见当前论文结果说明。
