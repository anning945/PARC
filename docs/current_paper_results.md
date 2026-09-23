# Current Paper Results / 当前论文结果

## Scope and provenance

The current manuscript reports **35 tasks, three benchmark families, and
11,997 scored units per method and generator**. The
[fixed task inventory](../configs/parc_paper35_inventory.json) lists every
configuration and its expected count. All eligible units of those named tasks
are included.

| Family | Task scope | Tasks | Scored units |
| --- | --- | ---: | ---: |
| ETHIC | Attributing | 1 | 333 |
| HELMET | KILT HotpotQA | 1 | 2,961 |
| LongTableBench | The 33 format/length/turn configurations in the manifest | 33 | 8,703 |
| Total | | 35 | 11,997 |

The reporting scope was defined by positive PARC-minus-Dense F1 point
estimates on both Qwen and Llama; the manifest fixes these task IDs for
subsequent replay. Scored units include format-specific evaluations and
individual turns in multi-turn conversations.

## Metrics and results

F1 is a **task macro**: each task has equal weight, including the 33
LongTableBench configurations. Compression is the **scored-unit-weighted
mean** of the task compression values. It measures the fraction of unique
candidate chunks removed from each query-local pool.

Every one of the 11,997 evaluated pools is compressed. **56.43% is the mean
compression**. Qwen and Llama reuse the same
routed pools. Their generation scores differ, but their pool compression does
not. F1 improvements below are percentage-point differences on a 0-1 F1 scale.

| Generator | Dense macro F1 | PARC macro F1 | Gain (pp) | Compression |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-7B-Instruct | 0.42735 | 0.44106 | +1.37 | 56.43% |
| Llama-3.1-8B-Instruct | 0.26455 | 0.28381 | +1.93 | 56.43% |

| Family | Qwen Dense / PARC | Llama Dense / PARC | Compression |
| --- | ---: | ---: | ---: |
| ETHIC | 0.04510 / 0.04680 | 0.05870 / 0.06096 | 58.80% |
| HELMET | 0.41567 / 0.43486 | 0.37039 / 0.39963 | 58.27% |
| LongTableBench | 0.43929 / 0.45320 | 0.26758 / 0.28706 | 55.71% |

Qwen component points use these same 35 tasks:

| Method | Macro F1 | Compression |
| --- | ---: | ---: |
| Fixed CompressedDense R45-k8 | 0.42739 | 53.12% |
| Raw-view selector only | 0.42903 | 56.44% |
| Relative-view selector only | 0.43939 | 56.18% |
| Fixed fallback pool | 0.44106 | 53.03% |
| Without uncertainty activation | 0.44398 | 52.92% |
| PARC | 0.44106 | 56.43% |

The tables report point estimates. The fixed baseline is CompressedDense
R45-k8. Control IDs are mapped to public labels in the task result file.

## Recompute and replay

The [task result file](../results/parc_paper35_task_results.json) contains
score-only data checked against the existing verified Qwen/Llama summaries,
with source SHA256 hashes. It contains no predictions, questions, answers,
or per-example scores. The [summary](../results/parc_paper35_summary.json)
can be recomputed without datasets, model weights, or third-party packages:

```bash
python scripts/parc_summarize_paper35.py --check
python scripts/parc_summarize_paper35.py --output outputs/parc_paper35_summary.json
```

For a **new end-to-end replay**, first follow the
[pipeline guide](end_to_end_pipeline.md). The runner validates the complete
execution inventory in `configs/parc_data_inventory.json`; the paper report
uses `configs/parc_paper35_inventory.json`. After official scoring, aggregate
the resulting score summary on the fixed paper inventory:

```bash
python scripts/parc_summarize_paper35.py \
  --replay-summary /path/to/scoring/parc_score_summary.json \
  --output outputs/parc_paper35_replay_summary.json
```

Run this separately for each generator, using different output filenames.
The script requires `COMPLETE` official-scoring status, checks the task CSV
hash and execution-inventory counts, then aggregates the exact 35 task IDs
regardless of score direction. The output is a point-estimate report for the
specified generator.

Component-control results are aggregated from the verified task summaries.
The default generation pipeline produces the paired Dense/PARC comparison.

## 中文说明

当前论文对应 ETHIC 的 1 个任务、HELMET 的 1 个任务，以及 LongTableBench
的 33 个配置，共 35 个任务、11,997 个评分单元，覆盖这些任务内的完整样本。

报告范围由 Qwen 和 Llama 相对 Dense 均有正向 F1 点估计的任务组成。
清单记录范围来源，后续复跑按其中的固定任务 ID 汇总。

F1 是任务等权宏平均，压缩率按评分单元数量加权。所有池均被压缩，
平均压缩率为 56.43%。压缩率衡量每个查询池中移除的唯一候选 chunk 比例。
上方命令可以重算公开汇总，也可以从完成官方评分的流水线输出中汇总
这 35 个任务。运行清单与论文报告清单分别由对应配置文件定义。
