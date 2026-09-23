# End-to-End Pipeline / 完整流水线

The repository includes runnable retrieval, routing, generation, and official
scoring stages. Raw datasets, model weights, and upstream evaluator code remain
external because their licenses are separate from PARC.

The pipeline validates the execution inventory in
`configs/parc_data_inventory.json`. After scoring, the manifest-based
aggregation reports the paper's 35 tasks and 11,997 scored units. See
[current paper results](current_paper_results.md#recompute-and-replay).

## Install

Create the Python 3.10.20 environment first. On CUDA hosts, install the PyTorch
wheel matching the local CUDA driver, then install the complete requirements:

```bash
python -m pip install -r requirements-full.txt
```

The paper environment recorded PyTorch 2.6.0, Transformers 5.9.0,
SentenceTransformers 5.5.1, and jieba 0.42.1. CUDA builds append a local build
tag to the PyTorch version. The Linux/macOS core CI checks avoid large model
dependencies; a separate Linux CPU job loads tiny randomly initialized local
models to test all three backend interfaces without downloading checkpoints.

## Configure

Copy `configs/parc_pipeline.example.json` to an ignored location under
`outputs/`, then replace every `/path/to/...` entry. Dataset layouts and pinned
revisions are documented in [benchmarks.md](benchmarks.md). The official source
tree must contain pinned checkouts named `BAMBOO`, `ETHIC`, and `HELMET`;
LongTableBench's scorer is loaded from `longtable_root`.

The model entries may be local directories or Hugging Face identifiers:

- `BAAI/bge-m3` for normalized Dense embeddings;
- `BAAI/bge-reranker-v2-m3` for cross-encoder scores;
- `Qwen/Qwen2.5-7B-Instruct` for the primary run, or
  `meta-llama/Meta-Llama-3.1-8B-Instruct` for transfer.

No API key is required for public models. A gated model such as Llama requires
the user's own Hugging Face access and acceptance of its license.

Set the optional `embedder_revision`, `reranker_revision`, and
`generator_revision` fields to immutable hub commit IDs for a repeatable run.
A null revision resolves the current hub version, which may differ from the
paper checkpoint. The runner resolves a snapshot once, hashes its actual
weight/tokenizer files, then loads that snapshot. Hashing large weights takes
time. Model loaders use `trust_remote_code=True`; use only trusted sources.

## Run All Stages

```bash
python scripts/parc_run_pipeline.py \
  --config outputs/parc_pipeline.json
```

Run or resume selected stages with `--stages retrieval`, `--stages generation`,
or `--stages scoring`. Each stage rejects a changed configuration when resuming.
Generated JSONL, predictions, caches, and per-run manifests stay under ignored
`outputs/` and `cache/` paths.

### 1. Retrieval and routing

`scripts/parc_materialize_retrieval.py` reconstructs the benchmark chunk
universe, BGE-M3 Dense scores, deterministic BM25 scores, RRF scores,
cross-encoder scores, the Dense baseline, the compressed Dense anchor, and all
nine PARC candidates. The defaults are embedding batch size 32, reranker batch
size 64, reranker maximum length 512, and float16-boundary quantization of
embeddings and reranker scores before float32 route arithmetic. It passes only `query_key`, question, chunks, and method
observations to the frozen selector. Answers, labels, generated text, quality
scores, benchmark identity, and generator identity do not enter routing.

Embedding caches bind model files, input text, environment, and batch settings.
Launch the standalone retrieval CLI with `PYTHONHASHSEED=0`; the orchestrator
sets this automatically to fix the archived BM25 term-accumulation order.
`--shard-count` and `--shard-index`
partition whole evaluations deterministically; `--limit` supports smoke tests.
A complete one-shard run covers every query in the execution inventory.

### 2. Generation

`scripts/parc_generate.py` produces paired Dense and PARC outputs with the same
tokenizer, prompt contract, 28,672-token hard limit, 28,416-token packing
target, and greedy decoding. The family-specific output limits are BAMBOO 32,
ETHIC 4,096, HELMET 20, and LongTableBench 1,024 tokens. Overflow handling is
prediction blind: history, context, and instruction segments are reduced in
that order by deterministic head-tail packing while protected questions and
answer cues remain unchanged. LongTableBench multi-turn prompts use each
method's real previous outputs.

Generation supports stable evaluation-level sharding and record-level resume.
A complete one-shard run contains paired Dense and PARC records for every
scored unit in the execution inventory. Generation uses prediction-blind inputs.

### 3. Official scoring

`scripts/parc_score.py` is the only stage that reads untouched test answers.
It verifies SHA256 values for the pinned BAMBOO, ETHIC, HELMET, and
LongTableBench evaluator sources before loading their official functions. It
requires paired outputs and validates the complete execution inventory by
default. `--allow-partial` produces a separately marked diagnostic output.

Outputs are score-only `parc_task_scores.csv` and `parc_score_summary.json`.
The summary includes task-macro F1, unit-weighted chunk compression, family
aggregates, task wins/ties/losses, and a deterministic task-resampling interval.
This interval is descriptive and resamples task configurations, which can
reuse source questions. LongTableBench also reports its secondary
conversation-averaged F1. Predictions and gold are not copied into the summary.

## Shards and Integrity

Run each shard into a separate JSONL file, using the same configuration and
environment for every shard. Merge through the verified merge command:

```bash
python scripts/parc_merge_shards.py \
  --inputs outputs/part0.jsonl outputs/part1.jsonl \
  --output outputs/parc_merged.jsonl
```

The merger requires every shard, the same model/configuration hashes, correct
shard assignments, unique keys, and complete execution-inventory coverage.
The downstream stages use the resulting merged manifest.
Generation shards read the same completed full retrieval file. Retrieval and
generation outputs include per-record hashes and manifests written before the
first record; interrupted or mismatched runs cannot be mistaken for completion.
Truncated JSONL is rejected rather than silently skipped. Use a separate output
directory for each generator; the two models may share the frozen retrieval
file using the standalone generation CLI.

The included runner covers the primary Dense/PARC comparison, and the
published component summaries have separate task-level sources. It preserves the pinned
LongTableBench formatter's file-order convention, and records the resulting
input identities. Different file ordering, hub revisions, GPU kernels, or
library builds can affect outputs.

Every stage supports `--help`. Dataset roots use the same names throughout:
`--bamboo-root`, `--ethic-root`, `--helmet-root`, and `--longtable-root`.
Standalone entry points are `parc_materialize_retrieval.py` (models, cache and
retrieval output), `parc_generate.py` (retrieval file, generator and generation
output), and `parc_score.py` (generation file, scorer repositories and scores
directory), all under `scripts/`.

## 中文说明

仓库现已公开可运行的检索、路由、生成与官方评分代码。原始数据、模型权重和
上游评分器仍需按各自许可证下载，不随 PARC 仓库重新分发。

1. 使用 Python 3.10.20，并安装 `requirements-full.txt`。CUDA 机器应先安装
   与驱动匹配的 PyTorch wheel。
2. 根据 `configs/parc_pipeline.example.json` 填写四个数据目录、固定版本的
   上游评分仓库和三个模型。配置文件建议放在已忽略的 `outputs/` 下。
3. 执行 `python scripts/parc_run_pipeline.py --config outputs/parc_pipeline.json`。
4. 检索阶段构造 Dense、BM25、RRF、XE、九条候选路由和压缩锚点，并调用冻结
   selector；该阶段不读取答案、预测或 F1。
5. 生成阶段对 Dense/PARC 使用相同 tokenizer、提示词打包和贪心解码；支持稳定
   分片、断点续跑以及 LongTableBench 的真实多轮历史。
6. 评分阶段读取 gold，并先校验固定上游 scorer 的 SHA256，按执行清单核对完整性。
7. 使用 `parc_summarize_paper35.py` 按论文清单汇总 35 个任务、11,997 个评分单元。

单独执行检索 CLI 时需设置 `PYTHONHASHSEED=0`，总入口会自动设置。
填写模型的固定 hub revision 可锁定下载版本。模型文件、数据输入和运行环境
均绑定到 manifest。分片通过 `parc_merge_shards.py` 校验后合并，并生成合并清单。
每个输出文件使用一个写入进程，两个生成模型使用不同输出目录。
当前流水线生成 Dense/PARC 主对照；组件结果另由对应的任务级汇总提供。
评分输出中的任务重采样区间属于描述性统计。
