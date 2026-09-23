# PARC Selector Training / PARC 选择器训练

## What is trained

PARC trains a **routing selector**, not the generator, retriever, or reranker.
Each of two feature views has a HistGradientBoosting regressor and an
ExtraTrees classifier: four fitted tree-ensemble estimators in total.
The raw view has 221 features; the augmented view has 431. Predictions from
the views are averaged before scoring and confidence gating.

Training uses development answer-quality differences. Inference uses only
retrieval observations; it does not generate trial answers or consume test
labels. **Prediction-free is not training-free.** No Qwen or Llama fine-tuning
is performed by these scripts.

The released [paper model](../artifacts/parc_selector/) was fitted on 5,763
queries from the [12 development tasks](development_inventory.md).
Selector fitting and policy selection use these development tasks; evaluation
tasks remain separate.

## Two artifact paths

| Purpose | Entry point | Model |
| --- | --- | --- |
| Reuse the paper's frozen routing decisions | `src/parc_router/parc_runtime.py` | `artifacts/parc_selector/parc_frozen_selector.joblib` |
| Train a new development selector | `scripts/parc_train_selector.py` | A new output directory containing `parc_trained_selector.joblib` |
| Route with your new selector | `scripts/parc_route_trained.py` | Your new model and sibling manifest |

The paper loader keeps its pinned hashes. Do not edit those hashes to load
a retrained model. New artifacts use `PARC_DEVELOPMENT_SELECTOR_V1` and the
status `TRAINED_DEVELOPMENT_SELECTOR_NOT_EVALUATED`; they do not inherit the
paper's F1 results. A checksum proves integrity, not that a joblib file is
safe. Load only models you trained or otherwise trust.

## CPU smoke test

Install the existing pinned dependencies first. The following example uses
32 synthetic queries with **artificial labels**, not benchmark observations.
It needs no GPU, dataset download, LLM, or API key. Choose new output
directories on subsequent runs; commands do not overwrite trained artifacts.

```bash
python scripts/parc_make_training_example.py \
  --output-dir outputs/parc_training_example

python scripts/parc_train_selector.py \
  --data outputs/parc_training_example/parc_development.npz \
  --selection published \
  --output-dir outputs/parc_trained_example
```

The training command prints the new model's SHA256. Use that value below:

```bash
python scripts/parc_route_trained.py \
  --model outputs/parc_trained_example/parc_trained_selector.joblib \
  --model-sha256 <SHA256_PRINTED_BY_TRAINING> \
  --task-input examples/parc_synthetic_task_input.json \
  --output outputs/parc_trained_routing.json
```

To exercise both group-held-out schemes and the 144-policy search, train
into another directory with `--selection cross-validated` (the default):

```bash
python scripts/parc_train_selector.py \
  --data outputs/parc_training_example/parc_development.npz \
  --selection cross-validated \
  --output-dir outputs/parc_trained_cv_example
```

Neither synthetic fit is a benchmark result or evidence of model quality.

## Prepare real development inputs

For each complete development task, provide two separate JSON files:

1. Retrieval observations using the existing [public input contract](public_input_schema.md).
   Each query includes Dense, the compressed safe route, and all nine routes.
   Do not add task names, labels, answers, or generated text to these objects.
2. Development F1 labels mapping `query_key` to an object with all eleven
   method names and their official F1 scores on the **0-1 scale**. These
   labels are computed offline by your development generator/scorer, not
   obtained from retrieval scores. The label keys must exactly match the
   retrieval query keys; no dropped or extra queries are accepted.

Create a group manifest with paths relative to the manifest's directory:

```json
{
  "role": "development",
  "groups": [
    {
      "task": "longbench_triviaqa",
      "family": "LongBench",
      "retrieval": "longbench_triviaqa_retrieval.json",
      "labels": "longbench_triviaqa_labels.json"
    }
  ]
}
```

This illustrates one group, not the full 12-task inventory. For the paper
inputs, include all groups and preserve the recorded query/task ordering.
Task/family names are used for weighting and folds only; they are never
model input features. The preparation code rejects the known paper
evaluation family names, but cannot detect mislabeled data or semantic
overlap. The caller remains responsible for a genuine development/test split.

```bash
python scripts/parc_prepare_training.py \
  --groups data/parc_training_groups.json \
  --output outputs/parc_development.npz

python scripts/parc_train_selector.py \
  --data outputs/parc_development.npz \
  --selection cross-validated \
  --output-dir outputs/parc_retrained
```

The preparation script invokes the same feature extractor and anchor rules
as the frozen runtime. It creates an NPZ file that loads with
`allow_pickle=False` and contains:

| Field | Shape / type | Meaning |
| --- | --- | --- |
| `protocol` | Scalar string | `PARC_DEVELOPMENT_FEATURES_V1` |
| `raw_features` | `N x 9 x 221`, float32 | 212 descriptors plus nine route indicators, in frozen candidate order |
| `target` | `N x 9`, float64 | Each candidate's development F1 minus Dense F1 |
| `compression` | `N x 9`, float64 | Candidate chunk-removal fractions |
| `anchor_delta` | `N`, float64 | Development F1 difference for the query's selected anchor |
| `anchor_compression` | `N`, float64 | Selected anchor's chunk-removal fraction |
| `task`, `family`, `query_key` | `N`, Unicode strings | Grouping and alignment metadata, not features |

All numeric values must be finite. Compression is strictly between zero
and one. Relative features are reconstructed within each query, never
normalized using labels or held-out queries. Prepared labels/features remain
local and are ignored by Git; datasets remain at their upstream sources.

## Fitting and policy selection

The implementation retains the final-fit settings recorded by the reference
training code and frozen model:

- HGB: 220 maximum iterations, learning rate 0.05, 31 leaf nodes,
  minimum leaf size 30, L2 regularization 10, seed 20260719. Unspecified
  sklearn defaults, including `early_stopping='auto'`, remain unchanged.
- ExtraTrees: 160 estimators, depth 14, minimum leaf size 10,
  `max_features=0.65`, no bootstrap, seed 20260816, one worker.
- Regression target: `delta / (abs(delta) + 0.05)`.
- Classes: loss if `delta < -1e-12`, gain if `delta > 1e-12`, otherwise tie.
- Query weights equalize total task mass within each training fold.
  Classifier weights additionally multiply by the inverse class-frequency
  factors used in the original final fit.

`published` reuses policy 116 from `configs/parc_selection_protocol.json`
and only fits the two views. It does not reselect a policy on your data.

`cross-validated` fits both views while holding out each complete task
(LOTO), then each complete family (LOFO). It requires at least two groups
in each scheme. Fold seeds follow the recorded outer-fold rule:
`20260719 + 900000 + 100000 * is_family + sorted_fold_index`.
All nine candidate rows for a query stay in one fold. The final policy is
chosen by the published 144-policy grid and the 13-component objective,
using componentwise LOTO/LOFO minima followed by means; exact ties retain
the first grid entry. Both views are then fitted on all supplied development
queries using the base seed.

The manifest records fold counts/index hashes, selected policy, feasibility
flags, model/data hashes, environment versions, and model dimensions. A false
`feasibility_passed` flag means the selected operating point did not satisfy
all development constraints. Training completion is not a quality gate.
The OOF summaries used to choose a policy are **selection diagnostics**, not
an unbiased assessment of that selected policy. This entry point does not
rerun the earlier nested method-development study or certify generalization.

## Input verification and replay

`--verify-paper-inputs` requires exact reference raw/relative tensor hashes,
target/compression/anchor hashes, and task/family fold-index hashes before
fitting. Matching counts alone is insufficient. It fails on the synthetic
example or differently ordered data. Passing it checks inputs; it does not
claim bit-identical fitted models or independently regenerated paper scores.

The reference OOF caches used eight ExtraTrees workers; this release uses
one worker, matching the final artifact and avoiding parallel probability
reduction variability. Floating-point details and platform/library versions
can still affect exact replay. The original frozen artifact and its
training provenance manifest remain unchanged.

This release includes preparation from **already materialized retrieval
observations and development labels**, final model fitting, OOF policy
selection, and trained-model routing. The benchmark retrieval,
generation and official-scoring pipeline is now included. That pipeline does
not materialize the separate 12-task development labels or replay
all nested-development studies. Downloading raw datasets alone is not
sufficient to reconstruct the recorded training inputs.
See [release boundaries](release_boundary.md) and the
[pinned development sources](development_inventory.md#pinned-upstream-sources).

## 中文说明

我们训练的是小型树模型路由选择器，不是对 Qwen、Llama、检索器或重排器
做微调。两个特征视图各包含一个 HGB 回归器和一个 ExtraTrees 分类器。
训练使用开发集的回答 F1 差值；推理不读取标签，也不生成多份试答。
因此 Prediction-free 不等于 training-free。

先运行上面的合成示例，可以在 CPU 上走通准备、训练和路由流程。
其中标签是人工构造的，只用于检查软件，不是论文证据。真实训练需要
完整开发任务的检索观测，以及按 query_key 对齐的十一种方法的开发 F1。
`parc_prepare_training.py` 生成特征文件，`parc_train_selector.py` 训练模型，
`parc_route_trained.py` 使用新模型路由。数据集来源见开发任务清单，
不上传原始数据、逐样本标签、生成答案或本地训练产物。

默认 `cross-validated` 在开发集上执行 LOTO/LOFO 和 144 策略选择；
`published` 直接复用已公开的策略 116。任务与 family 身份仅用于权重和
划分，不作为特征。选择策略所用的分数不是独立测试成绩。
训练完成也不代表满足质量约束，需检查清单中的可行性标记。

新模型与论文冻结模型采用不同文件名和加载入口，不覆盖论文权重，
不继承论文成绩。`--verify-paper-inputs` 能核对原训练输入的精确哈希，
但不自动证明模型逐字节相同或全流程复现。benchmark 评测的检索、生成与评分
代码已公开；12 个开发任务的检索观测和 F1 标签需另行准备。
