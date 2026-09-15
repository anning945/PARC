# PARC: Prediction-Free Adaptive Routing and Chunk Compression

## English

This is the proposed public code package for PARC. PARC selects a compressed retrieval view for each query using prediction-free retrieval features, a dual-view selector, confidence gates, and a compressed Dense-order anchor.

### Included

- `src/parc_router/parc_runtime.py`: the public v9 retrieval-only routing entry point;
- `src/parc_router/`: PARC feature extraction, selector, compatibility, and runtime modules;
- `src/parc_benchmarks/`: prediction-free adapters for the four benchmark families;
- `artifacts/parc_selector/`: the frozen selector model, manifest, and SHA256 checksums;
- `configs/parc_policy.json`: the nine-route portfolio and frozen policy constants;
- `scripts/`: schema checks, deterministic self-tests, and synthetic fixtures;
- `tests/`: public adapter and routing contract tests;
- `results/`: score-only task and operating-point summaries;
- `docs/`: input schema, benchmark provenance, workflow, and release boundaries.

### Excluded

Generator, embedding, and reranker weights, private caches, benchmark source
files, prompts, answers, labels, retrieved chunk text, formal predictions,
intermediate scores, server paths, SSH credentials, logs, historical
exploratory scripts, and paper files are intentionally excluded. The frozen
routing selector is the explicitly included exception.

The frozen selector model is included for routing-only reproduction. The model
manifest records `FROZEN_FINAL_MODEL_NOT_FORMALLY_CONFIRMED`: it was frozen
before formal generation and does not read formal predictions, answers, F1, or
official quality scores. The released artifact is tied to Python 3.10.20,
NumPy 2.2.6, joblib 1.5.3, and scikit-learn 1.7.2; install the exact versions
from `requirements-lock.txt` or `environment.yml` before loading it.

The package records the four benchmark upstream sources in
`docs/benchmarks.md` and provides prediction-free adapters plus a full-scope
schema check. It does not bundle retriever/reranker checkpoints or a generator
evaluation pipeline, so it is not a one-command reproduction of every paper
number.

### Quick start

Use the locked Python 3.10 environment. The frozen joblib artifact is not
promised to load under arbitrary future scikit-learn versions.

```bash
python -m pip install -r requirements-lock.txt
python src/parc_router/parc_runtime.py --self-test
```

Expected output:

```text
PASS parc_runtime self-test
```

Generate a synthetic schema fixture:

```bash
python scripts/make_synthetic_fixture.py
```

Run the public adapter contract tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

After downloading the four upstream datasets into the layout in
`docs/reproduction_workflow.md`, run the prediction-free full-scope schema
check:

```bash
python -m pip install -r requirements-benchmark.txt
PYTHON=/path/to/python bash scripts/run_parc_schema_check.sh
```

Set `PARC_SELECTOR_STUB_OUTPUT` when the external retriever should consume a
selector-only JSONL export; see `docs/reproduction_workflow.md`.

The full routing CLI requires a frozen selector model and manifest:

```bash
python src/parc_router/parc_runtime.py \
  --model artifacts/parc_selector/parc_frozen_selector.joblib \
  --task-input /path/to/opaque_task.json \
  --output routed.json
```

Run the included synthetic routing example:

```bash
bash scripts/run_parc_example.sh
```

When more than one Python installation is present, select the locked
interpreter explicitly, for example:

```bash
PYTHON=/path/to/python bash scripts/run_parc_example.sh
```

The synthetic fixture is only for schema testing and must never be used as paper evidence.

Verify checksums, dependency versions, the self-test, and the synthetic
routing example together:

```bash
bash scripts/verify_parc_release.sh
```

The verification script checks both the source-file checksums and the frozen
artifact checksums. Set `PYTHON` in the same way when multiple interpreters
are installed.

### Reproduction status

The source, frozen routing artifact, and score-only summaries are suitable for
code review and routing audit. Exact end-to-end reproduction of the paper
remains pending public benchmark acquisition instructions, clean benchmark
adapters, and an independent full replay.

### License

This software is released under the MIT License. See `LICENSE`.

## 中文说明

这是 PARC 拟公开的代码包。PARC 使用不依赖预测结果的检索特征、双视图选择器、置信度门控和压缩 Dense 顺序锚点，为每个 query 选择压缩后的检索视图。

包含：PARC v9 检索路由运行时、冻结 selector 模型、9 种路由配置、自检脚本、合成输入样例、score-only 汇总结果、benchmark 来源文档和开源边界文档。

不包含：生成模型、embedding 和 reranker 权重、私有缓存、benchmark 原始数据、prompt、答案、标签、chunk 原文、正式预测、中间分数、服务器路径、SSH 凭据、日志、历史探索脚本以及论文文件；冻结 selector 是明确包含的例外。

当前目录已经包含冻结 selector、四族 benchmark 的无预测 adapter 和全量 schema 检查流程，可以复现检索路由；依赖应严格使用 Python 3.10.20、NumPy 2.2.6、joblib 1.5.3 和 scikit-learn 1.7.2。检索器/重排器权重和生成评测流水线仍需由使用者按 benchmark 规范提供，因此仍不能宣称一键精确复现论文全部数字。

运行自检：

```bash
python -m pip install -r requirements-lock.txt
python src/parc_router/parc_runtime.py --self-test
```

本软件采用 MIT License，具体条款见 `LICENSE`。
