# PARC: Prediction-Free Adaptive Routing and Chunk Compression

## English

This is the proposed public code package for PARC. PARC selects a compressed retrieval view for each query using prediction-free retrieval features, a dual-view selector, confidence gates, and a compressed Dense-order anchor.

### Included

- `src/parc_router/parc_runtime.py`: the public v9 retrieval-only routing entry point;
- `src/parc_router/`: PARC feature extraction, selector, compatibility, and runtime modules;
- `artifacts/parc_selector/`: the frozen selector model, manifest, and SHA256 checksums;
- `configs/parc_policy.json`: the nine-route portfolio and frozen policy constants;
- `scripts/`: deterministic self-tests and synthetic schema fixtures;
- `results/`: score-only task and operating-point summaries;
- `docs/`: input schema, benchmark provenance, and release boundaries.

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
`docs/benchmarks.md` but still does not provide their download/preprocessing
adapters, retriever/reranker checkpoints, or generator evaluation pipeline, so
it is not a one-command reproduction of every paper number.

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

The full routing CLI requires a frozen selector model and manifest:

```bash
python src/parc_router/parc_runtime.py \
  --model artifacts/parc_selector/parc_frozen_selector.joblib \
  --task-input /path/to/opaque_task.json \
  --output routed.json
```

Run the included synthetic routing example:

```bash
bash scripts/run_release_example.sh
```

When more than one Python installation is present, select the locked
interpreter explicitly, for example:

```bash
PYTHON=/path/to/python bash scripts/run_release_example.sh
```

The synthetic fixture is only for schema testing and must never be used as paper evidence.

Verify checksums, dependency versions, the self-test, and the synthetic
routing example together:

```bash
bash scripts/verify_release.sh
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

No open-source license is asserted yet. The authors must choose and add a
license before making the repository public.

## 中文说明

这是 PARC 拟公开的代码包。PARC 使用不依赖预测结果的检索特征、双视图选择器、置信度门控和压缩 Dense 顺序锚点，为每个 query 选择压缩后的检索视图。

包含：PARC v9 检索路由运行时、冻结 selector 模型、9 种路由配置、自检脚本、合成输入样例、score-only 汇总结果、benchmark 来源文档和开源边界文档。

不包含：生成模型、embedding 和 reranker 权重、私有缓存、benchmark 原始数据、prompt、答案、标签、chunk 原文、正式预测、中间分数、服务器路径、SSH 凭据、日志、历史探索脚本以及论文文件；冻结 selector 是明确包含的例外。

当前目录已经包含冻结 selector 模型和 manifest，可以复现检索路由；依赖应严格使用 Python 3.10.20、NumPy 2.2.6、joblib 1.5.3 和 scikit-learn 1.7.2。但还没有四个 benchmark 的公开下载/预处理适配器、检索器/重排器权重和生成评测流水线，因此仍不能宣称一键精确复现论文全部数字。正式公开前还需要完成独立干净环境的全流程复跑。

运行自检：

```bash
python -m pip install -r requirements-lock.txt
python src/parc_router/parc_runtime.py --self-test
```

当前尚未声明开源许可证，正式公开 GitHub 前需要由作者确定并加入许可证文件。
