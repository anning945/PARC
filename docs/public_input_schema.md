# Public Input Schema / 公开输入字段

The v9 runtime accepts a non-empty JSON list. Each query must contain exactly:

```text
query_key, question, chunks, methods
```

`chunks` is a non-empty list of chunk strings. `methods` must contain the Dense baseline, the safe compressed route, and all nine candidate routes. Each method object contains:

```text
kept_indices, retrieved_indices,
dense_top_k_indices, bm25_top_k_indices, rrf_top_k_indices,
retrieved_dense_scores, retrieved_bm25_scores,
retrieved_rrf_scores, retrieved_reranker_scores
```

The runtime checks unique indices, positive compression, finite scores, and consistent retrieval references. It rejects answers, labels, predictions, metrics, F1, gold data, task identity, benchmark identity, and generator identity.

A sibling frozen model manifest is required by the v9 runtime and is included in `artifacts/parc_selector/`. The runtime verifies both the model and manifest SHA256 values before loading them.

v9 运行时接收非空 JSON 列表。每个 query 必须包含上述四个顶层字段；每个 method 必须包含英文列出的检索字段。原始 chunk 数量和文本长度由运行时根据 `chunks` 与索引派生，不作为输入字段。运行时会检查索引、压缩率、分数和检索参考的一致性，并拒绝答案、标签、预测、指标、F1、gold、task/benchmark 身份和生成模型身份。

合成样例用于输入结构与路由接口测试。
