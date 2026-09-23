"""Materialize the eleven PARC retrieval observations and frozen route.

This module contains the retrieval policy used by the public PARC selector:
Dense and BM25 rankings, reciprocal-rank fusion, compressed candidate pools,
the safe compressed route, and all nine selector candidates. It never reads
answers, predictions, task scores, or generator outputs.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Protocol, Sequence

import numpy as np

DENSE_METHOD = "FullPool_Dense_k8"
SAFE_METHOD = "PARC_CompressedDense_R45_k8"


@dataclass(frozen=True)
class MethodConfig:
    name: str
    family: str
    top_k: int = 8
    keep_ratio: float = 1.0
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    rrf_k: int = 60
    neighbor_fraction: float = 0.0
    window_radius: int = 1
    dense_protect: int = 0
    replacement_margin: float = 0.0
    cjk_replacement_margin: float = -1.0
    order_mode: str = "none"


_LANG_R45 = MethodConfig(
    "PARC_Hybrid_RRFXE_LangMargin_D7A1_EN25_CJK20_R45_k8",
    family="hybrid_language_margin",
    keep_ratio=0.45,
    neighbor_fraction=0.10,
    dense_protect=7,
    replacement_margin=0.25,
    cjk_replacement_margin=0.20,
)

METHODS = (
    MethodConfig(DENSE_METHOD, family="dense"),
    MethodConfig(
        SAFE_METHOD, family="compressed_dense", keep_ratio=0.45, neighbor_fraction=0.10
    ),
    MethodConfig(
        "PARC_CompressedDense_DocOrder_R45_k8",
        family="compressed_order",
        keep_ratio=0.45,
        neighbor_fraction=0.10,
        order_mode="document",
    ),
    MethodConfig(
        "PARC_CompressedDense_RRFOrder_R45_k8",
        family="compressed_order",
        keep_ratio=0.45,
        neighbor_fraction=0.10,
        order_mode="rrf",
    ),
    MethodConfig(
        "PARC_CompressedDense_RerankOrder_R45_k8",
        family="compressed_rerank_order",
        keep_ratio=0.45,
        neighbor_fraction=0.10,
        order_mode="rerank",
    ),
    MethodConfig(
        "PARC_Hybrid_BM25_D7A1_R45_k8",
        family="hybrid_bm25",
        keep_ratio=0.45,
        neighbor_fraction=0.10,
        dense_protect=7,
    ),
    dataclasses.replace(
        _LANG_R45,
        name="PARC_Hybrid_RRFXE_LangMargin_D7A1_EN25_CJK20_R35_k8",
        keep_ratio=0.35,
    ),
    _LANG_R45,
    dataclasses.replace(
        _LANG_R45,
        name="PARC_Hybrid_RRFXE_LangMargin_D7A1_EN25_CJK20_R55_k8",
        keep_ratio=0.55,
    ),
    MethodConfig(
        "PARC_Hybrid_RRF_D6A2_R45_k8",
        family="hybrid_rrf",
        keep_ratio=0.45,
        neighbor_fraction=0.10,
        dense_protect=6,
    ),
    MethodConfig(
        "PARC_Hybrid_Rerank_D7A1_M10_R45_k8",
        family="hybrid_rerank",
        keep_ratio=0.45,
        neighbor_fraction=0.10,
        dense_protect=7,
        replacement_margin=0.10,
    ),
)

METHOD_BY_NAME = {method.name: method for method in METHODS}


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class Reranker(Protocol):
    def score(self, question: str, chunks: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model: str, device: str, batch_size: int) -> None:
        from sentence_transformers import SentenceTransformer

        self.batch_size = batch_size
        self.model = SentenceTransformer(model, device=device, trust_remote_code=True)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        values = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(values, dtype=np.float32)


class TransformersReranker:
    def __init__(
        self, model: str, device: str, batch_size: int, max_length: int
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model, trust_remote_code=True, dtype=dtype
        ).to(self.device)
        self.model.eval()

    def score(self, question: str, chunks: Sequence[str]) -> np.ndarray:
        batches: list[np.ndarray] = []
        for start in range(0, len(chunks), self.batch_size):
            batch = list(chunks[start : start + self.batch_size])
            encoded = self.tokenizer(
                [question] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with self.torch.inference_mode():
                logits = self.model(**encoded, return_dict=True).logits
            batches.append(logits.reshape(-1).float().cpu().numpy())
        if not batches:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(batches).astype(np.float16).astype(np.float32)


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))


def lexical_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]", str(text).lower())


def bm25_scores(
    query: str, documents: Sequence[str], k1: float = 1.5, b: float = 0.75
) -> np.ndarray:
    tokenized = [lexical_tokens(document) for document in documents]
    if not tokenized:
        return np.zeros(0, dtype=np.float32)
    lengths = np.asarray([len(tokens) for tokens in tokenized], dtype=np.float32)
    average = max(float(lengths.mean()), 1.0)
    frequencies = [Counter(tokens) for tokens in tokenized]
    document_frequency: Counter[str] = Counter()
    for values in frequencies:
        document_frequency.update(values.keys())
    scores = np.zeros(len(tokenized), dtype=np.float32)
    for term in set(lexical_tokens(query)):
        frequency = float(document_frequency.get(term, 0))
        if frequency <= 0:
            continue
        inverse = math.log(1.0 + (len(tokenized) - frequency + 0.5) / (frequency + 0.5))
        for index, values in enumerate(frequencies):
            tf = float(values.get(term, 0))
            if tf:
                normalizer = tf + k1 * (1.0 - b + b * float(lengths[index]) / average)
                scores[index] += inverse * tf * (k1 + 1.0) / normalizer
    return scores


def descending_order(scores: np.ndarray) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))


def reciprocal_rank_fusion(
    dense_scores: np.ndarray, sparse_scores: np.ndarray, rrf_k: int = 60
) -> tuple[np.ndarray, list[int]]:
    fused = np.zeros(len(dense_scores), dtype=np.float32)
    for rank, index in enumerate(descending_order(dense_scores), 1):
        fused[index] += 1.0 / (rrf_k + rank)
    for rank, index in enumerate(descending_order(sparse_scores), 1):
        fused[index] += 1.0 / (rrf_k + rank)
    return fused, descending_order(fused)


def compressed_limits(method: MethodConfig, total_chunks: int) -> tuple[int, int, int]:
    if total_chunks <= 1:
        raise ValueError("strict compression requires at least two chunks")
    top_k = min(method.top_k, total_chunks - 1)
    protected = min(method.dense_protect, top_k)
    target = min(
        total_chunks - 1, max(top_k, math.ceil(total_chunks * method.keep_ratio))
    )
    if not 0 < top_k <= target < total_chunks:
        raise ValueError(f"{method.name}: invalid compressed limits")
    return top_k, protected, target


def select_pool(
    fused_order: Sequence[int],
    method: MethodConfig,
    total_chunks: int,
    protected_indices: Sequence[int] = (),
) -> list[int]:
    top_k, _, target = compressed_limits(method, total_chunks)
    neighbor_slots = min(target - top_k, round(target * method.neighbor_fraction))
    anchor_target = max(top_k, target - neighbor_slots)
    selected: list[int] = []
    seen: set[int] = set()
    for raw in protected_indices:
        index = int(raw)
        if 0 <= index < total_chunks and index not in seen:
            selected.append(index)
            seen.add(index)
        if len(selected) == target:
            return selected
    for raw in fused_order:
        index = int(raw)
        if index not in seen:
            selected.append(index)
            seen.add(index)
        if len(selected) >= anchor_target:
            break
    anchors = list(selected)
    for distance in range(1, method.window_radius + 1):
        for anchor in anchors:
            for index in (anchor - distance, anchor + distance):
                if len(selected) >= target:
                    return selected
                if 0 <= index < total_chunks and index not in seen:
                    selected.append(index)
                    seen.add(index)
                    if len(selected) == target:
                        return selected
    for raw in fused_order:
        index = int(raw)
        if index not in seen:
            selected.append(index)
            seen.add(index)
        if len(selected) == target:
            break
    if len(selected) != target:
        raise RuntimeError(f"{method.name}: incomplete candidate pool")
    return selected


def rank_fill(
    dense_order: Sequence[int],
    alternate_order: Sequence[int],
    kept: Sequence[int],
    top_k: int,
    protect: int,
) -> list[int]:
    selected = [int(index) for index in dense_order[:protect]]
    kept_set = {int(index) for index in kept}
    for order in (alternate_order, dense_order):
        for raw in order:
            index = int(raw)
            if index in kept_set and index not in selected:
                selected.append(index)
            if len(selected) >= top_k:
                return selected[:top_k]
    return selected[:top_k]


def normalize_relevance(scores: np.ndarray) -> np.ndarray:
    if not len(scores):
        return scores.astype(np.float32)
    minimum, maximum = float(np.min(scores)), float(np.max(scores))
    if maximum - minimum < 1e-8:
        return np.ones_like(scores, dtype=np.float32)
    return ((scores - minimum) / (maximum - minimum)).astype(np.float32)


def reranker_gate(
    dense_order: Sequence[int],
    proposal_order: Sequence[int],
    kept: Sequence[int],
    scores: np.ndarray,
    top_k: int,
    protect: int,
    margin: float,
) -> list[int]:
    baseline = [int(index) for index in dense_order[:top_k]]
    proposal = rank_fill(dense_order, proposal_order, kept, top_k, protect)
    added = [index for index in proposal if index not in set(baseline)]
    removed = [index for index in baseline if index not in set(proposal)]
    if len(added) != 1 or len(removed) != 1:
        return baseline
    relevance = normalize_relevance(np.asarray(scores, dtype=np.float32))
    score_by_index = {
        int(index): float(relevance[local]) for local, index in enumerate(kept)
    }
    if (
        score_by_index.get(added[0], -math.inf)
        <= score_by_index.get(removed[0], -math.inf) + margin
    ):
        return baseline
    return proposal


def adaptive_rerank(
    dense_order: Sequence[int],
    kept: Sequence[int],
    scores: np.ndarray,
    top_k: int,
    protect: int,
    margin: float,
) -> list[int]:
    protected = [int(index) for index in dense_order[:protect]]
    tail = [int(index) for index in dense_order[protect:top_k]]
    relevance = normalize_relevance(np.asarray(scores, dtype=np.float32))
    score_by_index = {
        int(index): float(relevance[local]) for local, index in enumerate(kept)
    }
    extras = sorted(
        (int(index) for index in kept if index not in set(protected + tail)),
        key=lambda index: (-score_by_index[index], index),
    )
    for extra in extras[: max(0, top_k - protect)]:
        if not tail:
            break
        position = min(
            range(len(tail)),
            key=lambda value: (score_by_index.get(tail[value], -math.inf), -value),
        )
        if (
            score_by_index[extra]
            <= score_by_index.get(tail[position], -math.inf) + margin
        ):
            break
        tail[position] = extra
    return (protected + tail)[:top_k]


def materialize_methods(
    question: str,
    chunks: Sequence[str],
    chunk_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    reranker: Reranker,
) -> dict[str, dict[str, Any]]:
    if (
        not chunks
        or chunk_embeddings.ndim != 2
        or chunk_embeddings.shape[0] != len(chunks)
    ):
        raise ValueError("chunk/embedding mismatch")
    if (
        query_embedding.ndim != 1
        or query_embedding.shape[0] != chunk_embeddings.shape[1]
    ):
        raise ValueError("query embedding dimension mismatch")
    if not np.all(np.isfinite(chunk_embeddings)) or not np.all(
        np.isfinite(query_embedding)
    ):
        raise ValueError("non-finite embeddings")
    dense_scores = np.asarray(chunk_embeddings @ query_embedding, dtype=np.float32)
    sparse_scores = bm25_scores(question, chunks)
    fused_scores, fused_order = reciprocal_rank_fusion(dense_scores, sparse_scores)
    dense_order, sparse_order = (
        descending_order(dense_scores),
        descending_order(sparse_scores),
    )
    cache: MutableMapping[tuple[int, ...], np.ndarray] = {}
    output: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        reranker_values: list[float] = []
        if method.family == "dense":
            kept = list(range(len(chunks)))
            retrieved = dense_order[: method.top_k]
        else:
            top_k, protect, _ = compressed_limits(method, len(chunks))
            protected = dense_order[:top_k]
            kept = select_pool(fused_order, method, len(chunks), protected)
            if method.family == "compressed_dense":
                retrieved = dense_order[:top_k]
            elif method.family == "compressed_order":
                retrieved = list(dense_order[:top_k])
                if method.order_mode == "document":
                    retrieved.sort()
                elif method.order_mode == "rrf":
                    retrieved.sort(
                        key=lambda index: (-float(fused_scores[index]), index)
                    )
            elif method.family in {"hybrid_bm25", "hybrid_rrf"}:
                alternate = (
                    sparse_order if method.family == "hybrid_bm25" else fused_order
                )
                retrieved = rank_fill(dense_order, alternate, kept, top_k, protect)
            else:
                key = tuple(kept)
                if key not in cache:
                    cache[key] = np.asarray(
                        reranker.score(question, [chunks[index] for index in kept]),
                        dtype=np.float32,
                    )
                scores = cache[key]
                if len(scores) != len(kept):
                    raise ValueError("reranker score alignment mismatch")
                if scores.ndim != 1 or not np.all(np.isfinite(scores)):
                    raise ValueError("invalid reranker scores")
                if method.family == "compressed_rerank_order":
                    score_by_index = {
                        index: float(scores[local]) for local, index in enumerate(kept)
                    }
                    retrieved = sorted(
                        dense_order[:top_k],
                        key=lambda index: (-score_by_index[index], index),
                    )
                elif method.family == "hybrid_rerank":
                    retrieved = adaptive_rerank(
                        dense_order,
                        kept,
                        scores,
                        top_k,
                        protect,
                        method.replacement_margin,
                    )
                elif method.family == "hybrid_language_margin":
                    margin = (
                        method.cjk_replacement_margin
                        if contains_cjk(question.strip())
                        else method.replacement_margin
                    )
                    retrieved = reranker_gate(
                        dense_order, fused_order, kept, scores, top_k, protect, margin
                    )
                else:
                    raise ValueError(f"unsupported family: {method.family}")
                reranker_values = [
                    float(scores[kept.index(index)]) for index in retrieved
                ]
        if not set(retrieved).issubset(set(kept)):
            raise RuntimeError(f"{method.name}: retrieved index outside pool")
        output[method.name] = {
            "kept_indices": [int(index) for index in kept],
            "retrieved_indices": [int(index) for index in retrieved],
            "dense_top_k_indices": [
                int(index) for index in dense_order[: method.top_k]
            ],
            "bm25_top_k_indices": [
                int(index) for index in sparse_order[: method.top_k]
            ],
            "rrf_top_k_indices": [int(index) for index in fused_order[: method.top_k]],
            "retrieved_dense_scores": [
                float(dense_scores[index]) for index in retrieved
            ],
            "retrieved_bm25_scores": [
                float(sparse_scores[index]) for index in retrieved
            ],
            "retrieved_rrf_scores": [float(fused_scores[index]) for index in retrieved],
            "retrieved_reranker_scores": reranker_values,
        }
    return output


def encode_query(
    embedder: Embedder, question: str, chunks: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    chunk_embeddings = np.asarray(embedder.encode(chunks), dtype=np.float32)
    query_values = np.asarray(embedder.encode([question]), dtype=np.float32)
    if query_values.shape[0] != 1:
        raise ValueError("embedder returned an invalid query matrix")
    return chunk_embeddings, query_values[0]


def selected_record(
    methods: Mapping[str, Mapping[str, Any]], selected_method: str
) -> dict[str, Any]:
    if selected_method not in methods:
        raise ValueError(f"selector returned unknown method: {selected_method}")
    observation = methods[selected_method]
    original = len(methods[DENSE_METHOD]["kept_indices"])
    kept = len(observation["kept_indices"])
    compression = 1.0 - kept / original
    if compression <= 0.0:
        raise ValueError("PARC selected a non-compressed route")
    return {
        "selected_method": selected_method,
        "selected_indices": list(observation["retrieved_indices"]),
        "kept_chunks": kept,
        "original_chunks": original,
        "chunk_compression": compression,
        "dense_indices": list(methods[DENSE_METHOD]["retrieved_indices"]),
    }
