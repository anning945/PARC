#!/usr/bin/env python3
"""Extract label-free query/retrieval features for Adaptive-v3 routing.

Inputs are opened-development retrieval JSONL files only.  The output contains
numeric summaries and merge keys; it deliberately omits raw questions,
contexts, prompts, predictions, gold answers, and official scores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL = "AdaptivePARC-v3-label-free-enhanced-features-v1"
EXPECTED_TASKS = 12
EXPECTED_METHODS = 11
EXPECTED_QUERY_ROWS = 5763
EXPECTED_ROWS = EXPECTED_QUERY_ROWS * EXPECTED_METHODS
EPS = 1e-12

EN_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
NUMBER_RE = re.compile(r"(?<!\w)[+-]?\d+(?:[.,]\d+)*(?:%|\b)")
YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|2100)\b")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")

QUESTION_CATEGORIES = {
    "person": ("who", "whose", "whom", "谁", "哪位", "何人"),
    "time": (
        "when",
        "what year",
        "what date",
        "什么时候",
        "何时",
        "哪年",
        "日期",
    ),
    "location": ("where", "which country", "which city", "哪里", "哪儿", "何地"),
    "reason": ("why", "what caused", "because", "为什么", "为何", "原因"),
    "manner": ("how", "in what way", "如何", "怎么", "怎样"),
    "quantity": (
        "how many",
        "how much",
        "what percentage",
        "多少",
        "几",
        "比例",
        "百分比",
    ),
    "choice": ("which", "what type", "what kind", "哪一个", "哪种", "哪个"),
    "boolean": (
        "is ",
        "are ",
        "was ",
        "were ",
        "does ",
        "do ",
        "did ",
        "can ",
        "是否",
        "是不是",
        "能否",
    ),
    "definition": ("what is", "what are", "define", "是什么", "指什么", "定义"),
}

NEGATIONS = (
    " not ",
    "n't",
    " never ",
    " no ",
    " except",
    "least",
    "不",
    "没有",
    "未",
    "除",
    "最少",
)
COMPARISONS = (
    "more than",
    "less than",
    "largest",
    "smallest",
    "highest",
    "lowest",
    "difference",
    "compare",
    "相比",
    "比较",
    "最大",
    "最小",
    "最高",
    "最低",
    "差异",
)
MULTIHOP = (
    " and ",
    " both ",
    " respectively",
    "relationship",
    "共同",
    "分别",
    "以及",
    "与",
    "和",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field}: non-finite")
    return result


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def std(values: Sequence[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), EPS)


def normalized_entropy(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    probabilities = [value / total for value in exponentials]
    entropy = -sum(
        probability * math.log(max(probability, EPS))
        for probability in probabilities
    )
    return entropy / math.log(len(probabilities))


def cjk_ngrams(text: str, n: int = 2) -> set[str]:
    chars = CJK_RE.findall(text)
    return {
        "".join(chars[index : index + n])
        for index in range(max(0, len(chars) - n + 1))
    }


def english_words(text: str) -> set[str]:
    return {word.lower() for word in EN_WORD_RE.findall(text)}


def lexical_sets(text: str) -> tuple[set[str], set[str]]:
    return english_words(text), cjk_ngrams(text)


def overlap_features(question: str, context: str, prefix: str) -> dict[str, float]:
    q_words, q_cjk = lexical_sets(question)
    c_words, c_cjk = lexical_sets(context)
    word_intersection = len(q_words & c_words)
    cjk_intersection = len(q_cjk & c_cjk)
    q_numbers = set(NUMBER_RE.findall(question))
    c_numbers = set(NUMBER_RE.findall(context))
    return {
        f"{prefix}_word_recall": ratio(word_intersection, len(q_words)),
        f"{prefix}_word_jaccard": ratio(
            word_intersection, len(q_words | c_words)
        ),
        f"{prefix}_cjk_bigram_recall": ratio(cjk_intersection, len(q_cjk)),
        f"{prefix}_cjk_bigram_jaccard": ratio(
            cjk_intersection, len(q_cjk | c_cjk)
        ),
        f"{prefix}_number_recall": ratio(
            len(q_numbers & c_numbers), len(q_numbers)
        ),
    }


def question_features(question: str) -> dict[str, float]:
    if not isinstance(question, str):
        raise ValueError("question must be a string")
    length = len(question)
    categories = CounterLike(unicodedata.category(char) for char in question)
    cjk_count = len(CJK_RE.findall(question))
    ascii_alpha = sum(char.isascii() and char.isalpha() for char in question)
    digits = sum(char.isdigit() for char in question)
    whitespace = sum(char.isspace() for char in question)
    punctuation = sum(category.startswith("P") for category in categories.items)
    symbols = sum(category.startswith("S") for category in categories.items)
    lowered = f" {question.lower()} "
    words = EN_WORD_RE.findall(question)
    output: dict[str, float] = {
        "question_chars": float(length),
        "question_log1p_chars": math.log1p(length),
        "question_utf8_bytes": float(len(question.encode("utf-8"))),
        "question_english_words": float(len(words)),
        "question_unique_word_ratio": ratio(
            len({word.lower() for word in words}), len(words)
        ),
        "question_cjk_chars": float(cjk_count),
        "question_cjk_ratio": ratio(cjk_count, length),
        "question_ascii_alpha_ratio": ratio(ascii_alpha, length),
        "question_digit_count": float(digits),
        "question_digit_ratio": ratio(digits, length),
        "question_number_token_count": float(len(NUMBER_RE.findall(question))),
        "question_year_count": float(len(YEAR_RE.findall(question))),
        "question_whitespace_ratio": ratio(whitespace, length),
        "question_punctuation_ratio": ratio(punctuation, length),
        "question_symbol_ratio": ratio(symbols, length),
        "question_unique_char_ratio": ratio(len(set(question)), length),
        "question_mark_count": float(question.count("?") + question.count("？")),
        "question_quote_count": float(
            sum(question.count(char) for char in "\"'“”‘’《》")
        ),
        "question_negation_hits": float(
            sum(lowered.count(pattern) for pattern in NEGATIONS)
        ),
        "question_comparison_hits": float(
            sum(lowered.count(pattern) for pattern in COMPARISONS)
        ),
        "question_multihop_hits": float(
            sum(lowered.count(pattern) for pattern in MULTIHOP)
        ),
    }
    category_hits = 0
    for category, patterns in QUESTION_CATEGORIES.items():
        hit = float(any(pattern in lowered for pattern in patterns))
        output[f"question_type_{category}"] = hit
        category_hits += int(hit)
    output["question_type_count"] = float(category_hits)
    return output


class CounterLike:
    """Tiny category container avoiding collections.Counter in the hot path."""

    def __init__(self, items: Iterable[str]) -> None:
        self.items = list(items)


def sequence_features(
    indices: Sequence[Any],
    original_chunks: int,
    prefix: str,
) -> dict[str, float]:
    values = [int(value) for value in indices]
    if not values:
        names = (
            "count",
            "unique_ratio",
            "mean_norm",
            "std_norm",
            "min_norm",
            "max_norm",
            "range_norm",
            "first_norm",
            "last_norm",
            "mean_abs_gap_norm",
            "adjacent_fraction",
            "ascending_fraction",
            "descending_fraction",
            "early_fraction",
            "late_fraction",
        )
        return {f"{prefix}_{name}": 0.0 for name in names}
    denominator = max(original_chunks - 1, 1)
    normalized = [value / denominator for value in values]
    gaps = [
        values[index + 1] - values[index]
        for index in range(len(values) - 1)
    ]
    return {
        f"{prefix}_count": float(len(values)),
        f"{prefix}_unique_ratio": ratio(len(set(values)), len(values)),
        f"{prefix}_mean_norm": mean(normalized),
        f"{prefix}_std_norm": std(normalized),
        f"{prefix}_min_norm": min(normalized),
        f"{prefix}_max_norm": max(normalized),
        f"{prefix}_range_norm": max(normalized) - min(normalized),
        f"{prefix}_first_norm": normalized[0],
        f"{prefix}_last_norm": normalized[-1],
        f"{prefix}_mean_abs_gap_norm": ratio(
            mean([abs(value) for value in gaps]), denominator
        ),
        f"{prefix}_adjacent_fraction": ratio(
            sum(abs(value) == 1 for value in gaps), len(gaps)
        ),
        f"{prefix}_ascending_fraction": ratio(
            sum(value > 0 for value in gaps), len(gaps)
        ),
        f"{prefix}_descending_fraction": ratio(
            sum(value < 0 for value in gaps), len(gaps)
        ),
        f"{prefix}_early_fraction": ratio(
            sum(value <= 0.25 for value in normalized), len(normalized)
        ),
        f"{prefix}_late_fraction": ratio(
            sum(value >= 0.75 for value in normalized), len(normalized)
        ),
    }


def permutation_features(
    candidate: Sequence[Any],
    dense: Sequence[Any],
) -> dict[str, float]:
    candidate_values = [int(value) for value in candidate]
    dense_values = [int(value) for value in dense]
    dense_rank = {value: rank for rank, value in enumerate(dense_values)}
    common = [value for value in candidate_values if value in dense_rank]
    if not common:
        return {
            "candidate_dense_common_fraction": 0.0,
            "candidate_dense_rank_displacement": 1.0,
            "candidate_dense_rank_agreement": 0.0,
            "candidate_dense_first_rank_norm": 1.0,
        }
    candidate_rank = {value: rank for rank, value in enumerate(candidate_values)}
    denominator = max(len(dense_values) - 1, 1)
    displacement = mean(
        [
            abs(candidate_rank[value] - dense_rank[value]) / denominator
            for value in common
        ]
    )
    pairs = 0
    agreements = 0
    for left in range(len(common)):
        for right in range(left + 1, len(common)):
            first = common[left]
            second = common[right]
            pairs += 1
            agreements += int(
                (candidate_rank[first] - candidate_rank[second])
                * (dense_rank[first] - dense_rank[second])
                > 0
            )
    return {
        "candidate_dense_common_fraction": ratio(
            len(common), len(dense_values)
        ),
        "candidate_dense_rank_displacement": displacement,
        "candidate_dense_rank_agreement": ratio(agreements, pairs),
        "candidate_dense_first_rank_norm": ratio(
            dense_rank.get(candidate_values[0], len(dense_values)),
            max(len(dense_values), 1),
        ),
    }


def score_features(values: Sequence[Any], prefix: str) -> dict[str, float]:
    scores = [finite(value, prefix) for value in values]
    if not scores:
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_top1": 0.0,
            f"{prefix}_top2_margin": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_range": 0.0,
            f"{prefix}_cv_abs": 0.0,
            f"{prefix}_entropy": 0.0,
            f"{prefix}_slope": 0.0,
        }
    ordered = sorted(scores, reverse=True)
    return {
        f"{prefix}_count": float(len(scores)),
        f"{prefix}_top1": ordered[0],
        f"{prefix}_top2_margin": (
            ordered[0] - ordered[1] if len(ordered) > 1 else 0.0
        ),
        f"{prefix}_mean": mean(scores),
        f"{prefix}_std": std(scores),
        f"{prefix}_range": max(scores) - min(scores),
        f"{prefix}_cv_abs": ratio(std(scores), abs(mean(scores))),
        f"{prefix}_entropy": normalized_entropy(scores),
        f"{prefix}_slope": ratio(scores[-1] - scores[0], len(scores) - 1),
    }


def segment_overlap_features(question: str, context: str) -> dict[str, float]:
    segments = [
        segment.strip()
        for segment in re.split(r"\n\s*\n", context)
        if segment.strip()
    ]
    if not segments:
        return {
            "segment_count": 0.0,
            "segment_overlap_mean": 0.0,
            "segment_overlap_std": 0.0,
            "segment_overlap_max": 0.0,
            "segment_overlap_margin": 0.0,
            "segment_overlap_best_rank_norm": 0.0,
            "segment_overlap_entropy": 0.0,
        }
    q_words, q_cjk = lexical_sets(question)
    values: list[float] = []
    for segment in segments:
        s_words, s_cjk = lexical_sets(segment)
        word = ratio(len(q_words & s_words), len(q_words))
        cjk = ratio(len(q_cjk & s_cjk), len(q_cjk))
        values.append(max(word, cjk))
    ordered = sorted(values, reverse=True)
    best = max(range(len(values)), key=values.__getitem__)
    return {
        "segment_count": float(len(segments)),
        "segment_overlap_mean": mean(values),
        "segment_overlap_std": std(values),
        "segment_overlap_max": ordered[0],
        "segment_overlap_margin": (
            ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
        ),
        "segment_overlap_best_rank_norm": ratio(best, len(values) - 1),
        "segment_overlap_entropy": normalized_entropy(values),
    }


def extract_row(record: Mapping[str, Any]) -> dict[str, Any]:
    question = str(record["question"])
    context = str(record["retrieved_context"])
    original_chunks = int(record["original_chunks"])
    retrieved_indices = record["retrieved_indices"]
    dense_indices = record["dense_top_k_indices"]
    kept_indices = record["kept_indices"]
    output: dict[str, Any] = {
        "family": str(record["benchmark_family"]),
        "dataset": str(record["dataset"]),
        "sample_index": int(record["sample_index"]),
        "method": str(record["method"]),
        "record_id_sha256": hashlib.sha256(
            (
                f"{record['dataset']}\0{record['sample_index']}\0"
                f"{record['method']}\0{record['id']}"
            ).encode("utf-8")
        ).hexdigest(),
        "question_sha256": hashlib.sha256(
            question.encode("utf-8")
        ).hexdigest(),
        "original_text_chars": int(record["original_text_chars"]),
        "candidate_text_chars": int(record["candidate_text_chars"]),
        "retrieved_context_chars": len(context),
        "retrieved_context_to_original_ratio": ratio(
            len(context), int(record["original_text_chars"])
        ),
    }
    output.update(question_features(question))
    output.update(overlap_features(question, context, "context"))
    output.update(segment_overlap_features(question, context))
    output.update(
        sequence_features(retrieved_indices, original_chunks, "retrieved_index")
    )
    output.update(
        sequence_features(dense_indices, original_chunks, "dense_index")
    )
    output.update(
        sequence_features(record["bm25_top_k_indices"], original_chunks, "bm25_index")
    )
    output.update(
        sequence_features(record["rrf_top_k_indices"], original_chunks, "rrf_index")
    )
    output.update(
        sequence_features(kept_indices, original_chunks, "kept_index")
    )
    output.update(permutation_features(retrieved_indices, dense_indices))
    output.update(score_features(record["retrieved_dense_scores"], "dense_score"))
    output.update(score_features(record["retrieved_bm25_scores"], "bm25_score"))
    output.update(score_features(record["retrieved_rrf_scores"], "rrf_score"))
    output.update(
        score_features(record["retrieved_reranker_scores"], "reranker_score")
    )
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write empty output")
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if list(row) != fields:
                raise ValueError("inconsistent output schema")
            writer.writerow(row)
    temporary.replace(path)


def run(input_dir: Path, output_csv: Path, output_summary: Path) -> dict[str, Any]:
    files = sorted(input_dir.glob("*__retrieval.jsonl"))
    if len(files) != EXPECTED_TASKS * EXPECTED_METHODS:
        raise ValueError(
            f"retrieval files {len(files)} != "
            f"{EXPECTED_TASKS * EXPECTED_METHODS}"
        )
    rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    task_methods: set[tuple[str, str]] = set()
    query_questions: dict[tuple[str, int], str] = {}
    for path in files:
        source_hashes[path.name] = sha256_file(path)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                row = extract_row(record)
                key = (str(row["dataset"]), int(row["sample_index"]))
                previous = query_questions.setdefault(
                    key, str(row["question_sha256"])
                )
                if previous != row["question_sha256"]:
                    raise ValueError(f"{key}: question differs across methods")
                task_methods.add((str(row["dataset"]), str(row["method"])))
                rows.append(row)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"rows {len(rows)} != {EXPECTED_ROWS}")
    if len(query_questions) != EXPECTED_QUERY_ROWS:
        raise ValueError(
            f"query rows {len(query_questions)} != {EXPECTED_QUERY_ROWS}"
        )
    if len(task_methods) != EXPECTED_TASKS * EXPECTED_METHODS:
        raise ValueError("task/method coverage mismatch")
    rows.sort(
        key=lambda row: (
            str(row["family"]),
            str(row["dataset"]),
            int(row["sample_index"]),
            str(row["method"]),
        )
    )
    write_csv(output_csv, rows)
    summary = {
        "status": "COMPLETE_LABEL_FREE_FEATURE_EXTRACTION",
        "protocol": PROTOCOL,
        "claim_boundary": "opened_development_only",
        "source_directory": str(input_dir.resolve()),
        "source_file_count": len(files),
        "source_hash_inventory": source_hashes,
        "tasks": EXPECTED_TASKS,
        "methods": EXPECTED_METHODS,
        "query_rows": EXPECTED_QUERY_ROWS,
        "feature_rows": len(rows),
        "output_columns": list(rows[0]),
        "raw_question_or_context_emitted": False,
        "gold_prediction_or_official_score_read": False,
        "output_csv": str(output_csv.resolve()),
        "output_csv_sha256": sha256_file(output_csv),
        "extractor_sha256": sha256_file(Path(__file__).resolve()),
    }
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    summary = run(args.input_dir, args.output_csv, args.output_summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "feature_rows": summary["feature_rows"],
                "output_csv_sha256": summary["output_csv_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
