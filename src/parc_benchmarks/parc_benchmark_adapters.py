#!/usr/bin/env python3
"""Prediction-free adapters for the frozen four-family PARC confirmation.

The adapters in this module have two deliberately separated views:

* an internal benchmark view, which contains only information needed to rebuild
  the official prompt after retrieval; and
* a selector view containing exactly ``query_key``, ``question``, and
  ``chunks``.  Retrieval code may add the fourth public runtime field,
  ``methods``, only after all candidate retrieval artifacts are materialized.

No adapter reads a test answer value, prediction, or score.  HELMET's released
two-shot demonstration answers are read because they are part of the official
input prompt, not the untouched test labels.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import json
import math
import os
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


PROTOCOL = "parc-four-family-adapters-20260718"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 8

SELECTOR_STUB_FIELDS = frozenset({"query_key", "question", "chunks"})
RUNTIME_QUERY_FIELDS = frozenset(
    {"query_key", "question", "chunks", "methods"}
)
FORBIDDEN_SELECTOR_FIELD_FRAGMENTS = (
    "answer",
    "benchmark",
    "family",
    "f1",
    "gold",
    "label",
    "metric",
    "model",
    "oracle",
    "prediction",
    "score",
    "task",
)

BAMBOO_TASKS = (
    "abshallu_4k",
    "abshallu_16k",
    "senhallu_4k",
    "senhallu_16k",
)
BAMBOO_EXPECTED_ROWS = {task: 200 for task in BAMBOO_TASKS}

ETHIC_SPECS = {
    "Recalling": ("recalling.zip", "test_recalling.jsonl", 662),
    "Attributing": ("attributing.zip", "test_attributing.jsonl", 333),
}

HELMET_SPECS = {
    "kilt_hotpotqa": {
        "test": "hotpotqa-dev-multikilt_1000_k1000_dep3.jsonl",
        "demo": "hotpotqa-train-multikilt_1000_k3_dep3.jsonl",
        "expected": 2961,
        "key": "question",
    },
    "kilt_popqa_3": {
        "test": "popqa_test_1000_k1000_dep6.jsonl",
        "demo": "popqa_test_1000_k3_dep6.jsonl",
        "expected": 882,
        "key": "id",
    },
}

LONGTABLE_FORMATS = (
    "markdown",
    "html",
    "json",
    "latex",
    "sql",
    "xml",
    "csv",
)
LONGTABLE_QUESTION_FILES = (
    "datasets/questions/32k/longtablebench_multi.json",
    "datasets/questions/32k/longtablebench_single.json",
    "datasets/questions/8k/longtablebench_multi.json",
    "datasets/questions/8k/longtablebench_single.json",
    "datasets/questions/inf/longtablebench_multi.json",
    "datasets/questions/inf/longtablebench_single.json",
)

EXPECTED_FAMILY_COUNTS = {
    "BAMBOO": {"evaluations": 800, "scored_units": 800},
    "ETHIC": {"evaluations": 995, "scored_units": 995},
    "HELMET": {"evaluations": 3843, "scored_units": 3843},
    "LongTableBench": {"evaluations": 6671, "scored_units": 12397},
}
EXPECTED_TOTAL_EVALUATIONS = 12309
EXPECTED_TOTAL_SCORED_UNITS = 18035


LONGTABLE_SYSTEM_SINGLE = """### Requirements:
Please read the following table and then answer the questions based on the table.  
Organize your answers into a list of strings, with each element being an answer item (The number of answer items is less than 11). If the question includes a special requirement, such as outputting a dictionary, please fulfill that specific request. Otherwise, always output a list of strings, even if there is only one answer item.  
Please place your answers between the ``` and ```.

### Notes:
The table may include non-standard formats for numbers or dates. 

For numbers, formats may include Roman numerals, English words, or scientific notation. Whenever possible, please convert these into Arabic numerals in your response.

For dates, the following formats may be encountered:
%Y-%m-%d <other time elements> (in Arabic numerals)
%d/%m/%Y <other time elements> (in Arabic numerals)
%m.%d.%Y <other time elements> (in Arabic numerals)
%Y.%m.%d <other time elements> (in English words for numbers)
%Y-%m-%d <other time elements> (in Roman numerals)
Please convert all dates to the format %Y-%m-%d <other time elements> (in Arabic numerals) in your response.

Additionally, even if there are duplicate answer items, you need to output all of them.

# Example Output Format:  
list[str]:
```
['answer_item_1', 'answer_item_2', ..., 'answer_item_n']
```

dict:
```
{'column_1': value_1, 'column_2': value_2, ..., 'column_n': value_n}
```

Please do not generate any text after outputting the final answer."""

LONGTABLE_SYSTEM_MULTI = """### Requirements:
Please read the following table and then answer the questions based on the table.  
Organize your answers into a list of strings, with each element being an answer item. If the question includes a special requirement, such as outputting a dictionary, please fulfill that specific request. Otherwise, always output a list of strings, even if there is only one answer item.  
Please place your answers between the ``` and ```.

### Notes:
The table may include non-standard formats for numbers or dates. 

For numbers, formats may include Roman numerals, English words, or scientific notation. Whenever possible, please convert these into Arabic numerals in your response.

For dates, the following formats may be encountered:
%Y-%m-%d <other time elements> (in Arabic numerals)
%d/%m/%Y <other time elements> (in Arabic numerals)
%m.%d.%Y <other time elements> (in Arabic numerals)
%Y.%m.%d <other time elements> (in English words for numbers)
%Y-%m-%d <other time elements> (in Roman numerals)
Please convert all dates to the format %Y-%m-%d <other time elements> (in Arabic numerals) in your response.

Additionally, if there are duplicate answer_items, they should be output repeatedly (i.e., no deduplication).

# Example Output Format:  
list[str]:
```
['answer_item_1', 'answer_item_2', ..., 'answer_item_n']
```

dict:
```
{'column_1': value_1, 'column_2': value_2, ..., 'column_n': value_n}
```

Please do not generate any text after outputting the final answer."""

HELMET_USER_TEMPLATE = (
    "Use the given documents to write a concise and short answer to the "
    "question. Write your answer in the following format:\n"
    "Answer: [answer]\n\n{demos}{context}\n\nQuestion: {question}"
)
HELMET_SYSTEM_TEMPLATE = "Answer:"
HELMET_PASSAGE_TEMPLATE = "Document (Title: {title}): {text}"
HELMET_DEMO_TEMPLATE = (
    "{documents}\n\nQuestion: {question}\nAnswer: {answer}"
)


@dataclass(frozen=True)
class AdapterRound:
    """One independently routed scored unit."""

    query_key: str
    question: str
    round_index: int


@dataclass(frozen=True)
class ChunkRef:
    """Private reconstruction metadata; never passed to the selector."""

    source: str
    table_name: str | None = None
    row_index: int | None = None


@dataclass(frozen=True)
class AdapterEvaluation:
    """One official record/conversation evaluation."""

    family: str
    task: str
    evaluation_key: str
    chunks: tuple[str, ...]
    rounds: tuple[AdapterRound, ...]
    generation: Mapping[str, Any]
    chunk_refs: tuple[ChunkRef, ...] = ()

    def selector_stub(self, round_index: int) -> dict[str, Any]:
        round_spec = self.rounds[round_index]
        value = {
            "query_key": round_spec.query_key,
            "question": round_spec.question,
            "chunks": list(self.chunks),
        }
        validate_selector_stub(value)
        return value

    def runtime_query(
        self,
        round_index: int,
        methods: Mapping[str, Any],
    ) -> dict[str, Any]:
        value = self.selector_stub(round_index)
        value["methods"] = dict(methods)
        if set(value) != set(RUNTIME_QUERY_FIELDS):
            raise ValueError("runtime query schema mismatch")
        return value


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def opaque_key(*parts: Any) -> str:
    payload = canonical_json([str(part) for part in parts])
    return "q_" + sha256_text(payload)[:32]


def validate_selector_stub(value: Mapping[str, Any]) -> None:
    if set(value) != set(SELECTOR_STUB_FIELDS):
        raise ValueError(
            "selector schema mismatch: "
            f"observed={sorted(value)}, expected={sorted(SELECTOR_STUB_FIELDS)}"
        )
    for key in value:
        lowered = str(key).lower()
        if any(
            fragment in lowered
            for fragment in FORBIDDEN_SELECTOR_FIELD_FRAGMENTS
        ):
            raise ValueError(f"forbidden selector field: {key}")
    query_key = value["query_key"]
    question = value["question"]
    chunks = value["chunks"]
    if not isinstance(query_key, str) or not query_key.startswith("q_"):
        raise ValueError("query_key must be an opaque q_ hash")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if (
        not isinstance(chunks, list)
        or not chunks
        or any(not isinstance(chunk, str) or not chunk.strip() for chunk in chunks)
    ):
        raise ValueError("chunks must be a non-empty string list")


def split_context(
    context: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> tuple[str, ...]:
    """Use the exact splitter configuration frozen by prior experiments."""

    if not 0 <= overlap < chunk_size:
        raise ValueError("require 0 <= overlap < chunk_size")
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "！", "？", "，", ". ", " "],
    )
    chunks = tuple(
        chunk.strip()
        for chunk in splitter.split_text(str(context))
        if chunk.strip()
    )
    if chunks:
        return chunks
    stripped = str(context).strip()
    if stripped:
        return (stripped,)
    raise ValueError("context produced no chunks")


def selected_context(
    evaluation: AdapterEvaluation,
    selected_indices: Sequence[int],
) -> str:
    indices = _validate_selected_indices(evaluation, selected_indices)
    return "\n\n".join(evaluation.chunks[index] for index in indices)


def _validate_selected_indices(
    evaluation: AdapterEvaluation,
    selected_indices: Sequence[int],
) -> tuple[int, ...]:
    values = tuple(int(index) for index in selected_indices)
    if not values:
        raise ValueError("at least one selected chunk is required")
    if len(values) != len(set(values)):
        raise ValueError("selected chunk indices contain duplicates")
    if any(not 0 <= index < len(evaluation.chunks) for index in values):
        raise ValueError("selected chunk index out of range")
    return values


def iter_bamboo_evaluations(root: Path) -> Iterator[AdapterEvaluation]:
    prompt_path = root / "prompt.json"
    prompts = json.loads(prompt_path.read_text(encoding="utf-8"))
    for task in BAMBOO_TASKS:
        prompt_name = task.split("_", 1)[0]
        prompt = prompts[prompt_name]
        path = root / "datasets" / f"{task}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                # Deliberately never index row["answer"].
                content = row["content"]
                hypothesis = row["hypothesis"]
                if not isinstance(content, str) or not isinstance(hypothesis, str):
                    raise ValueError(f"{path}:{row_index + 1}: invalid text fields")
                chunks = split_context(content)
                evaluation_key = opaque_key("BAMBOO-eval", task, row_index)
                round_spec = AdapterRound(
                    query_key=opaque_key("BAMBOO-query", task, row_index),
                    question=hypothesis,
                    round_index=0,
                )
                yield AdapterEvaluation(
                    family="BAMBOO",
                    task=task,
                    evaluation_key=evaluation_key,
                    chunks=chunks,
                    rounds=(round_spec,),
                    generation={
                        "kind": "bamboo_hallucination",
                        "prompt_name": prompt_name,
                        "system": str(prompt["system"]),
                        "final_answer_template": str(prompt["final_answer"]),
                        "hypothesis": hypothesis,
                        "dense_context": content,
                        "dense_context_sha256": sha256_text(content),
                    },
                )


def render_bamboo_prompt(
    evaluation: AdapterEvaluation,
    selected_indices: Sequence[int],
) -> str:
    if evaluation.generation.get("kind") != "bamboo_hallucination":
        raise ValueError("not a BAMBOO hallucination evaluation")
    context = selected_context(evaluation, selected_indices)
    return str(evaluation.generation["final_answer_template"]).format(
        content=context,
        hypothesis=evaluation.generation["hypothesis"],
    )


def render_bamboo_dense_prompt(evaluation: AdapterEvaluation) -> str:
    """Rebuild the uncompressed official BAMBOO user prompt exactly."""

    if evaluation.generation.get("kind") != "bamboo_hallucination":
        raise ValueError("not a BAMBOO hallucination evaluation")
    return str(evaluation.generation["final_answer_template"]).format(
        content=evaluation.generation["dense_context"],
        hypothesis=evaluation.generation["hypothesis"],
    )


def iter_ethic_evaluations(root: Path) -> Iterator[AdapterEvaluation]:
    for task, (archive_name, member_name, _) in ETHIC_SPECS.items():
        archive_path = root / archive_name
        with zipfile.ZipFile(archive_path) as archive:
            with archive.open(member_name, "r") as handle:
                for row_index, raw in enumerate(handle):
                    row = json.loads(raw)
                    # Deliberately never index row["Answer"].
                    context = row["Context"]
                    user_msg = row["User_msg"]
                    system_msg = row["System_msg"]
                    source_id = row["ID"]
                    if not all(
                        isinstance(value, str)
                        for value in (context, user_msg, system_msg, source_id)
                    ):
                        raise ValueError(
                            f"{archive_name}:{row_index + 1}: invalid text fields"
                        )
                    if user_msg.count(context) != 1:
                        raise ValueError(
                            f"{archive_name}:{row_index + 1}: Context must occur once"
                        )
                    start = user_msg.index(context)
                    prefix = user_msg[:start]
                    suffix = user_msg[start + len(context) :]
                    selector_question = (
                        system_msg + "\n\n" + prefix + suffix
                    ).strip()
                    chunks = split_context(context)
                    yield AdapterEvaluation(
                        family="ETHIC",
                        task=task,
                        evaluation_key=opaque_key(
                            "ETHIC-eval", task, source_id, row_index
                        ),
                        chunks=chunks,
                        rounds=(
                            AdapterRound(
                                query_key=opaque_key(
                                    "ETHIC-query", task, source_id, row_index
                                ),
                                question=selector_question,
                                round_index=0,
                            ),
                        ),
                        generation={
                            "kind": "ethic_context_slot",
                            "system_msg": system_msg,
                            "prefix": prefix,
                            "suffix": suffix,
                            "dense_context": context,
                            "original_user_sha256": sha256_text(user_msg),
                            "original_context_sha256": sha256_text(context),
                        },
                    )


def render_ethic_messages(
    evaluation: AdapterEvaluation,
    selected_indices: Sequence[int],
) -> list[dict[str, str]]:
    if evaluation.generation.get("kind") != "ethic_context_slot":
        raise ValueError("not an ETHIC evaluation")
    compressed = selected_context(evaluation, selected_indices)
    user_msg = (
        str(evaluation.generation["prefix"])
        + compressed
        + str(evaluation.generation["suffix"])
    )
    return [
        {
            "role": "system",
            "content": str(evaluation.generation["system_msg"]),
        },
        {"role": "user", "content": user_msg},
    ]


def render_ethic_dense_messages(
    evaluation: AdapterEvaluation,
) -> list[dict[str, str]]:
    """Rebuild the uncompressed official ETHIC messages exactly."""

    if evaluation.generation.get("kind") != "ethic_context_slot":
        raise ValueError("not an ETHIC evaluation")
    user_msg = (
        str(evaluation.generation["prefix"])
        + str(evaluation.generation["dense_context"])
        + str(evaluation.generation["suffix"])
    )
    return [
        {
            "role": "system",
            "content": str(evaluation.generation["system_msg"]),
        },
        {"role": "user", "content": user_msg},
    ]


class HelmetDemoSampler:
    """Exact low-cost replay of HELMET's deterministic two-shot sampler."""

    def __init__(self, data_root: Path, task: str, shots: int = 2):
        if task not in HELMET_SPECS:
            raise ValueError(f"unknown HELMET task: {task}")
        self.task = task
        self.shots = int(shots)
        self.key_name = str(HELMET_SPECS[task]["key"])
        demo_path = data_root / str(HELMET_SPECS[task]["demo"])
        rows: list[dict[str, Any]] = []
        with demo_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if task == "kilt_popqa_3" and not helmet_popqa_eligible(row):
                    continue
                rows.append(row)
        self.rows = rows

    def selected_indices(self, sample_key: Any) -> tuple[int, ...]:
        if self.shots <= 0:
            return ()
        import numpy as np

        seed = (
            int(
                hashlib.sha256(str(sample_key).encode("utf-8")).hexdigest(),
                16,
            )
            % (2**31)
        )
        sample_key_string = str(sample_key)
        eligible_indices = [
            index
            for index, row in enumerate(self.rows)
            if not (
                self.task == "kilt_popqa_3"
                and str(row[self.key_name]) == sample_key_string
            )
        ]
        order = np.random.default_rng(seed).permutation(len(eligible_indices))
        selected: list[int] = []
        seen_keys: set[str] = set()
        for raw_position in order:
            index = eligible_indices[int(raw_position)]
            row = self.rows[index]
            key = str(row[self.key_name])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            selected.append(index)
            if len(selected) == self.shots:
                break
        if len(selected) != self.shots:
            raise ValueError(
                f"{self.task}: only selected {len(selected)}/{self.shots} demos"
            )
        return tuple(selected)

    def render(self, sample_key: Any) -> str:
        if self.shots <= 0:
            return ""
        rendered = []
        for index in self.selected_indices(sample_key):
            row = self.rows[index]
            documents = "\n\n".join(
                HELMET_PASSAGE_TEMPLATE.format(
                    title=context["title"],
                    text=context["text"],
                )
                for context in row["ctxs"]
            )
            # Released demonstration answers are official input prompt content.
            answer = row["answers"][0]
            rendered.append(
                HELMET_DEMO_TEMPLATE.format(
                    documents=documents,
                    question=row["question"],
                    answer=answer,
                )
            )
        return "\n\n".join(rendered) + "\n\n"


def helmet_popqa_eligible(row: Mapping[str, Any]) -> bool:
    return math.log10(float(row["s_pop"])) < 3.0


def iter_helmet_evaluations(
    data_root: Path,
    *,
    shots: int = 2,
) -> Iterator[AdapterEvaluation]:
    for task, spec in HELMET_SPECS.items():
        sampler = HelmetDemoSampler(data_root, task, shots=shots)
        path = data_root / str(spec["test"])
        with path.open(encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                if task == "kilt_popqa_3" and not helmet_popqa_eligible(row):
                    continue
                # Deliberately never index row["answers"], possible_answers,
                # obj, or any has_answer field from the untouched test row.
                question = row["question"]
                sample_key = row[str(spec["key"])]
                contexts = row["ctxs"]
                if not isinstance(question, str) or not isinstance(contexts, list):
                    raise ValueError(f"{path}:{row_index + 1}: invalid schema")
                chunks = tuple(
                    HELMET_PASSAGE_TEMPLATE.format(
                        title=context["title"],
                        text=context["text"],
                    )
                    for context in contexts
                )
                if not chunks:
                    raise ValueError(f"{path}:{row_index + 1}: empty contexts")
                depth = row.get("depth", "")
                yield AdapterEvaluation(
                    family="HELMET",
                    task=task,
                    evaluation_key=opaque_key(
                        "HELMET-eval", task, row_index, sample_key, depth
                    ),
                    chunks=chunks,
                    rounds=(
                        AdapterRound(
                            query_key=opaque_key(
                                "HELMET-query",
                                task,
                                row_index,
                                sample_key,
                                depth,
                            ),
                            question=question,
                            round_index=0,
                        ),
                    ),
                    generation={
                        "kind": "helmet_rag",
                        "question": question,
                        "demos": sampler.render(sample_key),
                        "shots": shots,
                        "user_template": HELMET_USER_TEMPLATE,
                        "system_template": HELMET_SYSTEM_TEMPLATE,
                    },
                )


def render_helmet_prompt(
    evaluation: AdapterEvaluation,
    selected_indices: Sequence[int],
) -> str:
    if evaluation.generation.get("kind") != "helmet_rag":
        raise ValueError("not a HELMET evaluation")
    context = selected_context(evaluation, selected_indices)
    user = str(evaluation.generation["user_template"]).format(
        demos=evaluation.generation["demos"],
        context=context,
        question=evaluation.generation["question"],
    )
    return user + "\n" + str(evaluation.generation["system_template"])


@functools.lru_cache(maxsize=None)
def _load_longtable_reformat(root: Path):
    path = root / "eval" / "reformat.py"
    name = "parc_longtable_reformat_" + sha256_text(str(path))[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _longtable_files(root: Path, item: Mapping[str, Any]) -> list[Path]:
    db_path = str(item["db_path"])
    db_id = str(item["db_id"])
    path = Path(db_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe LongTableBench db_path: {db_path}")
    directory = root / "datasets" / "tables" / db_path / db_id
    # Preserve the official implementation's os.listdir order.  The pinned
    # execution manifest binds this order because LongTableBench's official
    # load_tables() does not sort multi-table files.
    files = [
        directory / name
        for name in os.listdir(directory)
        if name.endswith(".csv") and (directory / name).is_file()
    ]
    if not bool(item["is_multi_table"]):
        highlighted = item.get("highlighted_table") or []
        if not highlighted:
            raise ValueError("single-table LongTableBench row lacks highlighted_table")
        wanted = f"{highlighted[0]}.csv".casefold()
        files = [candidate for candidate in files if candidate.name.casefold() == wanted]
    if not files:
        raise ValueError(
            f"LongTableBench table resolution failed for {db_path}/{db_id}"
        )
    return files


def _scalar_text(value: Any) -> str:
    try:
        if value is None or bool(value != value):
            return "<NA>"
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    return str(value)


def _longtable_chunk_universe(
    root: Path,
    item: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[ChunkRef, ...]]:
    reformat = _load_longtable_reformat(root)
    chunks: list[str] = []
    refs: list[ChunkRef] = []
    multi = bool(item["is_multi_table"])
    highlighted_name = (
        None if multi else str(item["highlighted_table"][0])
    )
    for path in _longtable_files(root, item):
        frame = reformat.pd.read_csv(path, low_memory=False)
        columns = " | ".join(str(column) for column in frame.columns)
        relative = path.relative_to(root).as_posix()
        table_name = path.stem if multi else str(highlighted_name)
        if len(frame) == 0:
            chunks.append(
                f"Table: {table_name}\nColumns: {columns}\nRow: <EMPTY>"
            )
            refs.append(
                ChunkRef(
                    source=relative,
                    table_name=table_name,
                    row_index=None,
                )
            )
            continue
        for row_index, values in enumerate(frame.itertuples(index=False, name=None)):
            rendered_values = " | ".join(_scalar_text(value) for value in values)
            chunks.append(
                f"Table: {table_name}\nColumns: {columns}\n"
                f"Row {row_index}: {rendered_values}"
            )
            refs.append(
                ChunkRef(
                    source=relative,
                    table_name=table_name,
                    row_index=row_index,
                )
            )
    if not chunks:
        raise ValueError("LongTableBench row produced no canonical chunks")
    return tuple(chunks), tuple(refs)


def _longtable_round_questions(
    item: Mapping[str, Any],
    *,
    is_multi_turn: bool,
) -> tuple[str, ...]:
    if is_multi_turn:
        qa_rows = item["QA"]
        questions = [str(row["question"]) for row in qa_rows]
        evidence = str(item.get("evidence", "")).strip()
        if evidence:
            questions[0] = evidence + " " + questions[0]
        return tuple(questions)
    question = str(item["question"])
    evidence = str(item.get("evidence", "")).strip()
    if evidence:
        question = evidence + " " + question
    return (question,)


def iter_longtable_evaluations(root: Path) -> Iterator[AdapterEvaluation]:
    universe_cache: dict[
        tuple[str, ...],
        tuple[tuple[str, ...], tuple[ChunkRef, ...]],
    ] = {}
    for relative_question_file in LONGTABLE_QUESTION_FILES:
        question_path = root / relative_question_file
        rows = json.loads(question_path.read_text(encoding="utf-8"))
        for row_index, item in enumerate(rows):
            # Deliberately never index item["answer"] or QA[*]["answer"].
            files = _longtable_files(root, item)
            cache_key = tuple(path.relative_to(root).as_posix() for path in files)
            if cache_key not in universe_cache:
                universe_cache[cache_key] = _longtable_chunk_universe(root, item)
            chunks, refs = universe_cache[cache_key]
            is_multi_turn = relative_question_file.endswith(
                "longtablebench_multi.json"
            )
            is_multi_table = bool(item["is_multi_table"])
            questions = _longtable_round_questions(
                item,
                is_multi_turn=is_multi_turn,
            )
            length_band = relative_question_file.split("/")[2]
            content_type = "multi" if is_multi_turn else "single"
            for table_format in LONGTABLE_FORMATS:
                task = f"{length_band}_{content_type}_{table_format}"
                rounds = tuple(
                    AdapterRound(
                        query_key=opaque_key(
                            "LongTableBench-query",
                            relative_question_file,
                            row_index,
                            table_format,
                            round_index,
                        ),
                        question=question,
                        round_index=round_index,
                    )
                    for round_index, question in enumerate(questions)
                )
                yield AdapterEvaluation(
                    family="LongTableBench",
                    task=task,
                    evaluation_key=opaque_key(
                        "LongTableBench-eval",
                        relative_question_file,
                        row_index,
                        table_format,
                    ),
                    chunks=chunks,
                    rounds=rounds,
                    generation={
                        "kind": "longtablebench",
                        "table_format": table_format,
                        "is_multi_turn": is_multi_turn,
                        "is_multi_table": is_multi_table,
                        "round_questions": questions,
                        "question_file": relative_question_file,
                        "question_row_index": row_index,
                    },
                    chunk_refs=refs,
                )


def render_longtable_context(
    evaluation: AdapterEvaluation,
    selected_indices: Sequence[int],
    root: Path,
) -> str:
    if evaluation.generation.get("kind") != "longtablebench":
        raise ValueError("not a LongTableBench evaluation")
    indices = _validate_selected_indices(evaluation, selected_indices)
    if len(evaluation.chunk_refs) != len(evaluation.chunks):
        raise ValueError("LongTableBench chunk reference coverage mismatch")
    reformat = _load_longtable_reformat(root)
    grouped: dict[str, list[int | None]] = {}
    table_names: dict[str, str] = {}
    for index in indices:
        ref = evaluation.chunk_refs[index]
        grouped.setdefault(ref.source, []).append(ref.row_index)
        table_names[ref.source] = str(ref.table_name)
    parts: list[str] = []
    for source, row_indices in grouped.items():
        path = root / source
        frame = reformat.pd.read_csv(path, low_memory=False)
        concrete = [index for index in row_indices if index is not None]
        subset = frame.iloc[concrete] if concrete else frame.iloc[0:0]
        parts.append(
            reformat.trans_tables(
                table_names[source],
                subset,
                str(evaluation.generation["table_format"]),
            )
        )
    return "".join(parts)


def build_longtable_messages(
    evaluation: AdapterEvaluation,
    selected_indices: Sequence[int],
    root: Path,
    *,
    round_index: int,
    previous_outputs: Sequence[str] = (),
) -> list[dict[str, str]]:
    if evaluation.generation.get("kind") != "longtablebench":
        raise ValueError("not a LongTableBench evaluation")
    if not 0 <= round_index < len(evaluation.rounds):
        raise ValueError("LongTableBench round index out of range")
    if len(previous_outputs) != round_index:
        raise ValueError(
            "previous_outputs must contain exactly one real output per prior round"
        )
    table = render_longtable_context(evaluation, selected_indices, root)
    questions = tuple(evaluation.generation["round_questions"])
    multi_turn = bool(evaluation.generation["is_multi_turn"])
    messages = [
        {
            "role": "system",
            "content": (
                LONGTABLE_SYSTEM_MULTI
                if multi_turn
                else LONGTABLE_SYSTEM_SINGLE
            ),
        },
        {
            "role": "user",
            "content": (
                f"table information:\n{table}\n\n"
                f"Question: {questions[0]}\nAnswer:"
            ),
        },
    ]
    for index in range(1, round_index + 1):
        messages.append(
            {"role": "system", "content": str(previous_outputs[index - 1])}
        )
        messages.append(
            {
                "role": "user",
                "content": f"Question: {questions[index]}\nAnswer:",
            }
        )
    return messages


def iter_all_evaluations(
    *,
    bamboo_root: Path,
    ethic_root: Path,
    helmet_data_root: Path,
    longtable_root: Path,
) -> Iterator[AdapterEvaluation]:
    yield from iter_bamboo_evaluations(bamboo_root)
    yield from iter_ethic_evaluations(ethic_root)
    yield from iter_helmet_evaluations(helmet_data_root)
    yield from iter_longtable_evaluations(longtable_root)


def _new_family_state() -> dict[str, Any]:
    return {
        "evaluations": 0,
        "scored_units": 0,
        "tasks": Counter(),
        "chunk_count_distribution": Counter(),
        "chunk_count_min": None,
        "chunk_count_max": None,
        "queries_with_at_most_top_k_chunks": 0,
        "queries_with_single_chunk": 0,
        "selector_payload_digest": hashlib.sha256(),
    }


def run_schema_only_dry_run(
    *,
    bamboo_root: Path,
    ethic_root: Path,
    helmet_data_root: Path,
    longtable_root: Path,
) -> dict[str, Any]:
    states: dict[str, dict[str, Any]] = defaultdict(_new_family_state)
    query_keys: set[str] = set()
    evaluation_keys: set[str] = set()
    duplicate_query_keys: list[str] = []
    duplicate_evaluation_keys: list[str] = []
    errors: list[dict[str, Any]] = []

    for evaluation_index, evaluation in enumerate(
        iter_all_evaluations(
            bamboo_root=bamboo_root,
            ethic_root=ethic_root,
            helmet_data_root=helmet_data_root,
            longtable_root=longtable_root,
        ),
        start=1,
    ):
        state = states[evaluation.family]
        state["evaluations"] += 1
        state["scored_units"] += len(evaluation.rounds)
        state["tasks"][evaluation.task] += 1
        if evaluation.evaluation_key in evaluation_keys:
            duplicate_evaluation_keys.append(evaluation.evaluation_key)
        evaluation_keys.add(evaluation.evaluation_key)
        chunk_count = len(evaluation.chunks)
        state["chunk_count_distribution"][str(chunk_count)] += len(
            evaluation.rounds
        )
        current_min = state["chunk_count_min"]
        current_max = state["chunk_count_max"]
        state["chunk_count_min"] = (
            chunk_count if current_min is None else min(current_min, chunk_count)
        )
        state["chunk_count_max"] = (
            chunk_count if current_max is None else max(current_max, chunk_count)
        )
        if chunk_count <= TOP_K:
            state["queries_with_at_most_top_k_chunks"] += len(evaluation.rounds)
        if chunk_count == 1:
            state["queries_with_single_chunk"] += len(evaluation.rounds)

        chunk_digest = hashlib.sha256()
        for chunk in evaluation.chunks:
            chunk_digest.update(chunk.encode("utf-8"))
            chunk_digest.update(b"\0")
        for round_index in range(len(evaluation.rounds)):
            try:
                stub = evaluation.selector_stub(round_index)
            except Exception as exc:
                if len(errors) < 20:
                    errors.append(
                        {
                            "evaluation_index": evaluation_index,
                            "error": type(exc).__name__,
                            "message": str(exc)[:300],
                        }
                    )
                continue
            query_key = stub["query_key"]
            if query_key in query_keys:
                duplicate_query_keys.append(query_key)
            query_keys.add(query_key)
            digest_row = {
                "query_key": query_key,
                "question_sha256": sha256_text(stub["question"]),
                "chunks": chunk_count,
                "chunks_sha256": chunk_digest.hexdigest(),
            }
            state["selector_payload_digest"].update(
                (canonical_json(digest_row) + "\n").encode("utf-8")
            )

    family_summaries: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for family in sorted(EXPECTED_FAMILY_COUNTS):
        state = states[family]
        expected = EXPECTED_FAMILY_COUNTS[family]
        family_summaries[family] = {
            "evaluations": state["evaluations"],
            "scored_units": state["scored_units"],
            "expected_evaluations": expected["evaluations"],
            "expected_scored_units": expected["scored_units"],
            "tasks": dict(sorted(state["tasks"].items())),
            "chunk_count_distribution": dict(
                sorted(
                    state["chunk_count_distribution"].items(),
                    key=lambda item: int(item[0]),
                )
            ),
            "chunk_count_min": state["chunk_count_min"],
            "chunk_count_max": state["chunk_count_max"],
            "queries_with_at_most_top_k_chunks": state[
                "queries_with_at_most_top_k_chunks"
            ],
            "queries_with_single_chunk": state["queries_with_single_chunk"],
            "selector_payload_digest_sha256": state[
                "selector_payload_digest"
            ].hexdigest(),
        }
        checks[f"{family}_evaluation_count"] = (
            state["evaluations"] == expected["evaluations"]
        )
        checks[f"{family}_scored_unit_count"] = (
            state["scored_units"] == expected["scored_units"]
        )

    total_evaluations = sum(
        item["evaluations"] for item in family_summaries.values()
    )
    total_scored_units = sum(
        item["scored_units"] for item in family_summaries.values()
    )
    checks.update(
        {
            "four_families_present": (
                set(family_summaries) == set(EXPECTED_FAMILY_COUNTS)
            ),
            "total_evaluations": (
                total_evaluations == EXPECTED_TOTAL_EVALUATIONS
            ),
            "total_scored_units": (
                total_scored_units == EXPECTED_TOTAL_SCORED_UNITS
            ),
            "selector_schema_all_rows": not errors,
            "query_keys_unique": not duplicate_query_keys,
            "evaluation_keys_unique": not duplicate_evaluation_keys,
        }
    )
    passed = all(checks.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "protocol": PROTOCOL,
        "mode": "SCHEMA_ONLY_PREDICTION_FREE",
        "checks": checks,
        "families": family_summaries,
        "totals": {
            "benchmark_families": len(family_summaries),
            "record_or_conversation_evaluations": total_evaluations,
            "scored_units_and_selector_queries": total_scored_units,
            "unique_evaluation_keys": len(evaluation_keys),
            "unique_query_keys": len(query_keys),
        },
        "failures": {
            "selector_schema_first20": errors,
            "duplicate_query_keys_first20": duplicate_query_keys[:20],
            "duplicate_evaluation_keys_first20": duplicate_evaluation_keys[:20],
        },
        "data_access_attestation": {
            "untouched_test_answer_values_read": False,
            "prediction_values_read": False,
            "score_values_read": False,
            "helmet_released_demo_answer_values_read_for_official_prompt": True,
        },
        "compression_note": (
            "This dry run verifies input universes and reports low-chunk rows. "
            "Strict positive compression remains a per-task result gate after "
            "frozen retrieval decisions; schema counts are not quality claims."
        ),
    }


def write_selector_stubs(
    *,
    bamboo_root: Path,
    ethic_root: Path,
    helmet_data_root: Path,
    longtable_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Write selector-only JSONL without answers, labels, or quality fields."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    digest = hashlib.sha256()
    queries = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for evaluation in iter_all_evaluations(
            bamboo_root=bamboo_root,
            ethic_root=ethic_root,
            helmet_data_root=helmet_data_root,
            longtable_root=longtable_root,
        ):
            for round_index in range(len(evaluation.rounds)):
                payload = evaluation.selector_stub(round_index)
                line = canonical_json(payload) + "\n"
                handle.write(line)
                digest.update(line.encode("utf-8"))
                queries += 1
    os.replace(temporary, output_path)
    return {
        "path": str(output_path),
        "queries": queries,
        "sha256": digest.hexdigest(),
        "contains_answers": False,
        "contains_predictions": False,
        "contains_scores": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bamboo-root", type=Path, required=True)
    parser.add_argument("--ethic-root", type=Path, required=True)
    parser.add_argument("--helmet-data-root", type=Path, required=True)
    parser.add_argument("--longtable-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selector-stub-output", type=Path)
    args = parser.parse_args()

    summary = run_schema_only_dry_run(
        bamboo_root=args.bamboo_root.resolve(),
        ethic_root=args.ethic_root.resolve(),
        helmet_data_root=args.helmet_data_root.resolve(),
        longtable_root=args.longtable_root.resolve(),
    )
    output_dir = args.output_dir.resolve()
    summary_path = output_dir / "parc_adapter_schema_dry_run_summary.json"
    write_json_atomic(summary_path, summary)
    summary_hash = sha256_file(summary_path)
    selector_stub_artifact = None
    if args.selector_stub_output is not None:
        selector_stub_artifact = write_selector_stubs(
            bamboo_root=args.bamboo_root.resolve(),
            ethic_root=args.ethic_root.resolve(),
            helmet_data_root=args.helmet_data_root.resolve(),
            longtable_root=args.longtable_root.resolve(),
            output_path=args.selector_stub_output.resolve(),
        )
    status = {
        "status": (
            "COMPLETE_VERIFIED_SCHEMA_ONLY"
            if summary["status"] == "PASS"
            else "FAILED"
        ),
        "summary": str(summary_path),
        "summary_sha256": summary_hash,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "formal_generation_started": False,
        "predictions_or_scores_read": False,
    }
    if selector_stub_artifact is not None:
        status["selector_stub_artifact"] = selector_stub_artifact
    write_json_atomic(output_dir / "pipeline_status.json", status)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "summary": str(summary_path),
                "summary_sha256": summary_hash,
                "totals": summary["totals"],
                "selector_stub_artifact": selector_stub_artifact,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
