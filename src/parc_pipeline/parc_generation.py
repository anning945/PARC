"""Prompt reconstruction, prediction-blind packing, and greedy generation."""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SEGMENT_KINDS = frozenset({"protected", "history", "context", "instruction"})


@dataclass(frozen=True)
class PromptSegment:
    kind: str
    text: str
    label: str

    def __post_init__(self) -> None:
        if (
            self.kind not in SEGMENT_KINDS
            or not self.label
            or not isinstance(self.text, str)
        ):
            raise ValueError("invalid prompt segment")


@dataclass(frozen=True)
class PromptMessage:
    role: str
    segments: tuple[PromptSegment, ...]

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"} or not self.segments:
            raise ValueError("invalid prompt message")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def messages_from_plan(plan: Sequence[PromptMessage]) -> list[dict[str, str]]:
    return [
        {
            "role": message.role,
            "content": "".join(segment.text for segment in message.segments),
        }
        for message in plan
    ]


def build_prompt_plan(
    evaluation: Any,
    selected_indices: Sequence[int],
    round_index: int,
    previous_outputs: Sequence[str],
    longtable_root: Path,
    adapters: Any,
) -> tuple[PromptMessage, ...]:
    if evaluation.family == "BAMBOO":
        if round_index or previous_outputs:
            raise ValueError("BAMBOO must be single turn")
        sentinel = "<PARC_CONTEXT_SLOT>"
        rendered = str(evaluation.generation["final_answer_template"]).format(
            content=sentinel, hypothesis=evaluation.generation["hypothesis"]
        )
        if rendered.count(sentinel) != 1:
            raise ValueError("BAMBOO context slot mismatch")
        prefix, suffix = rendered.split(sentinel)
        return (
            PromptMessage(
                "system",
                (
                    PromptSegment(
                        "protected",
                        str(evaluation.generation["system"]),
                        "bamboo.system",
                    ),
                ),
            ),
            PromptMessage(
                "user",
                (
                    PromptSegment("instruction", prefix, "bamboo.prefix"),
                    PromptSegment(
                        "context",
                        adapters.selected_context(evaluation, selected_indices),
                        "bamboo.context",
                    ),
                    PromptSegment("protected", suffix, "bamboo.suffix"),
                ),
            ),
        )
    if evaluation.family == "ETHIC":
        if round_index or previous_outputs:
            raise ValueError("ETHIC must be single turn")
        return (
            PromptMessage(
                "system",
                (
                    PromptSegment(
                        "protected",
                        str(evaluation.generation["system_msg"]),
                        "ethic.system",
                    ),
                ),
            ),
            PromptMessage(
                "user",
                (
                    PromptSegment(
                        "instruction",
                        str(evaluation.generation["prefix"]),
                        "ethic.prefix",
                    ),
                    PromptSegment(
                        "context",
                        adapters.selected_context(evaluation, selected_indices),
                        "ethic.context",
                    ),
                    PromptSegment(
                        "instruction",
                        str(evaluation.generation["suffix"]),
                        "ethic.suffix",
                    ),
                ),
            ),
        )
    if evaluation.family == "HELMET":
        if round_index or previous_outputs:
            raise ValueError("HELMET must be single turn")
        demo_slot, context_slot = "<PARC_DEMOS_SLOT>", "<PARC_CONTEXT_SLOT>"
        rendered = (
            str(evaluation.generation["user_template"]).format(
                demos=demo_slot,
                context=context_slot,
                question=evaluation.generation["question"],
            )
            + "\n"
            + str(evaluation.generation["system_template"])
        )
        if rendered.count(demo_slot) != 1 or rendered.count(context_slot) != 1:
            raise ValueError("HELMET prompt slot mismatch")
        prefix, remainder = rendered.split(demo_slot)
        middle, suffix = remainder.split(context_slot)
        return (
            PromptMessage(
                "user",
                (
                    PromptSegment("instruction", prefix, "helmet.prefix"),
                    PromptSegment(
                        "instruction",
                        str(evaluation.generation["demos"]),
                        "helmet.demos",
                    ),
                    PromptSegment("instruction", middle, "helmet.separator"),
                    PromptSegment(
                        "context",
                        adapters.selected_context(evaluation, selected_indices),
                        "helmet.context",
                    ),
                    PromptSegment("protected", suffix, "helmet.question"),
                ),
            ),
        )
    if evaluation.family == "LongTableBench":
        if len(previous_outputs) != round_index:
            raise ValueError("LongTableBench history length mismatch")
        context = adapters.render_longtable_context(
            evaluation, selected_indices, longtable_root
        )
        questions = tuple(evaluation.generation["round_questions"])
        system = (
            adapters.LONGTABLE_SYSTEM_MULTI
            if evaluation.generation["is_multi_turn"]
            else adapters.LONGTABLE_SYSTEM_SINGLE
        )
        plan: list[PromptMessage] = [
            PromptMessage(
                "system", (PromptSegment("protected", system, "longtable.system"),)
            ),
            PromptMessage(
                "user",
                (
                    PromptSegment(
                        "protected", "table information:\n", "longtable.table_prefix"
                    ),
                    PromptSegment("context", context, "longtable.context"),
                    PromptSegment(
                        "protected",
                        f"\n\nQuestion: {questions[0]}\nAnswer:",
                        "longtable.question.0",
                    ),
                ),
            ),
        ]
        for index in range(1, round_index + 1):
            # The pinned LongTableBench adapter represents prior outputs as
            # system-role history; preserve that unusual upstream contract.
            plan.append(
                PromptMessage(
                    "system",
                    (
                        PromptSegment(
                            "history",
                            str(previous_outputs[index - 1]),
                            f"longtable.answer.{index - 1}",
                        ),
                    ),
                )
            )
            plan.append(
                PromptMessage(
                    "user",
                    (
                        PromptSegment(
                            "protected",
                            f"Question: {questions[index]}\nAnswer:",
                            f"longtable.question.{index}",
                        ),
                    ),
                )
            )
        return tuple(plan)
    raise ValueError(f"unsupported family: {evaluation.family}")


def _render(tokenizer: Any, plan: Sequence[PromptMessage]) -> tuple[str, Any, int]:
    prompt = tokenizer.apply_chat_template(
        messages_from_plan(plan), tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False, truncation=False
    )
    input_ids = encoded["input_ids"]
    shape = getattr(input_ids, "shape", None)
    tokens = int(shape[-1]) if shape is not None else len(input_ids[0])
    if tokens <= 0:
        raise ValueError("prompt has no tokens")
    return str(prompt), encoded, tokens


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    return [
        int(value)
        for value in tokenizer.encode(text, add_special_tokens=False, truncation=False)
    ]


def _head_tail(
    tokenizer: Any, values: Sequence[int], retained: int, marker: str
) -> str:
    if not 0 <= retained < len(values):
        raise ValueError("invalid retained-token count")
    head = (retained + 1) // 2
    tail = retained - head
    prefix = (
        tokenizer.decode(
            list(values[:head]),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if head
        else ""
    )
    suffix = (
        tokenizer.decode(
            list(values[len(values) - tail :]),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if tail
        else ""
    )
    return str(prefix) + marker + str(suffix)


def _replace(
    plan: Sequence[PromptMessage], message_index: int, segment_index: int, text: str
) -> tuple[PromptMessage, ...]:
    result = list(plan)
    message = result[message_index]
    segments = list(message.segments)
    source = segments[segment_index]
    segments[segment_index] = PromptSegment(source.kind, text, source.label)
    result[message_index] = PromptMessage(message.role, tuple(segments))
    return tuple(result)


def _render_packed_candidate(
    tokenizer: Any,
    plan: Sequence[PromptMessage],
    message_index: int,
    segment_index: int,
    values: Sequence[int],
    retained: int,
    marker: str,
) -> tuple[tuple[PromptMessage, ...], str, Any, int]:
    candidate = _replace(
        plan,
        message_index,
        segment_index,
        _head_tail(tokenizer, values, retained, marker),
    )
    rendered, encoded, tokens = _render(tokenizer, candidate)
    return candidate, rendered, encoded, tokens


def pack_prompt(
    tokenizer: Any, plan: Sequence[PromptMessage], config: Mapping[str, Any]
) -> dict[str, Any]:
    decoding, packing = config["decoding"], config["packing"]
    maximum, target = (
        int(decoding["max_input_tokens"]),
        int(decoding["target_input_tokens"]),
    )
    priority = tuple(packing["priority"])
    if priority != ("history", "context", "instruction") or not 0 < target < maximum:
        raise ValueError("unsupported packing configuration")
    source_plan = tuple(plan)
    prompt, encoded, source_tokens = _render(tokenizer, source_plan)
    if source_tokens <= maximum:
        return {
            "prompt": prompt,
            "encoded": encoded,
            "input_tokens": source_tokens,
            "pre_pack_input_tokens": source_tokens,
            "packing_applied": False,
            "truncated_segments": [],
        }
    current = source_plan
    current_prompt, current_encoded, current_tokens = prompt, encoded, source_tokens
    truncated: list[dict[str, Any]] = []
    for kind in priority:
        candidates = [
            (message_index, segment_index, segment)
            for message_index, message in enumerate(source_plan)
            for segment_index, segment in enumerate(message.segments)
            if segment.kind == kind and segment.text
        ]
        for message_index, segment_index, source in candidates:
            active = current[message_index].segments[segment_index]
            values = _token_ids(tokenizer, active.text)
            if not values:
                continue

            low, high = 0, len(values) - 1
            candidate_args = (
                tokenizer,
                current,
                message_index,
                segment_index,
                values,
            )
            marker = str(packing["omission_marker"])
            chosen = _render_packed_candidate(*candidate_args, 0, marker)
            retained = 0
            if chosen[3] <= target:
                while low <= high:
                    middle = (low + high) // 2
                    attempt = _render_packed_candidate(*candidate_args, middle, marker)
                    if attempt[3] <= target:
                        chosen, retained, low = attempt, middle, middle + 1
                    else:
                        high = middle - 1
            current, current_prompt, current_encoded, current_tokens = chosen
            truncated.append(
                {
                    "label": source.label,
                    "kind": source.kind,
                    "source_sha256": sha256_text(active.text),
                    "source_tokens": len(values),
                    "retained_source_tokens": retained,
                }
            )
            if current_tokens <= target:
                break
        if current_tokens <= target:
            break
    if current_tokens > target:
        raise ValueError(
            f"prompt cannot be packed without modifying protected segments: {current_tokens} > {target}"
        )
    for message_index, message in enumerate(source_plan):
        for segment_index, segment in enumerate(message.segments):
            if (
                segment.kind == "protected"
                and current[message_index].segments[segment_index].text != segment.text
            ):
                raise AssertionError("protected prompt segment changed")
    return {
        "prompt": current_prompt,
        "encoded": current_encoded,
        "input_tokens": current_tokens,
        "pre_pack_input_tokens": source_tokens,
        "packing_applied": True,
        "truncated_segments": truncated,
    }


def load_generator(
    model_name_or_path: str, device: str, dtype_name: str, seed: int
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = "auto" if dtype_name == "auto" else getattr(torch, dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return model, tokenizer


def generate_one(
    model: Any,
    tokenizer: Any,
    packed: Mapping[str, Any],
    device: str,
    max_new_tokens: int,
    use_cache: bool,
) -> dict[str, Any]:
    import torch

    encoded = packed["encoded"]
    encoded = (
        encoded.to(device)
        if hasattr(encoded, "to")
        else {key: value.to(device) for key, value in encoded.items()}
    )
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=use_cache,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    output_ids = generated[0][int(packed["input_tokens"]) :]
    return {
        "prediction": tokenizer.decode(output_ids, skip_special_tokens=True),
        "input_tokens": int(packed["input_tokens"]),
        "pre_pack_input_tokens": int(packed["pre_pack_input_tokens"]),
        "output_tokens": int(len(output_ids)),
        "packing_applied": bool(packed["packing_applied"]),
        "truncated_segments": list(packed["truncated_segments"]),
        "generation_latency_ms": (time.perf_counter() - started) * 1000.0,
    }
