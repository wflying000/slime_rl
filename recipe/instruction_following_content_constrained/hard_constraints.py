"""Machine-checkable instruction-following constraints used by the reward."""

from __future__ import annotations

from typing import Any

from .instructions_registry import INSTRUCTION_DICT


def strip_chain_of_thought(text: str) -> str:
    """Remove hidden reasoning and trailing chat tokens from a response."""
    if not text:
        return ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return text.replace("<|im_end|>", "")


def _filter_null_kwargs(kwargs: dict[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in (kwargs or {}).items():
        if value is None:
            continue
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        result[key] = value
    return result


def strict_constraint_hits(
    response: str,
    instruction_ids: list[str],
    instruction_kwargs: list[dict[str, Any] | None],
) -> list[bool]:
    """Return one hard-constraint result for each instruction."""
    if len(instruction_ids) != len(instruction_kwargs):
        raise ValueError(
            f"instruction count ({len(instruction_ids)}) does not match "
            f"kwargs count ({len(instruction_kwargs)})"
        )

    hits: list[bool] = []
    for instruction_id, kwargs in zip(
        instruction_ids,
        instruction_kwargs,
        strict=True,
    ):
        instruction_class = INSTRUCTION_DICT[instruction_id]
        instruction = instruction_class(instruction_id)
        instruction.build_description(**_filter_null_kwargs(kwargs))
        hits.append(bool(response.strip()) and bool(instruction.check_following(response)))
    return hits
