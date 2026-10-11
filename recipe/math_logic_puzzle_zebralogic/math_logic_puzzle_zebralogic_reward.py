"""MILES reward for ``math_logic_puzzle_zebralogic_slmfmt_1k.jsonl``.

Each sample is a ZebraLogic multiple-choice puzzle whose accepted output is a
single numeric choice index in the last complete ``\\boxed{...}`` block. The
dataset contains two to six choices, so valid labels are the strings ``0``
through ``5``. Correct predictions receive 1.0; all wrong, malformed, or
inconsistent samples receive 0.0.

Recommended custom RM path:
    --custom-rm-path recipe.math_logic_puzzle_zebralogic.math_logic_puzzle_zebralogic_reward.math_logic_puzzle_zebralogic_reward
"""

from __future__ import annotations

from recipe.rl_reward.timeout_utils import raise_if_timeout

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# Set MATH_LOGIC_PUZZLE_ZEBRALOGIC_REWARD_LOG_PATH to enable per-sample logging.
REWARD_LOG_PATH: str | None = os.environ.get("MATH_LOGIC_PUZZLE_ZEBRALOGIC_REWARD_LOG_PATH") or None

VALID_ANSWER_INDICES = frozenset(str(index) for index in range(6))
_BOXED_WRAPPER_PATTERN = re.compile(r"\\(?:text|mathrm|mathbf|operatorname)\s*\{(.*)\}", flags=re.DOTALL)


@dataclass(frozen=True)
class MathLogicPuzzleRewardResult:
    """Detailed result for one math logic puzzle sample."""

    reward: float
    expected_answer: str | None
    extracted_answer: str | None
    error_type: str | None = None
    message: str = ""


def _append_reward_log(sample: Any, result: MathLogicPuzzleRewardResult) -> None:
    """Append one JSONL record for a completed reward evaluation."""
    if REWARD_LOG_PATH is None:
        return
    try:
        metadata = getattr(sample, "metadata", None)
        record = {
            "reward": result.reward,
            "label": getattr(sample, "label", None),
            "response": getattr(sample, "response", None),
            "metadata": metadata if isinstance(metadata, dict) else {},
            "expected_answer": result.expected_answer,
            "extracted_answer": result.extracted_answer,
            "error_type": result.error_type,
            "message": result.message,
        }
        log_dir = os.path.dirname(REWARD_LOG_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(REWARD_LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        raise_if_timeout(exc)
        logger.warning("failed to append math logic puzzle reward log: %s", exc)


def _extract_last_boxed_answer(response: str) -> tuple[str | None, str | None]:
    r"""Extract a valid index from the response's final ``\boxed{...}``."""
    command_start = response.rfind("\\boxed")
    if command_start < 0:
        return None, "no_boxed_answer"

    brace_start = command_start + len("\\boxed")
    while brace_start < len(response) and response[brace_start].isspace():
        brace_start += 1
    if brace_start >= len(response) or response[brace_start] != "{":
        return None, "malformed_boxed_answer"

    depth = 0
    content_end: int | None = None
    for index in range(brace_start, len(response)):
        character = response[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                content_end = index
                break
    if content_end is None:
        return None, "malformed_boxed_answer"

    content = response[brace_start + 1 : content_end].strip()
    while content.startswith("$") and content.endswith("$") and len(content) >= 2:
        content = content[1:-1].strip()

    while True:
        wrapper_match = _BOXED_WRAPPER_PATTERN.fullmatch(content)
        if wrapper_match is None:
            break
        content = wrapper_match.group(1).strip()

    if content not in VALID_ANSWER_INDICES:
        return None, "invalid_boxed_answer"
    return content, None


def _normalize_expected_answer(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    answer = value.strip()
    return answer if answer in VALID_ANSWER_INDICES else None


def _score_math_logic_puzzle_sample(sample: Any) -> MathLogicPuzzleRewardResult:
    metadata_value = getattr(sample, "metadata", None)
    if metadata_value is None:
        metadata: dict[str, Any] = {}
    elif isinstance(metadata_value, dict):
        metadata = metadata_value
    else:
        return MathLogicPuzzleRewardResult(
            0.0,
            None,
            None,
            "invalid_metadata",
            "metadata must be a dictionary",
        )

    data_type = metadata.get("data_type")
    if data_type not in (None, "", "math_logic_puzzle"):
        return MathLogicPuzzleRewardResult(0.0, None, None, "unsupported_data_type", str(data_type))

    label_value = getattr(sample, "label", None)
    ground_truth_value = metadata.get("ground_truth")
    supplied_answers = [value for value in (label_value, ground_truth_value) if value not in (None, "")]
    if not supplied_answers:
        return MathLogicPuzzleRewardResult(0.0, None, None, "missing_expected_answer")

    normalized_answers = [_normalize_expected_answer(value) for value in supplied_answers]
    if any(answer is None for answer in normalized_answers):
        return MathLogicPuzzleRewardResult(0.0, None, None, "invalid_expected_answer", repr(supplied_answers))

    unique_answers = set(normalized_answers)
    if len(unique_answers) != 1:
        return MathLogicPuzzleRewardResult(
            0.0,
            None,
            None,
            "conflicting_expected_answer",
            repr(supplied_answers),
        )
    expected_answer = unique_answers.pop()

    response = getattr(sample, "response", None)
    if not isinstance(response, str):
        return MathLogicPuzzleRewardResult(
            0.0,
            expected_answer,
            None,
            "invalid_response",
            "response must be a string",
        )
    if not response:
        return MathLogicPuzzleRewardResult(0.0, expected_answer, None, "empty_response")

    extracted_answer, extraction_error = _extract_last_boxed_answer(response)
    if extraction_error is not None:
        return MathLogicPuzzleRewardResult(0.0, expected_answer, None, extraction_error)

    reward = 1.0 if extracted_answer == expected_answer else 0.0
    error_type = None if reward == 1.0 else "wrong_answer"
    return MathLogicPuzzleRewardResult(reward, expected_answer, extracted_answer, error_type)


def score_math_logic_puzzle_sample(sample: Any) -> MathLogicPuzzleRewardResult:
    """Score one sample without allowing malformed input to stop a rollout."""
    try:
        return _score_math_logic_puzzle_sample(sample)
    except Exception as exc:
        raise_if_timeout(exc)
        logger.warning("math logic puzzle reward failed; returning 0.0: %s", exc, exc_info=True)
        return MathLogicPuzzleRewardResult(0.0, None, None, "reward_exception", str(exc))


def _score_and_log_sample(sample: Any) -> float:
    result = score_math_logic_puzzle_sample(sample)
    _append_reward_log(sample, result)
    return float(result.reward)


async def math_logic_puzzle_zebralogic_reward(
    args: Any,
    sample_or_samples: Sample | list[Sample],
    **kwargs: Any,
) -> float | list[float]:
    """Return binary rewards for MILES single-sample or batch calls."""
    del args, kwargs

    if isinstance(sample_or_samples, list):
        return [_score_and_log_sample(sample) for sample in sample_or_samples]
    return _score_and_log_sample(sample_or_samples)


math_logic_puzzle_reward = math_logic_puzzle_zebralogic_reward
custom_rm = math_logic_puzzle_zebralogic_reward
