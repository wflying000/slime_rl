"""Slime ``custom_rm`` for math: pure correctness, no format requirement.

Returns ``1.0`` iff the answer extracted from ``sample.response`` matches the
ground truth in ``sample.label`` (or ``sample.metadata['ground_truth']`` /
``sample.metadata['label']``); ``0.0`` otherwise. There is no format penalty
and no requirement for ``<think>`` / ``</think>`` markers or a ``\\boxed{}``
wrapper.

Answer extraction order:
1. The contents of the last ``\\boxed{...}`` in the response.
2. The capture group of the last ``Answer: ...`` line.
3. The full response (post-``</think>`` if present), stripped.

Comparison uses ``math_verify`` when available (semantic equality), and falls
back to a normalized string match otherwise.

Usage:
    --custom-rm-path examples.klear_math.klear_math_reward.klear_math_reward
"""

from __future__ import annotations

from recipe.rl_reward.timeout_utils import raise_if_timeout

import json
import logging
import os
import re
from typing import Any

from slime.utils.types import Sample

from . import math_dapo

logger = logging.getLogger(__name__)

# Set MATH_REWARD_LOG_PATH to enable per-sample reward logging.
REWARD_LOG_PATH: str | None = os.environ.get("MATH_REWARD_LOG_PATH") or None

# Eagerly probe math_verify availability so mis-installation is visible.
_MATH_VERIFY_AVAILABLE: bool | None = None


def _check_math_verify() -> bool:
    global _MATH_VERIFY_AVAILABLE
    if _MATH_VERIFY_AVAILABLE is not None:
        return _MATH_VERIFY_AVAILABLE
    try:
        from math_verify import parse  # noqa: F401
        from math_verify import verify as m_verify  # noqa: F401

        _MATH_VERIFY_AVAILABLE = True
        logger.info("math_verify is available and loaded successfully")
    except ImportError as exc:
        _MATH_VERIFY_AVAILABLE = False
        logger.warning(
            "math_verify is NOT available (ImportError: %s). "
            "Reward will fall back to normalized string match, which is less "
            "accurate for LaTeX expressions. Install with: pip install math-verify",
            exc,
        )
    return _MATH_VERIFY_AVAILABLE

_ANSWER_LINE_RE = re.compile(r"(?i)Answer\s*[:：]\s*([^\n]+)")
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+?\|>")


def _strip_special_tokens(text: str) -> str:
    return _SPECIAL_TOKEN_RE.sub("", text).strip()


def _resolve_label(sample: Sample) -> str:
    if isinstance(sample.label, (str, int, float)) and str(sample.label) != "":
        return str(sample.label)

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    for key in ("ground_truth", "label", "answer"):
        v = metadata.get(key)
        if isinstance(v, (str, int, float)) and str(v) != "":
            return str(v)

    raise ValueError(
        "math sample must provide a non-empty ground truth via sample.label, "
        "metadata['ground_truth'], metadata['label'], or metadata['answer']"
    )


def _extract_predicted_answer(response: str) -> str:
    if not response:
        return ""

    text = _strip_special_tokens(response)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    text = text.strip()

    boxed = math_dapo.last_boxed_only_string(text)
    if boxed is not None:
        try:
            return _strip_special_tokens(math_dapo.remove_boxed(boxed))
        except AssertionError:
            pass

    matches = _ANSWER_LINE_RE.findall(text)
    if matches:
        return _strip_special_tokens(matches[-1])

    return text


def _is_correct(prediction: str, ground_truth: str) -> bool:
    prediction = _strip_special_tokens(prediction)
    ground_truth = _strip_special_tokens(ground_truth)
    if not prediction:
        return False

    norm_pred = math_dapo.normalize_final_answer(prediction)
    norm_gt = math_dapo.normalize_final_answer(ground_truth)
    if norm_pred and norm_pred == norm_gt:
        return True

    if not _check_math_verify():
        return False

    from math_verify import parse
    from math_verify import verify as m_verify

    try:
        gt_parsed = parse(f"\\boxed{{{ground_truth}}}", parsing_timeout=5, raise_on_error=True)
        pred_parsed = parse(f"\\boxed{{{prediction}}}", parsing_timeout=5, raise_on_error=True)
    except Exception as _reward_exc:
        raise_if_timeout(_reward_exc)
        return False

    try:
        return bool(m_verify(gt_parsed, pred_parsed, timeout_seconds=5, raise_on_error=True))
    except Exception as _reward_exc:
        raise_if_timeout(_reward_exc)
        return False


def _append_reward_log(
    response: str,
    label: str,
    prediction: str,
    reward: float,
    metadata: dict[str, Any] | None,
) -> None:
    if REWARD_LOG_PATH is None:
        return
    record = {
        "reward": reward,
        "label": label,
        "prediction": prediction,
        "response": response,
        "metadata": metadata or {},
    }
    try:
        os.makedirs(os.path.dirname(REWARD_LOG_PATH), exist_ok=True)
        with open(REWARD_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning(f"failed to append klear_math reward log: {exc}")


async def klear_math_reward(args, sample: Sample, **kwargs) -> float:
    del args, kwargs
    label = _resolve_label(sample)
    prediction = _extract_predicted_answer(sample.response or "")

    try:
        reward = 1.0 if _is_correct(prediction, label) else 0.0
    except Exception as exc:
        raise_if_timeout(exc)
        logger.warning(f"klear_math_reward crashed: {exc}; returning 0.0")
        reward = 0.0

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    _append_reward_log(sample.response or "", label, prediction, reward, metadata)
    return reward


custom_rm = klear_math_reward
