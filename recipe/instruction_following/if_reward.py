# coding: utf-8
"""
requirement:
pip install nltk emoji syllapy

  nltk 数据下载（4 个）：
  python -c "
  import nltk
  nltk.download('punkt')
  nltk.download('punkt_tab')
  nltk.download('stopwords')
  nltk.download('averaged_perceptron_tagger_eng')
  "

IFBench reward function for Slime.

Evaluates whether the model's response follows IFBench instruction-following constraints.
Returns 1.0 if ALL instructions are satisfied, 0.0 otherwise.

Usage:
    --custom-rm-path examples.if_bench.if_reward.if_reward
"""

import logging
import os
import sys
import json
from typing import Any

from slime.utils.types import Sample
from recipe.rl_reward.timeout_utils import raise_if_timeout

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

import instructions_registry

logger = logging.getLogger(__name__)

REWARD_LOG_PATH = os.environ.get("IF_REWARD_LOG_PATH") or None


def _append_reward_log(response: str, instruction_id_list: list[str], hits: list[bool], reward: float, raw_response: str):
    """Append a reward record to the JSONL log file."""
    if REWARD_LOG_PATH is None:
        return

    record = {
        "reward": reward,
        "hits": hits,
        "response": response,
        "raw_response": raw_response,
        "instruction_id_list": instruction_id_list,
    }

    try:
        os.makedirs(os.path.dirname(REWARD_LOG_PATH), exist_ok=True)
        with open(REWARD_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        raise_if_timeout(e)
        logger.warning(f"Failed to write reward log: {e}")

def _strip_chain_of_thought(text: str) -> str:
    """Remove <think>...</think> block and trailing special tokens."""
    if not text:
        return ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    text = text.replace("<|im_end|>", "")
    return text


def _filter_null_kwargs(kwargs: dict) -> dict:
    """Filter out null values from IFBench kwargs dicts.

    Also converts float values that are whole numbers to int,
    since JSON deserializes all numbers as float but instruction
    checkers use them as list indices / slice bounds.
    """
    result = {}
    for k, v in kwargs.items():
        if v is None:
            continue
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        result[k] = v
    return result


def _strict_hits(
    solution_str: str,
    instruction_id_list: list[str],
    kwargs_list: list[dict[str, Any]],
) -> list[bool]:
    """Return per-instruction hit list [True/False, ...]."""
    assert len(instruction_id_list) == len(kwargs_list), (
        f"instruction count ({len(instruction_id_list)}) != kwargs count ({len(kwargs_list)})"
    )

    hits: list[bool] = []
    for idx, inst_id in enumerate(instruction_id_list):
        inst_cls = instructions_registry.INSTRUCTION_DICT[inst_id]
        inst = inst_cls(inst_id)
        filtered_kwargs = _filter_null_kwargs(kwargs_list[idx] or {})
        inst.build_description(**filtered_kwargs)
        ok = bool(solution_str.strip()) and bool(inst.check_following(solution_str))
        hits.append(ok)
    return hits


async def if_reward(args, sample: Sample) -> float:
    """
    Slime-compatible reward function for IFBench instruction following.

    Reads instruction_id_list and instruction_kwargs from sample.metadata.
    Filters null values from kwargs (IFBench data includes all possible keys with null for unused ones).
    Returns 1.0 if ALL instructions are satisfied, 0.0 otherwise.

    Args:
        args: Training args (unused).
        sample: Sample with response and metadata containing:
            - "instruction_id_list": list[str]
            - "instruction_kwargs": list[dict]

    Returns:
        float: 1.0 if all instructions pass, 0.0 otherwise
    """
    metadata = sample.metadata
    instruction_id_list = metadata["instruction_id_list"]
    kwargs_list = metadata["instruction_kwargs"]

    response = _strip_chain_of_thought(sample.response)
    try:
        hits = _strict_hits(response, instruction_id_list, kwargs_list)
    except Exception as exc:
        raise_if_timeout(exc)
        raise

    reward = 1.0 if all(hits) else 0.0

    logger.debug(
        f"IFBench reward: {sum(hits)}/{len(hits)} = {reward:.4f}, "
        f"instructions={instruction_id_list}"
    )

    _append_reward_log(response, instruction_id_list, hits, reward, sample.response)

    return reward
