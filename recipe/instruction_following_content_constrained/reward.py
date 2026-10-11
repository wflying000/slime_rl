"""Content-constrained instruction-following reward for slime.

The reward first enforces machine-checkable constraints, then combines a
content heuristic with a lenient semantic judge. Configure the entry point as:

``--custom-rm-path recipe.instruction_following_content_constrained.reward.custom_rm``
"""

from __future__ import annotations

from recipe.rl_reward.timeout_utils import raise_if_timeout

import json
import logging
import os
from typing import Any

from slime.utils.types import Sample

from .hard_constraints import strict_constraint_hits, strip_chain_of_thought
from .llm_content_reward import llm_judge
from .rule_content_reward import (
    RULE_PASS_HIGH_CONF,
    RuleContentResult,
    compute_rule_content_reward,
)

logger = logging.getLogger(__name__)

REWARD_LOG_PATH = os.getenv("IF_CC_REWARD_LOG_PATH", "")
FAIL_ON_JUDGE_ERROR = os.getenv("IF_CC_FAIL_ON_JUDGE_ERROR", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _fail_on_judge_error(args: Any) -> bool:
    value = getattr(args, "if_bench_fail_on_judge_error", None) if args is not None else None
    if value is None:
        return FAIL_ON_JUDGE_ERROR
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _extract_instruction(sample: Sample, metadata: dict[str, Any]) -> str:
    """Extract the user instruction from a slime sample."""
    prompt = sample.prompt or metadata.get("prompt", "")
    if isinstance(prompt, list):
        user_parts = [
            item.get("content", "")
            for item in prompt
            if item.get("role") == "user"
        ]
        if user_parts:
            return "\n".join(user_parts)
        return "\n".join(item.get("content", "") for item in prompt)
    return str(prompt or "")


def _compute_disagreement_type(
    rule_result: RuleContentResult | None,
    judge_score: float | None,
    reward: float,
) -> str | None:
    if rule_result is None or judge_score is None or rule_result.score is None:
        return None
    if rule_result.score == judge_score:
        return "none"
    if rule_result.score == 1.0 and judge_score == 0.0:
        return f"rule_pass_judge_fail_final_{int(reward)}"
    if rule_result.score == 0.0 and judge_score == 1.0:
        return f"rule_fail_judge_pass_final_{int(reward)}"
    return f"score_mismatch_final_{int(reward)}"


def _append_reward_log(
    *,
    instruction: str,
    response: str,
    raw_response: str,
    instruction_ids: list[str],
    instruction_kwargs: list[dict[str, Any] | None],
    hits: list[bool],
    hard_constraint_score: float,
    reward: float,
    decision_source: str,
    rule_result: RuleContentResult | None = None,
    judge_score: float | None = None,
    judge_response: str | None = None,
    error: str | None = None,
) -> None:
    if not REWARD_LOG_PATH:
        return

    record = {
        "reward": reward,
        "hard_constraint_score": hard_constraint_score,
        "hits": hits,
        "instruction_id_list": instruction_ids,
        "instruction_kwargs": instruction_kwargs,
        "decision_source": decision_source,
        "rule_content_result": None if rule_result is None else rule_result.to_dict(),
        "judge_score": judge_score,
        "judge_response": judge_response,
        "error": error,
        "disagreement_type": _compute_disagreement_type(rule_result, judge_score, reward),
        "instruction": instruction,
        "response": response,
        "raw_response": raw_response,
    }

    try:
        log_dir = os.path.dirname(REWARD_LOG_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(REWARD_LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("Failed to write instruction-following reward log: %s", exc)


async def instruction_following_content_constrained_reward(
    args: Any,
    sample: Sample,
) -> float:
    """Return a binary reward for hard constraints and substantive content."""
    metadata = sample.metadata or {}
    instruction_ids = metadata["instruction_id_list"]
    instruction_kwargs = metadata["instruction_kwargs"]

    response = strip_chain_of_thought(sample.response)
    instruction = _extract_instruction(sample, metadata)
    hits = strict_constraint_hits(response, instruction_ids, instruction_kwargs)

    if not all(hits):
        reward = 0.0
        _append_reward_log(
            instruction=instruction,
            response=response,
            raw_response=sample.response,
            instruction_ids=instruction_ids,
            instruction_kwargs=instruction_kwargs,
            hits=hits,
            hard_constraint_score=0.0,
            reward=reward,
            decision_source="hard_constraint_fail",
        )
        return reward

    rule_result: RuleContentResult | None = None
    judge_score: float | None = None
    judge_response: str | None = None
    error: str | None = None

    try:
        rule_result = compute_rule_content_reward(
            response,
            instruction,
            instruction_ids,
            instruction_kwargs,
        )
        judge_score, judge_response = await llm_judge(
            response,
            instruction,
            metadata,
            args=args,
            rule_result=rule_result,
        )

        if judge_score > 0:
            reward = 1.0
            decision_source = "judge_pass"
        elif rule_result.verdict == RULE_PASS_HIGH_CONF:
            reward = 1.0
            decision_source = "rule_pass_high_conf"
        else:
            reward = 0.0
            decision_source = "judge_and_rule_fail"
    except Exception as exc:
        raise_if_timeout(exc)
        error = str(exc)
        logger.warning("Instruction-following content judge failed: %s", exc)
        reward = 0.0 if _fail_on_judge_error(args) else 1.0
        decision_source = "judge_error_fail" if reward == 0.0 else "judge_error_pass"

    logger.debug(
        "Instruction-following reward: hard=%s/%s reward=%.4f source=%s instructions=%s",
        sum(hits),
        len(hits),
        reward,
        decision_source,
        instruction_ids,
    )

    _append_reward_log(
        instruction=instruction,
        response=response,
        raw_response=sample.response,
        instruction_ids=instruction_ids,
        instruction_kwargs=instruction_kwargs,
        hits=hits,
        hard_constraint_score=1.0,
        reward=reward,
        decision_source=decision_source,
        rule_result=rule_result,
        judge_score=judge_score,
        judge_response=judge_response,
        error=error,
    )
    return reward


custom_rm = instruction_following_content_constrained_reward
