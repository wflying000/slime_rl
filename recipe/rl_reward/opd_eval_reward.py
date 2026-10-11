"""Reward dispatcher for OPD training with task-specific eval rewards.

Training samples use the OPD teacher SGLang server to collect token logprobs.
Eval samples marked with ``metadata["opd_eval_reward"]`` use the IMCL task
reward from ``joint_reward.py``.
"""

from __future__ import annotations

import asyncio

from typing import Any

from .isolated_reward import score_with_timeout
from .joint_reward import joint_reward


def _is_eval_sample(sample: Any) -> bool:
    metadata = getattr(sample, "metadata", None)
    return isinstance(metadata, dict) and bool(metadata.get("opd_eval_reward"))


async def _opd_teacher_reward(args: Any, sample: Any, **kwargs) -> Any:
    from slime.rollout.on_policy_distillation import reward_func

    return await reward_func(args, sample, **kwargs)


async def _score_one(args: Any, sample: Any, **kwargs) -> Any:
    if _is_eval_sample(sample):
        return await joint_reward(args, sample, **kwargs)
    return await score_with_timeout(_opd_teacher_reward, args, sample, **kwargs)


async def opd_eval_reward(args: Any, sample_or_samples: Any, **kwargs) -> Any:
    """Protect teacher requests and eval rewards, including heterogeneous batches."""
    if isinstance(sample_or_samples, list):
        return list(await asyncio.gather(*(_score_one(args, s, **kwargs) for s in sample_or_samples)))
    return await _score_one(args, sample_or_samples, **kwargs)


custom_rm = opd_eval_reward
