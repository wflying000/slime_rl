"""Exclude invalid rewards from group statistics and training loss."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from .timeout_utils import is_invalid_reward


def reward_post_process(args: Any, samples: list[Any]) -> tuple[list[float], list[float]]:
    invalid = [is_invalid_reward(sample) for sample in samples]
    for sample, excluded in zip(samples, invalid, strict=True):
        if excluded:
            sample.remove_sample = True
            sample.loss_mask = [0] * getattr(sample, "response_length", 0)
    if args.advantage_estimator == "ngrpo":
        from slime.utils.ngrpo import ngrpo_reward_post_process
        return ngrpo_reward_post_process(args, samples)

    # Do not even read an invalid result: it may be a placeholder for a
    # structured reward or contain NaN from a failed verifier.
    raw_rewards = [0.0 if bad else sample.get_reward_value(args)
                   for sample, bad in zip(samples, invalid, strict=True)]
    if not samples:
        return [], []
    if (args.advantage_estimator not in ("grpo", "gspo", "cispo", "reinforce_plus_plus_baseline")
            or not args.rewards_normalization):
        return raw_rewards, raw_rewards.copy()

    group_ids = [getattr(s, "group_index", None) for s in samples]
    if all(group_id is None for group_id in group_ids):
        size = args.n_samples_per_prompt
        if size < 1 or len(samples) % size:
            raise ValueError("Variable-sized reward groups require sample.group_index")
        group_ids = [i // size for i in range(len(samples))]
    elif any(group_id is None for group_id in group_ids):
        raise ValueError("group_index must be present on every sample")
    groups = defaultdict(list)
    for i, group_id in enumerate(group_ids):
        if not invalid[i]:
            groups[group_id].append(i)

    values = torch.tensor(raw_rewards, dtype=torch.float32)
    advantages = torch.zeros_like(values)
    for indices in groups.values():
        centered = values[indices] - values[indices].mean()
        if args.advantage_estimator in ("grpo", "gspo", "cispo") and args.grpo_std_normalization:
            # A singleton valid group has zero centered reward and variance.
            var = centered.square().sum() / max(1, len(indices) - 1)
            centered = centered / (var.sqrt() + 1e-6)
        advantages[indices] = centered
    return raw_rewards, advantages.tolist()
