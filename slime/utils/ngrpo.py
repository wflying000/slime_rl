"""Negative-enhanced GRPO reward calibration (arXiv:2509.18851).

Calibrate complete prompt groups on the rollout side, before DP balancing.
Virtual rewards contribute to statistics only, never to the training batch.
The released NGRPO code and Figures 1/3 use sample std (correction=1);
Equation 6 instead uses population std (correction=0). Both are explicit here.
"""

import math
from collections import defaultdict
from collections.abc import Hashable, Sequence
from typing import Any

import torch

from recipe.rl_reward.timeout_utils import is_invalid_reward


def validate_ngrpo_args(args: Any) -> None:
    """Reject settings that would remove or silently change NGRPO's signal."""
    if not math.isfinite(args.ngrpo_max_reward):
        raise ValueError("--ngrpo-max-reward must be finite")
    if args.ngrpo_num_virtual_samples < 1:
        raise ValueError("--ngrpo-num-virtual-samples must be positive")
    if args.normalize_advantages:
        raise ValueError("NGRPO requires --normalize-advantages to be off: whitening can erase all-wrong signals")
    if not args.rewards_normalization or not args.grpo_std_normalization:
        raise ValueError("NGRPO requires reward and group std normalization")
    if args.kl_coef != 0:
        raise ValueError(
            "NGRPO uses outcome rewards; use --use-kl-loss with --kl-loss-coef for optional KL regularization"
        )


@torch.no_grad()
def compute_ngrpo_advantages(
    rewards: torch.Tensor,
    group_ids: Sequence[Hashable],
    *,
    valid_mask: torch.Tensor | None = None,
    max_reward: float = 1.0,
    num_virtual_samples: int = 1,
    std_correction: int = 1,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Return one calibrated advantage per real response, preserving order.

    Invalid rewards (e.g. timeouts) do not enter the statistics and receive
    zero advantage. Singleton valid groups are calibrated with the virtual
    reward too, fixing the released implementation's singleton fallback.
    """
    if rewards.ndim != 1 or len(group_ids) != rewards.numel():
        raise ValueError("NGRPO requires 1D rewards and one group id per reward")
    if not math.isfinite(max_reward) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("NGRPO requires finite max_reward and positive finite epsilon")
    if not isinstance(num_virtual_samples, int) or num_virtual_samples < 1:
        raise ValueError("NGRPO requires a positive integer num_virtual_samples")
    if std_correction not in (0, 1):
        raise ValueError("NGRPO std_correction must be 0 or 1")
    # Keep FP64 oracle precision; promote low precision rewards for statistics.
    scores = rewards.to(dtype=torch.float64 if rewards.dtype == torch.float64 else torch.float32)
    if valid_mask is None:
        valid_mask = torch.ones_like(scores, dtype=torch.bool)
    elif valid_mask.shape != scores.shape:
        raise ValueError("NGRPO valid_mask must have the same shape as rewards")
    valid_mask = valid_mask.to(device=scores.device, dtype=torch.bool)
    valid_scores = scores[valid_mask]
    if not torch.isfinite(valid_scores).all():
        raise ValueError("NGRPO valid rewards must be finite")
    if (valid_scores > max_reward).any():
        raise ValueError("Observed reward exceeds --ngrpo-max-reward; set the task's actual maximum reward")

    groups = defaultdict(list)
    for i, group_id in enumerate(group_ids):
        groups[group_id].append(i)
    advantages = torch.zeros_like(scores)
    for indices in groups.values():
        indices = torch.tensor(indices, dtype=torch.long, device=scores.device)
        indices = indices[valid_mask[indices]]
        if indices.numel() == 0:
            continue
        group_rewards = scores[indices]
        augmented = torch.cat((group_rewards, scores.new_full((num_virtual_samples,), max_reward)))
        advantages[indices] = (group_rewards - augmented.mean()) / (augmented.std(correction=std_correction) + epsilon)
    return advantages


def ngrpo_reward_post_process(args: Any, samples: list[Any]) -> tuple[list[float], list[float]]:
    """Adapt slime Samples to calibration before groups are split across GPUs."""
    raw_rewards = [0.0 if is_invalid_reward(sample) else sample.get_reward_value(args) for sample in samples]
    if not samples:
        return raw_rewards, []
    group_ids = [getattr(sample, "group_index", None) for sample in samples]
    if all(group_id is None for group_id in group_ids):
        # Legacy custom rollouts may omit group_index; only accept complete,
        # contiguous groups instead of accidentally normalizing a whole batch.
        group_size = args.n_samples_per_prompt
        if group_size < 1 or len(samples) % group_size:
            raise ValueError("NGRPO needs group_index for incomplete or variable-sized prompt groups")
        group_ids = [i // group_size for i in range(len(samples))]
    elif any(group_id is None for group_id in group_ids):
        raise ValueError("NGRPO requires group_index on every sample when any group_index is present")
    valid = [not is_invalid_reward(sample) for sample in samples]
    for sample, keep in zip(samples, valid, strict=True):
        if not keep:
            sample.remove_sample = True
            sample.loss_mask = [0] * getattr(sample, "response_length", 0)
    advantages = compute_ngrpo_advantages(
        torch.tensor(raw_rewards, dtype=torch.float32),
        group_ids,
        valid_mask=torch.tensor(valid, dtype=torch.bool),
        max_reward=args.ngrpo_max_reward,
        num_virtual_samples=args.ngrpo_num_virtual_samples,
        std_correction=args.ngrpo_std_correction,
    )
    return raw_rewards, advantages.tolist()
