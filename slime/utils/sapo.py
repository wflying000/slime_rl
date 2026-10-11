"""Token-level Soft Adaptive Policy Optimization (arXiv:2511.20347, Eqs. 5-8).

The group-normalized advantages come from GRPO; no virtual rewards or
sequence-level importance ratios are used. The PPO loss API is unchanged.
"""

import math
from typing import Any

import torch


def _validate_temperature(name: str, value: float) -> None:
    # Loss arithmetic is at least FP32, including under mixed precision.
    max_float = torch.finfo(torch.float32).max
    if not math.isfinite(value) or value <= 0 or value > max_float or 4.0 / value > max_float:
        raise ValueError(f"{name} must be finite and positive, with tau and 4/tau representable in float32")


def validate_sapo_args(args: Any) -> None:
    """Fail before model loading rather than silently selecting a hybrid loss."""
    policy_loss_type = getattr(args, "policy_loss_type", "clip")
    if policy_loss_type not in ("clip", "sapo"):
        raise ValueError(f"Unknown policy loss type: {policy_loss_type}")
    if policy_loss_type != "sapo":
        return
    _validate_temperature("--sapo-tau-pos", args.sapo_tau_pos)
    _validate_temperature("--sapo-tau-neg", args.sapo_tau_neg)
    # if args.advantage_estimator != "grpo":
    #     raise ValueError("Pure SAPO requires --advantage-estimator grpo (not NGRPO, GSPO, CISPO or PPO)")
    if args.loss_type != "policy_loss" or getattr(args, "train_backend", "megatron") != "megatron":
        raise ValueError("SAPO requires --loss-type policy_loss and the Megatron training backend")
    if args.n_samples_per_prompt < 2:
        raise ValueError("SAPO requires --n-samples-per-prompt >= 2 for group-relative advantages")
    if not args.rewards_normalization or not args.grpo_std_normalization or args.normalize_advantages:
        raise ValueError("SAPO requires group reward/std normalization without --normalize-advantages")
    if args.eps_clip_c is not None:
        raise ValueError("SAPO replaces hard clipping; --eps-clip-c (dual clipping) is not supported")
    if getattr(args, "custom_advantage_function_path", None) is not None:
        raise ValueError("Pure SAPO does not support --custom-advantage-function-path")


@torch.compile(dynamic=True)
def _sapo_loss(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    tau_pos: float,
    tau_neg: float,
    eps_clip: float,
    eps_clip_high: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Preserve FP64 for reference/gradcheck tests; never exponentiate in FP16.
    dtype = torch.float64 if ppo_kl.dtype == torch.float64 or advantages.dtype == torch.float64 else torch.float32
    log_ratio = -ppo_kl.to(dtype)
    adv = advantages.to(dtype)
    tau = torch.where(adv > 0, torch.full_like(adv, tau_pos), torch.full_like(adv, tau_neg))
    # Numerical saturation only, far outside the policy clipping range. As in
    # upstream PR #1864, this avoids exp overflow and inf * 0 in backward.
    ratio = log_ratio.clamp(min=-60.0, max=60.0).exp()
    gate_input = (tau * (ratio - 1.0)).clamp(min=-60.0, max=60.0)
    probability = torch.sigmoid(gate_input)
    soft_ratio = probability * (4.0 / tau)
    loss = -soft_ratio * adv
    # Diagnostics do not multiply the loss again: differentiating soft_ratio
    # already produces the paper's 4*p*(1-p) gradient gate.
    metrics = {
        "sapo_soft_ratio": soft_ratio.detach(),
        "sapo_gate_weight": (4.0 * probability * torch.sigmoid(-gate_input)).detach(),
        "sapo_would_clipfrac": (
            ((ratio < 1.0 - eps_clip) & (adv < 0)) | ((ratio > 1.0 + eps_clip_high) & (adv > 0))
        ).to(dtype),
    }
    return loss, metrics


def compute_sapo_policy_loss(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    tau_pos: float = 1.0,
    tau_neg: float = 1.05,
    *,
    eps_clip: float = 0.2,
    eps_clip_high: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return unreduced -A * (4/tau) * sigmoid(tau * (exp(-KL)-1)).

    KL is old_log_prob - new_log_prob for each token, not a sequence mean.
    Masking and per-response averaging belong to the existing loss reducer.
    Clip thresholds affect only the counterfactual clipping diagnostic.
    """
    _validate_temperature("sapo_tau_pos", tau_pos)
    _validate_temperature("sapo_tau_neg", tau_neg)
    if ppo_kl.shape != advantages.shape:
        raise ValueError("SAPO requires one advantage per token log-ratio with matching shapes")
    return _sapo_loss(ppo_kl, advantages, tau_pos, tau_neg, eps_clip, eps_clip_high)
