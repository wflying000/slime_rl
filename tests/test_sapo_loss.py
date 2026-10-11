"""SAPO paper equations, actual Megatron loss dispatch and CLI checks on CPU."""

import argparse
import math
from types import SimpleNamespace

import _cp_dist_helpers  # noqa: F401; CPU Megatron stub, as in the CP tests
import pytest
import torch

from recipe.rl_reward.reward_post_process import reward_post_process
from slime.utils.ppo_utils import compute_policy_loss
from slime.utils.sapo import compute_sapo_policy_loss, validate_sapo_args
from slime.utils.types import Sample

NUM_GPUS = 0


def make_args(**overrides):
    values = dict(
        policy_loss_type="sapo",
        advantage_estimator="grpo",
        loss_type="policy_loss",
        train_backend="megatron",
        sapo_tau_pos=1.0,
        sapo_tau_neg=1.05,
        n_samples_per_prompt=4,
        rewards_normalization=True,
        grpo_std_normalization=True,
        normalize_advantages=False,
        eps_clip_c=None,
        custom_advantage_function_path=None,
        eps_clip=0.2,
        eps_clip_high=0.2,
        use_rollout_logprobs=True,
        use_opsm=False,
        get_mismatch_metrics=False,
        use_tis=False,
        entropy_coef=0.0,
        use_kl_loss=False,
        calculate_per_token_loss=False,
        reward_key=None,
        kl_coef=0.0,
        use_opd=False,
    )
    return SimpleNamespace(**(values | overrides))


@pytest.mark.parametrize("temps", [(1.0, 1.05), (0.5, 2.0), (2.0, 0.5)])
def test_paper_loss_and_analytic_gradient(temps):
    # Scalar, independent oracle for Eqs. 5-8, including both advantage signs.
    ratios = [0.01, 0.5, 0.8, 1.0, 1.2, 2.0, 5.0]
    for advantage in [-2.0, 0.0, 0.75]:
        log_ratios = torch.tensor([math.log(r) for r in ratios], dtype=torch.float64, requires_grad=True)
        advantages = torch.full_like(log_ratios, advantage)
        loss, aux = compute_sapo_policy_loss(-log_ratios, advantages, *temps)
        tau = temps[0] if advantage > 0 else temps[1]
        p = [1 / (1 + math.exp(-tau * (r - 1))) for r in ratios]
        expected = torch.tensor([-a * 4 / tau * advantage for a in p], dtype=torch.float64)
        grad = torch.tensor(
            [-4 * a * (1 - a) * r * advantage for a, r in zip(p, ratios, strict=True)], dtype=torch.float64
        )
        torch.testing.assert_close(loss, expected, atol=1e-12, rtol=1e-12)
        loss.sum().backward()
        torch.testing.assert_close(log_ratios.grad, grad, atol=1e-12, rtol=1e-12)
        assert all(not v.requires_grad for v in aux.values())


def test_gradcheck_and_on_policy_gradient_is_temperature_independent():
    x = torch.tensor([-0.4, 0.0, 0.5], dtype=torch.float64, requires_grad=True)
    adv = torch.tensor([1.0, -2.0, 0.0], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda z: compute_sapo_policy_loss(-z, adv, 1.0, 1.05)[0], (x,))
    for pos, neg in [(0.5, 1.5), (1.0, 1.05), (2.0, 3.0)]:
        x = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        loss, aux = compute_sapo_policy_loss(-x, adv, pos, neg)
        loss.sum().backward()
        torch.testing.assert_close(x.grad, -adv)
        torch.testing.assert_close(aux["sapo_gate_weight"], torch.ones_like(adv))


def test_soft_gate_preserves_gradient_beyond_ppo_clip_band():
    x = torch.tensor([math.log(1.5), math.log(0.5)], dtype=torch.float64, requires_grad=True)
    adv = torch.tensor([1.0, -1.0], dtype=torch.float64)
    hard, _ = compute_policy_loss(-x, adv, 0.2, 0.2)
    hard_grad = torch.autograd.grad(hard.sum(), x)[0]
    soft, aux = compute_sapo_policy_loss(-x, adv)
    soft_grad = torch.autograd.grad(soft.sum(), x)[0]
    assert (hard_grad == 0).all()
    assert soft_grad[0] < 0 < soft_grad[1]
    torch.testing.assert_close(aux["sapo_would_clipfrac"], torch.ones_like(adv))


def test_clip_thresholds_only_change_diagnostic_and_tokens_remain_independent():
    x = torch.tensor([0.0, 0.5, 10.0], dtype=torch.float64, requires_grad=True)
    adv = torch.ones_like(x)
    narrow, _ = compute_sapo_policy_loss(-x, adv, eps_clip=0.01, eps_clip_high=0.01)
    wide, _ = compute_sapo_policy_loss(-x, adv, eps_clip=0.9, eps_clip_high=5.0)
    torch.testing.assert_close(narrow, wide)
    narrow.sum().backward()
    assert x.grad[0].item() == pytest.approx(-1.0)
    assert x.grad[1] < 0
    assert x.grad[2] == 0  # saturated outlier does not zero the other tokens


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_extreme_ratios_zero_advantages_and_backward_are_finite(dtype):
    x = torch.tensor([-1000, -100, -12, 0, 12, 100, 1000], dtype=dtype, requires_grad=True)
    adv = torch.tensor([-1, 0, 1, -1, 1, 0, -1], dtype=dtype)
    loss, aux = compute_sapo_policy_loss(-x, adv)
    assert loss.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    assert torch.isfinite(loss).all()
    assert all(torch.isfinite(v).all() for v in aux.values())
    assert (loss[adv == 0] == 0).all()
    loss.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_empty_tokens_are_supported_and_shapes_are_checked():
    x = torch.empty(0, requires_grad=True)
    loss, aux = compute_sapo_policy_loss(x, torch.empty(0))
    loss.sum().backward()
    assert loss.numel() == 0 and x.grad.numel() == 0
    assert all(v.numel() == 0 for v in aux.values())
    with pytest.raises(ValueError, match="matching shapes"):
        compute_sapo_policy_loss(torch.zeros(2), torch.zeros(1))


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf"), -float("inf"), 1e-300, 1e300])
@pytest.mark.parametrize("name", ["tau_pos", "tau_neg"])
def test_invalid_temperatures_rejected(temperature, name):
    with pytest.raises(ValueError, match="finite and positive"):
        compute_sapo_policy_loss(torch.zeros(1), torch.ones(1), **{name: temperature})


@pytest.mark.parametrize(
    "override",
    [
        dict(advantage_estimator="ngrpo"),
        dict(advantage_estimator="gspo"),
        dict(advantage_estimator="cispo"),
        dict(advantage_estimator="ppo"),
        dict(loss_type="custom_loss"),
        dict(train_backend="fsdp"),
        dict(n_samples_per_prompt=1),
        dict(rewards_normalization=False),
        dict(grpo_std_normalization=False),
        dict(normalize_advantages=True),
        dict(eps_clip_c=2.0),
        dict(custom_advantage_function_path="custom.fn"),
    ],
)
def test_pure_sapo_rejects_incompatible_configurations(override):
    with pytest.raises(ValueError):
        validate_sapo_args(make_args(**override))


def test_real_cli_defaults_explicit_values_and_validation(monkeypatch):
    from test_megatron_argument_validation import load_slime_arguments_module, make_slime_validate_args

    module = load_slime_arguments_module(monkeypatch)
    parser = argparse.ArgumentParser()
    module.get_slime_extra_args_provider()(parser)
    default = parser.parse_args(["--rollout-batch-size", "2"])
    assert (default.policy_loss_type, default.sapo_tau_pos, default.sapo_tau_neg) == ("clip", 1.0, 1.05)
    parsed = parser.parse_args(
        [
            "--rollout-batch-size",
            "2",
            "--n-samples-per-prompt",
            "4",
            "--policy-loss-type",
            "sapo",
            "--sapo-tau-pos",
            "0.8",
            "--sapo-tau-neg",
            "1.2",
        ]
    )
    assert (parsed.policy_loss_type, parsed.sapo_tau_pos, parsed.sapo_tau_neg) == ("sapo", 0.8, 1.2)
    args = make_slime_validate_args(**vars(make_args()))
    module.slime_validate_args(args)
    assert not args.use_critic
    with pytest.raises(ValueError, match="not NGRPO"):
        module.slime_validate_args(make_slime_validate_args(**vars(make_args(advantage_estimator="ngrpo"))))


@pytest.mark.parametrize("reward", [0.0, 1.0])
def test_homogeneous_groups_have_zero_advantage_without_virtual_rewards(reward):
    samples = [Sample(reward=reward, group_index=0, response_length=2) for _ in range(4)]
    raw, adv = reward_post_process(make_args(), samples)
    assert raw == [reward] * 4 and adv == [0.0] * 4
    x = torch.zeros(4, requires_grad=True)
    loss, _ = compute_sapo_policy_loss(-x, torch.tensor(adv))
    loss.sum().backward()
    assert (x.grad == 0).all()


@pytest.mark.parametrize("all_masked", [False, True])
@pytest.mark.parametrize("use_rollout_logprobs", [False, True])
def test_real_megatron_loss_masks_metrics_and_optimizer_step(monkeypatch, all_masked, use_rollout_logprobs):
    from slime.backends.megatron_utils import loss as module
    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean

    args = make_args(use_rollout_logprobs=use_rollout_logprobs)
    # Last response represents a timed-out scorer: excluded from normalization
    # and all of its tokens masked, exactly as in the production postprocessor.
    samples = [Sample(reward=r, group_index=0, response_length=n) for r, n in [(1.0, 2), (0.0, 3), (0.0, 2)]]
    samples[-1].metadata = {"reward_timeout": True}
    _, calibrated = reward_post_process(args, samples)
    assert calibrated[-1] == 0 and samples[-1].loss_mask == [0, 0]
    lengths = [2, 3, 2]
    masks = [torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0, 0.0]), torch.zeros(2)]
    if all_masked:
        masks = [torch.zeros_like(m) for m in masks]
    logits = torch.nn.Parameter(torch.tensor([[0.2, -0.1, 0.4]] * 7, dtype=torch.float64))
    targets = torch.tensor([0, 1, 1, 2, 0, 2, 1])
    old = logits.detach().log_softmax(-1).gather(1, targets[:, None]).flatten()
    advantages = [torch.full((n,), a, dtype=torch.float64) for n, a in zip(lengths, calibrated, strict=True)]
    batch = dict(
        advantages=advantages,
        rollout_log_probs=list(old.split(lengths)),
        log_probs=list(old.split(lengths)),
        response_lengths=lengths,
        total_lengths=[n + 1 for n in lengths],
        loss_masks=masks,
        unconcat_tokens=[torch.zeros(n + 1, dtype=torch.long) for n in lengths],
    )

    def cpu_log_probs(logits, **kwargs):
        logp = logits.log_softmax(-1)
        chosen = logp.gather(1, targets[:, None]).flatten()
        entropy = -(logp.exp() * logp).sum(-1)
        return None, dict(log_probs=list(chosen.split(lengths)), entropy=list(entropy.split(lengths)))

    monkeypatch.setattr(module, "get_log_probs_and_entropy", cpu_log_probs)
    monkeypatch.setattr(module, "get_rollout_top_p_logprob_kwargs", lambda *a: {})
    reducer = get_sum_of_sample_mean(batch["total_lengths"], lengths, masks)
    actual, metrics = module.policy_loss_function(args, batch, logits, reducer)
    # At r=1 the exact gradient w.r.t. log pi equals -advantage, for any tau.
    expected = sum(
        -a * (2 / (1.0 if a > 0 else 1.05)) for a, mask in zip(calibrated, masks, strict=True) if mask.sum()
    )
    assert actual.item() == pytest.approx(expected)
    assert metrics["pg_clipfrac"] == 0
    for key in ["sapo_soft_ratio", "sapo_gate_weight", "sapo_would_clipfrac"]:
        assert key in metrics and torch.isfinite(metrics[key]) and not metrics[key].requires_grad
    before = logits.detach().clone()
    optimizer = torch.optim.SGD([logits], lr=0.01)
    actual.backward()
    flat_mask = torch.cat(masks).bool()
    assert (logits.grad[~flat_mask] == 0).all()
    optimizer.step()
    if all_masked:
        assert actual.item() == 0
        torch.testing.assert_close(logits, before)
    else:
        after = logits.detach().log_softmax(-1).gather(1, targets[:, None]).flatten()
        assert (after[:2] > old[:2]).all()
        assert (after[2:4] < old[2:4]).all()
    # Selecting the legacy loss must leave SAPO auxiliary metrics absent.
    args.policy_loss_type = "clip"
    _, legacy_metrics = module.policy_loss_function(args, batch, logits, reducer)
    assert not any(k.startswith("sapo_") for k in legacy_metrics)


@pytest.mark.parametrize("cp_size", [2, 4])
def test_context_parallel_shards_match_unsharded_loss_and_gradient(monkeypatch, cp_size):
    # SAPO is token-local: CP only shards the inputs and sums the existing
    # reducer's outputs. Check the real shard layout, including empty shards.
    from _cp_dist_helpers import cp_chunk_response_tensor
    from megatron.core import mpu
    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean

    total_lengths, lengths = [16, 16], [7, 2]
    masks = [torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]), torch.ones(2)]
    base_log_ratios = [
        torch.linspace(-0.9, 0.8, 7, dtype=torch.float64),
        torch.tensor([0.4, -0.5], dtype=torch.float64),
    ]
    advantages = [torch.ones(7, dtype=torch.float64), -torch.ones(2, dtype=torch.float64)]
    full = [x.clone().requires_grad_() for x in base_log_ratios]
    per_token, _ = compute_sapo_policy_loss(-torch.cat(full), torch.cat(advantages))
    expected = get_sum_of_sample_mean(total_lengths, lengths, masks)(per_token)
    expected_grad = torch.autograd.grad(expected, full)

    sharded = [x.clone().requires_grad_() for x in base_log_ratios]
    actual = torch.tensor(0.0, dtype=torch.float64)
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: cp_size)
    for rank in range(cp_size):
        monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda rank=rank: rank)
        local_x = torch.cat(
            [cp_chunk_response_tensor(x, t, r) for x, t, r in zip(sharded, total_lengths, lengths, strict=True)]
        )
        local_a = torch.cat(
            [cp_chunk_response_tensor(a, t, r) for a, t, r in zip(advantages, total_lengths, lengths, strict=True)]
        )
        per_token, _ = compute_sapo_policy_loss(-local_x, local_a)
        actual = actual + get_sum_of_sample_mean(total_lengths, lengths, masks)(per_token)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, sharded)
    for a, b in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, b)
