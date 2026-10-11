"""NGRPO numerical, clipping, rollout and trainer regression tests on CPU."""

import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import _cp_dist_helpers  # noqa: F401; CPU Megatron stub, same as CP tests
import pytest
import torch

from recipe.rl_reward.reward_post_process import reward_post_process
from recipe.rl_reward.timeout_utils import is_invalid_reward
from slime.utils.ngrpo import compute_ngrpo_advantages, ngrpo_reward_post_process, validate_ngrpo_args
from slime.utils.ppo_utils import compute_policy_loss
from slime.utils.types import Sample

NUM_GPUS = 0
ROOT = Path(__file__).resolve().parents[1]


def make_args(**overrides):
    args = dict(
        advantage_estimator="ngrpo",
        ngrpo_max_reward=1.0,
        ngrpo_num_virtual_samples=1,
        ngrpo_std_correction=1,
        n_samples_per_prompt=4,
        rollout_batch_size=2,
        normalize_advantages=False,
        rewards_normalization=True,
        grpo_std_normalization=True,
        kl_coef=0.0,
        use_rollout_logprobs=True,
        use_opd=False,
        custom_advantage_function_path=None,
        reward_key=None,
    )
    return SimpleNamespace(**(args | overrides))


def samples_for(rewards, ids):
    return [
        Sample(
            reward=r, group_index=g, response_length=3, index=i, rollout_id=i, tokens=[0, 1, 2, 3], loss_mask=[1, 1, 0]
        )
        for i, (r, g) in enumerate(zip(rewards, ids, strict=True))
    ]


def oracle(rewards, max_reward=1.0, virtual=1, correction=1):
    # Independent scalar reference for Equations 5/6 (with explicit correction).
    augmented = list(rewards) + [max_reward] * virtual
    mean = math.fsum(augmented) / len(augmented)
    std = math.sqrt(math.fsum((r - mean) ** 2 for r in augmented) / (len(augmented) - correction))
    return [(r - mean) / (std + 1e-6) for r in rewards]


@pytest.mark.parametrize("size", [1, 2, 3, 8, 16])
@pytest.mark.parametrize("correction", [0, 1])
def test_all_binary_group_compositions_match_independent_oracle(size, correction):
    for correct in range(size + 1):
        rewards = [0.0] * (size - correct) + [1.0] * correct
        actual = compute_ngrpo_advantages(
            torch.tensor(rewards, dtype=torch.float64), [9] * size, std_correction=correction
        )
        torch.testing.assert_close(actual, torch.tensor(oracle(rewards, correction=correction), dtype=torch.float64))
        assert not actual.requires_grad
        if correct == 0:
            assert (actual < 0).all()
        if correct == size:
            assert (actual == 0).all()


def test_paper_figures_and_continuous_rewards():
    actual = compute_ngrpo_advantages(torch.tensor([0.0, 0.0, 0.0]), [0] * 3)
    torch.testing.assert_close(actual, torch.full((3,), -0.5), atol=2e-6, rtol=0)
    low = compute_ngrpo_advantages(torch.tensor([1.0] + [0.0] * 7), [0] * 8)
    assert low[0].item() == pytest.approx(1.76383, abs=1e-5)
    high = compute_ngrpo_advantages(torch.tensor([1.0] * 7 + [0.0]), [0] * 8)
    assert high[-1].item() == pytest.approx(-2.66666, abs=1e-5)
    rewards = torch.tensor([-2.0, 0.2, 0.5, 2.0], dtype=torch.float64, requires_grad=True)
    actual = compute_ngrpo_advantages(rewards, [0] * 4, max_reward=2.0, num_virtual_samples=3)
    torch.testing.assert_close(actual, torch.tensor(oracle(rewards.tolist(), 2.0, 3), dtype=torch.float64))
    assert not actual.requires_grad


def test_shuffled_variable_groups_and_timeout_exclusion():
    samples = samples_for([0.0, 1.0, 0.0, 0.0, float("nan"), 0.0], [5, 8, 5, 8, 5, 11])
    samples[4].metadata = {"reward_timeout": True}
    samples[5].remove_sample = True
    _, actual = ngrpo_reward_post_process(make_args(), samples)
    expected = [*oracle([0.0, 0.0]), *oracle([1.0, 0.0])]
    assert actual == pytest.approx([expected[0], expected[2], expected[1], expected[3], 0.0, 0.0])
    _, custom = reward_post_process(make_args(), samples)
    assert custom == actual


def test_single_valid_sample_and_empty_groups_are_finite():
    samples = samples_for([0.0, 0.0, 0.0, 0.0], [0] * 4)
    for sample in samples[1:]:
        sample.metadata = {"reward_timeout": True}
    _, actual = ngrpo_reward_post_process(make_args(), samples)
    assert actual == pytest.approx(oracle([0.0]) + [0.0] * 3)
    samples[0].metadata = {"reward_timeout": True}
    assert ngrpo_reward_post_process(make_args(), samples)[1] == [0.0] * 4
    assert ngrpo_reward_post_process(make_args(), []) == ([], [])


def test_group_identity_fallback_and_missing_identity_rejected():
    samples = samples_for([0.0] * 4 + [1.0] * 4, [None] * 8)
    actual = ngrpo_reward_post_process(make_args(), samples)[1]
    assert actual == pytest.approx(oracle([0.0] * 4) + [0.0] * 4)
    with pytest.raises(ValueError, match="group_index"):
        ngrpo_reward_post_process(make_args(), samples[:-1])
    samples[0].group_index = 0
    with pytest.raises(ValueError, match="every sample"):
        ngrpo_reward_post_process(make_args(), samples)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(max_reward=float("nan")),
        dict(num_virtual_samples=0),
        dict(num_virtual_samples=1.5),
        dict(std_correction=2),
        dict(epsilon=0),
    ],
)
def test_invalid_calibration_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        compute_ngrpo_advantages(torch.tensor([0.0]), [0], **kwargs)


@pytest.mark.parametrize("rewards", [[float("nan")], [float("inf")], [1.1]])
def test_invalid_valid_rewards_rejected(rewards):
    with pytest.raises(ValueError):
        compute_ngrpo_advantages(torch.tensor(rewards), [0])


@pytest.mark.parametrize(
    "settings",
    [
        dict(normalize_advantages=True),
        dict(rewards_normalization=False),
        dict(grpo_std_normalization=False),
        dict(ngrpo_num_virtual_samples=0),
        dict(ngrpo_max_reward=float("inf")),
        dict(kl_coef=0.1),
    ],
)
def test_incompatible_arguments_rejected(settings):
    with pytest.raises(ValueError):
        validate_ngrpo_args(make_args(**settings))


def test_asymmetric_clipping_values_and_gradients():
    ratios = torch.tensor([0.7, 0.9, 1.1, 1.3, 0.7, 0.9, 1.1, 1.3])
    log_ratios = ratios.log().requires_grad_()
    advantages = torch.tensor([1.0] * 4 + [-1.0] * 4)
    loss, clipped = compute_policy_loss(-log_ratios, advantages, 0.16, 0.24)
    torch.testing.assert_close(loss, torch.tensor([-0.7, -0.9, -1.1, -1.24, 0.84, 0.9, 1.1, 1.3]))
    torch.testing.assert_close(clipped, torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]))
    loss.sum().backward()
    torch.testing.assert_close(log_ratios.grad, torch.tensor([-0.7, -0.9, -1.1, 0.0, 0.0, 0.9, 1.1, 1.3]))


def test_all_wrong_group_has_nonzero_gradient_and_all_correct_has_zero():
    for reward, expected_nonzero in [(0.0, True), (1.0, False)]:
        advantage = compute_ngrpo_advantages(torch.full((4,), reward), [0] * 4)
        log_probs = torch.zeros(4, requires_grad=True)
        losses, _ = compute_policy_loss(-log_probs, advantage, 0.16, 0.24)
        losses.mean().backward()
        assert bool((log_probs.grad > 0).all()) == expected_nonzero
        if not expected_nonzero:
            assert (log_probs.grad == 0).all()


def rollout_post_process_method():
    # Execute the actual actor method without initializing Ray/SGLang servers.
    module = ast.parse((ROOT / "slime/ray/rollout.py").read_text())
    actor = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "RolloutManager")
    method = next(n for n in actor.body if isinstance(n, ast.FunctionDef) and n.name == "_post_process_rewards")
    scope = dict(
        torch=torch,
        Sample=Sample,
        ngrpo_reward_post_process=ngrpo_reward_post_process,
        is_invalid_reward=is_invalid_reward,
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<RolloutManager method>", "exec"), scope)
    return scope["_post_process_rewards"]


@pytest.mark.parametrize("custom", [False, True])
def test_actual_rollout_dispatch_preserves_ngrpo_calibration(custom):
    samples = samples_for([0.0] * 4 + [1.0] * 4, [0] * 4 + [1] * 4)
    actor = SimpleNamespace(args=make_args(), custom_reward_post_process_func=reward_post_process if custom else None)
    _, actual = rollout_post_process_method()(actor, samples)
    assert actual == pytest.approx(oracle([0.0] * 4) + [0.0] * 4)


@pytest.mark.parametrize(
    "estimator", ["grpo", "gspo", "cispo", "ppo", "reinforce_plus_plus", "reinforce_plus_plus_baseline"]
)
@pytest.mark.parametrize("custom", [False, True])
def test_existing_reward_estimators_unchanged(estimator, custom):
    rewards = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0]
    samples = samples_for(rewards, [0] * 4 + [1] * 4)
    args = make_args(advantage_estimator=estimator)
    actor = SimpleNamespace(args=args, custom_reward_post_process_func=reward_post_process if custom else None)
    raw, actual = rollout_post_process_method()(actor, samples)
    assert raw == rewards
    centered = torch.tensor(rewards).reshape(2, 4)
    centered = centered - centered.mean(-1, keepdim=True)
    if estimator in ["grpo", "gspo", "cispo"]:
        expected = centered / (centered.std(-1, keepdim=True) + 1e-6)
    elif estimator == "reinforce_plus_plus_baseline":
        expected = centered
    else:
        expected = torch.tensor(rewards)
    assert actual == pytest.approx(expected.flatten().tolist())


def test_existing_grpo_timeout_behavior_unchanged():
    samples = samples_for([0.0, 1.0, 0.0, 0.0], [0] * 4)
    samples[-1].metadata = {"reward_timeout": True}
    raw, actual = reward_post_process(make_args(advantage_estimator="grpo"), samples)
    rewards = torch.tensor(raw[:3])
    expected = ((rewards - rewards.mean()) / (rewards.std() + 1e-6)).tolist() + [0.0]
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("estimator", ["ngrpo", "grpo", "gspo", "cispo"])
def test_actual_trainer_broadcasts_precalibrated_rewards_after_dp_split(monkeypatch, estimator):
    from slime.backends.megatron_utils import loss as loss_module

    monkeypatch.setattr(loss_module.mpu, "is_pipeline_last_stage", lambda: True, raising=False)
    samples = samples_for([0.0, 1.0, 0.0, 0.0], [0] * 4)
    calibrated = ngrpo_reward_post_process(make_args(), samples)[1]
    # Simulate balancing which moves non-contiguous members to a DP rank.
    data = dict(
        rewards=[calibrated[2], calibrated[0]],
        rollout_log_probs=[torch.zeros(3), torch.zeros(0)],
        response_lengths=[3, 0],
        total_lengths=[4, 10],
        loss_masks=[torch.tensor([1.0, 1.0, 0.0]), torch.zeros(0)],
    )
    original = copy.deepcopy(data)
    loss_module.compute_advantages_and_returns(make_args(advantage_estimator=estimator), data)
    assert len(data["advantages"]) == 2
    for i in range(2):
        torch.testing.assert_close(
            data["advantages"][i], torch.full_like(original["rollout_log_probs"][i], original["rewards"][i])
        )
        torch.testing.assert_close(data["returns"][i], data["advantages"][i])
    assert data["advantages"][0][0] < 0


@pytest.mark.parametrize("estimator", ["ngrpo", "grpo", "gspo", "cispo", "ppo"])
@pytest.mark.parametrize("explicit_clip", [None, 0.12])
def test_real_argument_validation_clip_defaults_and_singleton(monkeypatch, estimator, explicit_clip):
    from test_megatron_argument_validation import load_slime_arguments_module, make_slime_validate_args

    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        advantage_estimator=estimator,
        eps_clip=explicit_clip,
        ngrpo_max_reward=1.0,
        ngrpo_num_virtual_samples=1,
        ngrpo_std_correction=1,
        rewards_normalization=True,
        n_samples_per_prompt=1,
    )
    module.slime_validate_args(args)
    expected_low = explicit_clip if explicit_clip is not None else (0.16 if estimator == "ngrpo" else 0.2)
    assert args.eps_clip == expected_low
    assert args.eps_clip_high == (0.24 if estimator == "ngrpo" else expected_low)
    assert args.use_critic == (estimator == "ppo")
    assert args.grpo_std_normalization == (estimator == "ngrpo")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
