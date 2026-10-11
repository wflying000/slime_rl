"""Protected reward dispatcher for mixed instruction, math, code and logic data.

All registered routes use the same killable worker pool. Configure a positive
JOINT_REWARD_SAMPLE_TIMEOUT_S (default 240) and JOINT_REWARD_WORKERS (default 8).
Use recipe.rl_reward.joint_reward.joint_reward as the custom RM path.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .isolated_reward import get_timeout_s, score_with_timeout
from .timeout_utils import mark_invalid_reward


logger = logging.getLogger(__name__)


def _get_metadata(sample: Any) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        raise ValueError("sample.metadata must be a dict")
    return metadata


def _get_data_type(sample: Any) -> str:
    metadata = _get_metadata(sample)
    data_type = metadata.get("data_type")
    if not isinstance(data_type, str) or not data_type.strip():
        raise ValueError("sample.metadata['data_type'] must be a non-empty string")
    return data_type.strip().lower()



async def _score_if(args: Any, sample: Any, **kwargs) -> float:
    from recipe.instruction_following.if_reward import if_reward

    return float(await if_reward(args, sample, **kwargs))


async def _score_if_content_constrained(args: Any, sample: Any, **kwargs) -> float:

    from recipe.instruction_following_content_constrained.reward import instruction_following_content_constrained_reward
    
    return float(await instruction_following_content_constrained_reward(args, sample, **kwargs))


async def _score_klear_math(args: Any, sample: Any, **kwargs) -> float:
    from recipe.klear_math.klear_math_reward import klear_math_reward

    return float(await klear_math_reward(args, sample, **kwargs))


async def _score_klear_code(args: Any, sample: Any, **kwargs) -> float:
    from recipe.klear_code.klear_code_reward import klear_code_reward

    return float(await klear_code_reward(args, sample, **kwargs))


async def _score_guru_rl(args: Any, sample: Any, **kwargs) -> float:
    from recipe.guru_rl.guru_rl_reward import guru_rl_reward

    return float(await guru_rl_reward(args, sample, **kwargs))


async def _score_nemotron_rl_reasoning_gym(args: Any, sample: Any, **kwargs) -> float:
    from recipe.nemotron_rl_reasoning_gym.reasoning_gym_reward import reasoning_gym_reward

    return float(await reasoning_gym_reward(args, sample, **kwargs))


async def _score_math_logic_puzzle(args: Any, sample: Any, **kwargs) -> float:
    from recipe.math_logic_puzzle_zebralogic.math_logic_puzzle_zebralogic_reward import math_logic_puzzle_zebralogic_reward

    return float(await math_logic_puzzle_zebralogic_reward(args, sample, **kwargs))


async def _score_zebralogicbench(args: Any, sample: Any, **kwargs) -> float:
    from recipe.zebralogicbench.zebralogicbench_reward import zebralogicbench_reward

    return float(await zebralogicbench_reward(args, sample, **kwargs))


async def _score_structured_outputs(args: Any, sample: Any, **kwargs) -> float:
    from recipe.nemotron_structured_outputs.structured_outputs_reward import structured_outputs_reward

    return float(await structured_outputs_reward(args, sample, **kwargs))



async def _score_nemotron_if_adversarial(args: Any, sample: Any, **kwargs) -> float:
    from recipe.nemotron_rl_if_adversarial.if_adversarial_reward import if_adversarial_reward

    return float(await if_adversarial_reward(args, sample, **kwargs))


async def _score_nemotron_if_calendar(args: Any, sample: Any, **kwargs) -> float:
    from recipe.nemotron_rl_if_calendar.calendar_reward import calendar_reward

    return float(await calendar_reward(args, sample, **kwargs))


async def _score_nemotron_if_citation_formatting(args: Any, sample: Any, **kwargs) -> float:
    from recipe.nemotron_rl_if_citation_formatting.citation_formatting_reward import citation_formatting_reward

    return float(await citation_formatting_reward(args, sample, **kwargs))


async def _score_nemotron_if_free_form_formatting(args: Any, sample: Any, **kwargs) -> float:
    from recipe.nemotron_rl_if_free_form_formatting.free_form_formatting_reward import free_form_formatting_reward

    return float(await free_form_formatting_reward(args, sample, **kwargs))

async def _score_nemotron_if_multiturnchat(args: Any, sample: Any, **kwargs) -> float:
    from recipe.nemotron_rl_if_multiturnchat.if_multiturnchat_reward import if_multiturnchat_reward

    return float(await if_multiturnchat_reward(args, sample, **kwargs))


async def _score_nemotron_mcqa(args: Any, sample: Any, **kwargs) -> float:
    from recipe.nemotron_mcqa.mcqa_reward import mcqa_reward

    return float(await mcqa_reward(args, sample, **kwargs))

async def _score_nemotron_code(args: Any, sample: Any, **kwargs) -> float:
    from recipe.nemotron_code.code_reward import nemotron_code_reward

    return float(await nemotron_code_reward(args, sample, **kwargs))

async def _score_nemotron_math(args: Any, sample: Any, **kwargs) -> float:
    from recipe.nemotron_math.math_reward import nemotron_math_reward

    return float(await nemotron_math_reward(args, sample, **kwargs))

async def _score_nemotron_if(args: Any, sample: Any, **kwargs) -> float:
    from recipe.nemotron_rl_if.if_reward import if_reward 

    return float(await if_reward(args, sample, **kwargs))

# async def _score_kaoshibao(args: Any, sample: Any, **kwargs) -> float:
#     from recipe.kaoshibao.kaoshibao_reward import kaoshibao_reward

#     return float(await kaoshibao_reward(args, sample, **kwargs))

async def _score_kaoshibao_v3(args: Any, sample: Any, **kwargs) -> float:
    from recipe.kaoshibao_v3.kaoshibao_reward import kaoshibao_reward

    return float(await kaoshibao_reward(args, sample, **kwargs))

async def _score_gpqa(args: Any, sample: Any, **kwargs) -> float:
    from recipe.gpqa.gpqa_reward import gpqa_reward

    return float(await gpqa_reward(args, sample, **kwargs))

async def _score_mmlu(args: Any, sample: Any, **kwargs) -> float:
    from recipe.mmlu.mmlu_reward import mmlu_reward

    return float(await mmlu_reward(args, sample, **kwargs))

async def _score_dr_sci_verifiable(args: Any, sample: Any, **kwargs) -> float:
    from recipe.dr_sci.reward import reward_func

    return float(await reward_func(args, sample, **kwargs))


async def _score_goose_reason_stem(args: Any, sample: Any, **kwargs) -> float:
    from recipe.goose_reason_stem.reward import reward_func

    return float(await reward_func(args, sample, **kwargs))

async def _score_open_reasoning_rlvr_24k_science(args: Any, sample: Any, **kwargs) -> float:
    from recipe.open_reasoning_rlvr_24k.reward import reward_func

    return float(await reward_func(args, sample, **kwargs))



async def _score_ner(args: Any, sample: Any, **kwargs) -> float:
    from recipe.ner.ner_reward import ner_reward

    return float(await ner_reward(args, sample, **kwargs))


_REWARD_SCORERS = {
    'if': _score_if,
    'if_cc': _score_if_content_constrained,
    'guru_rl': _score_guru_rl,
    'nemotron_rl_reasoning_gym': _score_nemotron_rl_reasoning_gym,
    'math_logic_puzzle': _score_math_logic_puzzle,
    'zebralogicbench': _score_zebralogicbench,
    'klear_math': _score_klear_math,
    'klear_code': _score_klear_code,
    'nemotron_structured_outputs': _score_structured_outputs,
    
    
    'nemotron_if_adversarial': _score_nemotron_if_adversarial,
    'nemotron_if_calendar': _score_nemotron_if_calendar,
    'nemotron_if_citation_formatting': _score_nemotron_if_citation_formatting,
    'nemotron_if_free_form_formatting': _score_nemotron_if_free_form_formatting,
    'nemotron_if_multiturnchat': _score_nemotron_if_multiturnchat,
    'nemotron_mcqa': _score_nemotron_mcqa,
    'nemotron_code': _score_nemotron_code,
    'nemotron_math_skywork_no_omni': _score_nemotron_math,
    'nemotron_math_dapo17k': _score_nemotron_math,
    'nemotron_if': _score_nemotron_if,
    'kaoshibao': _score_kaoshibao_v3,
    "gpqa": _score_gpqa,
    "mmlu": _score_mmlu,
    "dr_sci_verifiable": _score_dr_sci_verifiable,
    "goose_reason_stem": _score_goose_reason_stem,
    "open_reasoning_rlvr_24k_science": _score_open_reasoning_rlvr_24k_science,
    
    "ner": _score_ner,
}


def _get_sample_timeout_s(args: Any) -> float:
    return get_timeout_s(args)


def _mark_reward_timeout(sample: Any) -> None:
    mark_invalid_reward(sample, timeout=True, reason="reward timed out")


async def _score_single_sample_with_timeout(args: Any, sample: Any, **kwargs) -> float:
    # Invalid timeout configuration must fail immediately, not silently disable protection.
    get_timeout_s(args)
    try:
        data_type = _get_data_type(sample)
        scorer = _REWARD_SCORERS[data_type]
    except (ValueError, KeyError) as exc:
        mark_invalid_reward(sample, timeout=False, reason=f"invalid reward route: {exc}")
        logger.error("Invalid reward route; excluding sample: %s", exc)
        return 0.0
    return await score_with_timeout(scorer, args, sample, **kwargs)


async def joint_reward(args: Any, sample_or_samples: Any, **kwargs) -> float | list[float]:
    """Score every single sample or batch item in a killable CPU worker."""
    if isinstance(sample_or_samples, list):
        return list(await asyncio.gather(
            *(_score_single_sample_with_timeout(args, sample, **kwargs) for sample in sample_or_samples)
        ))
    return await _score_single_sample_with_timeout(args, sample_or_samples, **kwargs)


custom_rm = joint_reward
