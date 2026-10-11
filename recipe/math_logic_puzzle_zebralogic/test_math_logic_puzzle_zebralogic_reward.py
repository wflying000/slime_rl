"""Tests for the math logic puzzle ZebraLogic reward function."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from recipe.math_logic_puzzle_zebralogic.math_logic_puzzle_zebralogic_reward import (
    custom_rm,
    math_logic_puzzle_reward,
    math_logic_puzzle_zebralogic_reward,
    score_math_logic_puzzle_sample,
)

DATASET_PATH = Path("/Users/a58/RL/data/math_logic_puzzle_zebralogic_slmfmt_1k.jsonl")


def _sample(response: Any, label: Any = "2", metadata: Any = None) -> SimpleNamespace:
    if metadata is None:
        metadata = {"data_type": "math_logic_puzzle", "ground_truth": label}
    return SimpleNamespace(response=response, label=label, metadata=metadata)


def _read_records() -> list[dict[str, Any]]:
    if not DATASET_PATH.is_file():
        pytest.skip(f"local ZebraLogic dataset is unavailable: {DATASET_PATH}")
    with DATASET_PATH.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def _choice_count(record: dict[str, Any]) -> int:
    prompt = record["prompt"][-1]["content"]
    choices_section = prompt.split("\nChoices:\n", 1)[1].split("\n\nWhen answering", 1)[0]
    return len(re.findall(r"(?m)^\d+\.\s+", choices_section))


def _reward(sample_or_samples: Any, **kwargs: Any) -> float | list[float]:
    return asyncio.run(math_logic_puzzle_zebralogic_reward(None, sample_or_samples, **kwargs))


def test_real_dataset_sample_correct_and_wrong() -> None:
    record = _read_records()[0]
    label = record["label"]
    sample = SimpleNamespace(
        prompt=record["prompt"],
        response=fr"Reasoning about houses and choices. Final: \boxed{{{label}}}",
        label=label,
        metadata=record["metadata"],
    )
    assert _reward(sample) == 1.0

    wrong_answer = str((int(label) + 1) % _choice_count(record))
    sample.response = fr"Reasoning about houses and choices. Final: \boxed{{{wrong_answer}}}"
    assert _reward(sample) == 0.0


@pytest.mark.parametrize(
    "response",
    (
        r"\boxed{2}",
        r"reasoning 1, 3, 4; final \boxed{ 2 } trailing 5",
        r"\boxed{\text{2}}",
        r"\boxed{\mathrm{2}}",
        r"\boxed{$2$}",
        r"draft \boxed{1}; corrected \boxed{2}",
    ),
)
def test_supported_correct_boxed_formats(response: str) -> None:
    assert _reward(_sample(response)) == 1.0


def test_last_complete_boxed_answer_wins() -> None:
    result = score_math_logic_puzzle_sample(_sample(r"draft \boxed{2}; final \boxed{1}"))
    assert result.extracted_answer == "1"
    assert result.reward == 0.0
    assert result.error_type == "wrong_answer"


@pytest.mark.parametrize("label", tuple(str(index) for index in range(6)))
def test_all_dataset_answer_indices(label: str) -> None:
    assert _reward(_sample(fr"\boxed{{{label}}}", label=label)) == 1.0
    wrong_answer = str((int(label) + 1) % 6)
    assert _reward(_sample(fr"\boxed{{{wrong_answer}}}", label=label)) == 0.0


@pytest.mark.parametrize(
    ("response", "error_type"),
    (
        ("2", "no_boxed_answer"),
        (r"The correct choice is 2.", "no_boxed_answer"),
        (r"\boxed 2", "malformed_boxed_answer"),
        (r"\boxed{2", "malformed_boxed_answer"),
        (r"\boxed{}", "invalid_boxed_answer"),
        (r"\boxed{6}", "invalid_boxed_answer"),
        (r"\boxed{02}", "invalid_boxed_answer"),
        (r"\boxed{2.0}", "invalid_boxed_answer"),
        (r"\boxed{option 2}", "invalid_boxed_answer"),
    ),
)
def test_missing_or_malformed_boxed_answer_returns_zero(response: str, error_type: str) -> None:
    result = score_math_logic_puzzle_sample(_sample(response))
    assert result.reward == 0.0
    assert result.extracted_answer is None
    assert result.error_type == error_type
    assert _reward(_sample(response)) == 0.0


def test_empty_response_returns_zero() -> None:
    result = score_math_logic_puzzle_sample(_sample(""))
    assert result.reward == 0.0
    assert result.error_type == "empty_response"
    assert _reward(_sample("")) == 0.0


def test_missing_expected_answer_returns_zero() -> None:
    sample = _sample(r"\boxed{2}", label=None, metadata={"data_type": "math_logic_puzzle"})
    result = score_math_logic_puzzle_sample(sample)
    assert result.error_type == "missing_expected_answer"
    assert _reward(sample) == 0.0


@pytest.mark.parametrize("label", ("6", "-1", "02", "2.0", 2, {"answer": "2"}))
def test_invalid_expected_answer_returns_zero(label: Any) -> None:
    sample = _sample(
        r"\boxed{2}",
        label=label,
        metadata={"data_type": "math_logic_puzzle", "ground_truth": label},
    )
    result = score_math_logic_puzzle_sample(sample)
    assert result.error_type == "invalid_expected_answer"
    assert _reward(sample) == 0.0


def test_conflicting_label_and_metadata_returns_zero() -> None:
    sample = _sample(
        r"\boxed{2}",
        label="2",
        metadata={"data_type": "math_logic_puzzle", "ground_truth": "1"},
    )
    result = score_math_logic_puzzle_sample(sample)
    assert result.error_type == "conflicting_expected_answer"
    assert _reward(sample) == 0.0


def test_unsupported_data_type_returns_zero() -> None:
    sample = _sample(r"\boxed{2}", metadata={"data_type": "mmlu", "ground_truth": "2"})
    result = score_math_logic_puzzle_sample(sample)
    assert result.error_type == "unsupported_data_type"
    assert _reward(sample) == 0.0


def test_invalid_metadata_and_response_types_return_zero() -> None:
    invalid_metadata = SimpleNamespace(response=r"\boxed{2}", label="2", metadata=[])
    invalid_response = _sample(None)
    assert score_math_logic_puzzle_sample(invalid_metadata).error_type == "invalid_metadata"
    assert score_math_logic_puzzle_sample(invalid_response).error_type == "invalid_response"
    assert _reward(invalid_metadata) == 0.0
    assert _reward(invalid_response) == 0.0


def test_unexpected_sample_exception_is_contained() -> None:
    class BrokenSample:
        @property
        def metadata(self) -> dict[str, str]:
            raise RuntimeError("broken metadata")

    result = score_math_logic_puzzle_sample(BrokenSample())
    assert result.reward == 0.0
    assert result.error_type == "reward_exception"
    assert "broken metadata" in result.message
    assert _reward(BrokenSample()) == 0.0


def test_batch_mode_is_ordered_and_isolates_bad_samples() -> None:
    rewards = _reward(
        [
            _sample(r"\boxed{2}"),
            _sample(r"\boxed{1}"),
            _sample(None),
            _sample(r"\boxed{5}", label="5"),
        ],
        ignored_keyword=True,
    )
    assert rewards == [1.0, 0.0, 0.0, 1.0]
    assert math_logic_puzzle_reward is math_logic_puzzle_zebralogic_reward
    assert custom_rm is math_logic_puzzle_zebralogic_reward
