"""Tests for zebralogicbench_reward."""

import json
import sys
from unittest.mock import MagicMock

import pytest

# Mock torch before importing slime modules that depend on it
if "torch" not in sys.modules:
    sys.modules["torch"] = MagicMock()

from slime.utils.types import Sample


def _make_sample(response: str, label: str = "", mode: str = "grid", **metadata_kwargs) -> Sample:
    metadata = {"data_type": "zebralogicbench", "ground_truth": label, "mode": mode}
    metadata.update(metadata_kwargs)
    return Sample(response=response, label=label, metadata=metadata)


# ---------------------------------------------------------------------------
# Grid-mode tests
# ---------------------------------------------------------------------------


class TestGridScoring:
    @pytest.mark.asyncio
    async def test_perfect_solution(self):
        from examples.zebralogicbench.zebralogicbench_reward import zebralogicbench_reward

        ground_truth = {
            "header": ["House", "Name", "Drink"],
            "rows": [
                ["1", "Arnold", "tea"],
                ["2", "Peter", "water"],
                ["3", "Eric", "milk"],
            ],
        }
        label = json.dumps(ground_truth)
        response = json.dumps({
            "reasoning": "...",
            "solution": {
                "House 1": {"Name": "Arnold", "Drink": "tea"},
                "House 2": {"Name": "Peter", "Drink": "water"},
                "House 3": {"Name": "Eric", "Drink": "milk"},
            },
        })
        sample = _make_sample(response, label, mode="grid")
        reward = await zebralogicbench_reward(None, sample)
        assert reward == 1.0

    @pytest.mark.asyncio
    async def test_partial_solution(self):
        from examples.zebralogicbench.zebralogicbench_reward import zebralogicbench_reward

        ground_truth = {
            "header": ["House", "Name", "Drink"],
            "rows": [
                ["1", "Arnold", "tea"],
                ["2", "Peter", "water"],
            ],
        }
        label = json.dumps(ground_truth)
        # Only Name in House 1 is correct
        response = json.dumps({
            "reasoning": "...",
            "solution": {
                "House 1": {"Name": "Arnold", "Drink": "coffee"},
                "House 2": {"Name": "Bob", "Drink": "water"},
            },
        })
        sample = _make_sample(response, label, mode="grid")
        reward = await zebralogicbench_reward(None, sample)
        # 2 correct cells (Arnold + water) out of 4 total = 0.5
        assert reward == 0.5

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        from examples.zebralogicbench.zebralogicbench_reward import zebralogicbench_reward

        ground_truth = {
            "header": ["House", "Name"],
            "rows": [["1", "Arnold"]],
        }
        label = json.dumps(ground_truth)
        response = json.dumps({
            "reasoning": "...",
            "solution": {"House 1": {"Name": "arnold"}},
        })
        sample = _make_sample(response, label, mode="grid")
        reward = await zebralogicbench_reward(None, sample)
        assert reward == 1.0

    @pytest.mark.asyncio
    async def test_no_json_output(self):
        from examples.zebralogicbench.zebralogicbench_reward import zebralogicbench_reward

        ground_truth = {
            "header": ["House", "Name"],
            "rows": [["1", "Arnold"]],
        }
        label = json.dumps(ground_truth)
        sample = _make_sample("I think the answer is Arnold.", label, mode="grid")
        reward = await zebralogicbench_reward(None, sample)
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_empty_response(self):
        from examples.zebralogicbench.zebralogicbench_reward import zebralogicbench_reward

        ground_truth = {"header": ["House", "Name"], "rows": [["1", "Arnold"]]}
        label = json.dumps(ground_truth)
        sample = _make_sample("", label, mode="grid")
        reward = await zebralogicbench_reward(None, sample)
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_no_solution_key(self):
        from examples.zebralogicbench.zebralogicbench_reward import zebralogicbench_reward

        ground_truth = {
            "header": ["House", "Name"],
            "rows": [["1", "Arnold"]],
        }
        label = json.dumps(ground_truth)
        response = json.dumps({"reasoning": "I thought about it but have no solution"})
        sample = _make_sample(response, label, mode="grid")
        reward = await zebralogicbench_reward(None, sample)
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_json_wrapped_in_text(self):
        """Model output with JSON embedded in surrounding text."""
        from examples.zebralogicbench.zebralogicbench_reward import zebralogicbench_reward

        ground_truth = {
            "header": ["House", "Name"],
            "rows": [["1", "Arnold"]],
        }
        label = json.dumps(ground_truth)
        response = (
            "Let me work through this puzzle step by step.\n"
            "The answer is:\n"
            + json.dumps({
                "reasoning": "deduction",
                "solution": {"House 1": {"Name": "Arnold"}},
            })
            + "\nDone!"
        )
        sample = _make_sample(response, label, mode="grid")
        reward = await zebralogicbench_reward(None, sample)
        assert reward == 1.0


# ---------------------------------------------------------------------------
# MC-mode tests
# ---------------------------------------------------------------------------


class TestMCScoring:
    @pytest.mark.asyncio
    async def test_exact_match(self):
        from examples.zebralogicbench.zebralogicbench_reward import zebralogicbench_reward

        sample = _make_sample("The answer is Peter", label="Peter", mode="mc")
        reward = await zebralogicbench_reward(None, sample)
        assert reward == 1.0

    @pytest.mark.asyncio
    async def test_wrong_answer(self):
        from examples.zebralogicbench.zebralogicbench_reward import zebralogicbench_reward

        sample = _make_sample("The answer is Bob", label="Peter", mode="mc")
        reward = await zebralogicbench_reward(None, sample)
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_case_insensitive_mc(self):
        from examples.zebralogicbench.zebralogicbench_reward import zebralogicbench_reward

        sample = _make_sample("The answer is peter", label="Peter", mode="mc")
        reward = await zebralogicbench_reward(None, sample)
        assert reward == 1.0


# ---------------------------------------------------------------------------
# Answer extraction tests
# ---------------------------------------------------------------------------


class TestExtractLastCompleteJson:
    def test_simple_json(self):
        from examples.zebralogicbench.zebralogicbench_reward import _extract_last_complete_json

        result = _extract_last_complete_json('{"a": 1}')
        assert result == {"a": 1}

    def test_nested_json(self):
        from examples.zebralogicbench.zebralogicbench_reward import _extract_last_complete_json

        result = _extract_last_complete_json('Some text {"a": {"b": 2}} more text')
        assert result == {"a": {"b": 2}}

    def test_multiple_json_objects(self):
        from examples.zebralogicbench.zebralogicbench_reward import _extract_last_complete_json

        text = 'First {"a": 1} then {"b": 2}'
        result = _extract_last_complete_json(text)
        assert result == {"b": 2}

    def test_no_json(self):
        from examples.zebralogicbench.zebralogicbench_reward import _extract_last_complete_json

        result = _extract_last_complete_json("No JSON here")
        assert result is None

    def test_incomplete_json(self):
        from examples.zebralogicbench.zebralogicbench_reward import _extract_last_complete_json

        result = _extract_last_complete_json('{"a": 1')
        assert result is None
