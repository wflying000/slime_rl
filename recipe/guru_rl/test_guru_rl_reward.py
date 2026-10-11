"""Quick smoke test for guru_rl_reward.

Runs without slime/torch dependency — mocks the Sample import.

Usage:
    cd /Users/a58/post-train/slime/examples/guru_rl
    python test_guru_rl_reward.py
"""

import asyncio
import sys
import os
from dataclasses import dataclass, field
from typing import Any
import types as _builtin_types

# ---------------------------------------------------------------------------
# Mock slime.utils.types.Sample BEFORE importing guru_rl_reward
# ---------------------------------------------------------------------------
_mock_types = _builtin_types.ModuleType("slime.utils.types")
_mock_types.Sample = dataclass(
    type(
        "Sample",
        (),
        {
            "__annotations__": {
                "prompt": "str | list[dict[str, str]]",
                "response": "str",
                "label": "Any",
                "metadata": "dict",
            },
            "prompt": "",
            "response": "",
            "label": None,
            "metadata": field(default_factory=dict),
        },
    )
)

# Inject mock into sys.modules before any slime import
sys.modules["slime"] = _builtin_types.ModuleType("slime")
sys.modules["slime.utils"] = _builtin_types.ModuleType("slime.utils")
sys.modules["slime.utils.types"] = _mock_types

# Now we can import the reward module
sys.path.insert(0, os.path.dirname(__file__))
from guru_rl_reward import guru_rl_reward, _score_answer

Sample = _mock_types.Sample


class _FakeArgs:
    pass


# ---- Zebra Puzzle ----
def _test_zebra():
    label = {
        "header": ["Color", "Pet"],
        "rows": [["Red", "Cat"], ["Blue", "Dog"]],
    }
    resp_correct = '<answer>{"header": ["Color", "Pet"], "rows": [["Red", "Cat"], ["Blue", "Dog"]]}</answer>'
    score = _score_answer(resp_correct, label, {"data_source": "logic__zebra_puzzle_dataset"})
    assert score == 1.0, f"zebra correct: expected 1.0, got {score}"

    resp_partial = '<answer>{"header": ["Color", "Pet"], "rows": [["Red", "Dog"], ["Blue", "Cat"]]}</answer>'
    score = _score_answer(resp_partial, label, {"data_source": "logic__zebra_puzzle_dataset"})
    assert score == 0.5, f"zebra partial: expected 0.5, got {score}"

    score = _score_answer("I think the answer is Red Cat", label, {"data_source": "logic__zebra_puzzle_dataset"})
    assert score == 0.0, f"zebra missing tag: expected 0.0, got {score}"

    print("  zebra_puzzle: PASS")


# ---- Graph Logical ----
def _test_graph():
    score = _score_answer(
        "<answer>True</answer>", "true", {"data_source": "logic__graph_logical_dataset"}
    )
    assert score == 1.0, f"graph correct: expected 1.0, got {score}"

    score = _score_answer(
        "<answer>False</answer>", "true", {"data_source": "logic__graph_logical_dataset"}
    )
    assert score == 0.0, f"graph wrong: expected 0.0, got {score}"

    score = _score_answer(
        "I think yes", "true", {"data_source": "logic__graph_logical_dataset"}
    )
    assert score == 0.0, f"graph no tag: expected 0.0, got {score}"

    print("  graph_logical: PASS")


# ---- Ordering Puzzle ----
def _test_ordering():
    label = ["wolf", "dolphin", "eagle"]

    score = _score_answer(
        '<answer>["wolf", "dolphin", "eagle"]</answer>',
        label,
        {"data_source": "logic__ordering_puzzle_dataset"},
    )
    assert score == 1.0, f"ordering exact: expected 1.0, got {score}"

    score = _score_answer(
        '<answer>["wolf", "eagle", "dolphin"]</answer>',
        label,
        {"data_source": "logic__ordering_puzzle_dataset"},
    )
    assert 0.0 < score < 1.0, f"ordering partial: expected 0<x<1, got {score}"

    score = _score_answer(
        "wolf dolphin eagle",
        label,
        {"data_source": "logic__ordering_puzzle_dataset"},
    )
    assert score == 0.0, f"ordering no tag: expected 0.0, got {score}"

    print("  ordering_puzzle: PASS")


# ---- Table ----
def _test_table():
    score = _score_answer(
        "The answer is \\boxed{42}", "42", {"data_source": "table__hitab"}
    )
    assert score == 1.0, f"table single: expected 1.0, got {score}"

    score = _score_answer(
        "The answer is \\boxed{42.001}", "42", {"data_source": "table__hitab"}
    )
    assert score == 1.0, f"table tolerance: expected 1.0, got {score}"

    score = _score_answer(
        "The answers are \\boxed{10|20|30}", "20|10|30", {"data_source": "table__multihier"}
    )
    assert score == 1.0, f"table multi: expected 1.0, got {score}"

    print("  table: PASS")


# ---- Async integration ----
async def _test_async():
    sample = Sample(
        response="<answer>True</answer>",
        label="true",
        metadata={"data_source": "logic__graph_logical_dataset", "ground_truth": "true"},
    )
    reward = await guru_rl_reward(_FakeArgs(), sample)
    assert reward == 1.0, f"async graph: expected 1.0, got {reward}"
    print("  async integration: PASS")


def main():
    print("Running guru_rl_reward tests...")
    _test_zebra()
    _test_graph()
    _test_ordering()
    _test_table()
    asyncio.run(_test_async())
    print("All tests passed!")


if __name__ == "__main__":
    main()
