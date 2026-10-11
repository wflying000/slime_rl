"""Tests for recipe/nemotron_rl_reasoning_gym/reasoning_gym_reward.py."""

import asyncio
import os
import sys
import types
import unittest
from unittest.mock import patch

# Ensure project root is on sys.path so both `examples.*` and `slime.*` resolve.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import recipe.nemotron_rl_reasoning_gym.reasoning_gym_reward as reward_module


def _run(coro):
    return asyncio.run(coro)


class TestReasoningGymAnswerExtraction(unittest.TestCase):
    def _extract(self, response: str) -> str:
        return reward_module._extract_answer(response)

    def test_answer_tag_has_highest_priority(self):
        response = "The answer is wrong. <answer>correct</answer><|im_end|>"
        self.assertEqual(self._extract(response), "correct")

    def test_boxed_answer(self):
        self.assertEqual(self._extract(r"Therefore, \boxed{42}."), "42")

    def test_boxed_answer_with_nested_braces(self):
        self.assertEqual(self._extract(r"Therefore, \boxed{\frac{1}{2}}."), r"\frac{1}{2}")

    def test_last_boxed_answer_wins(self):
        self.assertEqual(self._extract(r"First \boxed{1}, finally \boxed{2}."), "2")

    def test_answer_colon(self):
        self.assertEqual(self._extract("Answer: xxx"), "xxx")

    def test_lowercase_answer_colon(self):
        self.assertEqual(self._extract("answer: xxx."), "xxx")

    def test_final_answer_colon(self):
        self.assertEqual(self._extract("final answer: xxx"), "xxx")

    def test_markdown_final_answer_colon(self):
        self.assertEqual(self._extract("**Final Answer:** xxx<|endoftext|>"), "xxx")

    def test_answer_fullwidth_colon(self):
        self.assertEqual(self._extract("Answer：xxx"), "xxx")

    def test_final_answer_is_phrase(self):
        self.assertEqual(self._extract("Therefore the final answer is xxx."), "xxx")

    def test_strips_chain_of_thought_before_extracting(self):
        response = "<think>Answer: wrong</think>\nFinal Answer: right"
        self.assertEqual(self._extract(response), "right")


class TestReasoningGymRewardExtraction(unittest.TestCase):
    def setUp(self):
        self.log_patch = patch.object(reward_module, "REWARD_LOG_PATH", None)
        self.log_patch.start()
        self.addCleanup(self.log_patch.stop)

    def _score(self, response: str, label: str) -> float:
        sample = types.SimpleNamespace(
            response=response,
            label=label,
            metadata={"source_dataset": "unknown_dataset", "ground_truth": label},
        )
        return _run(reward_module.reasoning_gym_reward(None, sample))

    def test_reward_uses_boxed_answer(self):
        self.assertEqual(self._score(r"Reasoning... \boxed{42}", "42"), 1.0)

    def test_reward_uses_answer_colon(self):
        self.assertEqual(self._score("Reasoning...\nAnswer: hello", "hello"), 1.0)

    def test_reward_uses_final_answer_colon(self):
        self.assertEqual(self._score("Reasoning...\nfinal answer: hello", "hello"), 1.0)

    def test_reward_uses_final_answer_is_phrase(self):
        self.assertEqual(self._score("Reasoning... the final answer is hello.", "hello"), 1.0)


if __name__ == "__main__":
    unittest.main()
