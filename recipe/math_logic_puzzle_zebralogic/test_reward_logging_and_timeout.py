"""Coverage for reward JSONL records and timeout propagation."""

import asyncio
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from recipe.rl_reward.timeout_utils import RewardTimeout

reward = importlib.import_module(
    "recipe.math_logic_puzzle_zebralogic.math_logic_puzzle_zebralogic_reward"
)


def sample(response=r"\boxed{2}", label="2"):
    return SimpleNamespace(response=response, label=label,
        metadata={"data_type": "math_logic_puzzle", "ground_truth": label})


def wrapped_timeout():
    error = RuntimeError("wrapped scoring failure")
    error.__cause__ = TimeoutError("underlying deadline")
    return error


class ReadTimeout(Exception):
    pass


class TestRewardLogging(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.log_path = Path(self.directory.name) / "nested" / "reward.jsonl"
        log_patch = patch.object(reward, "REWARD_LOG_PATH", str(self.log_path))
        log_patch.start()
        self.addCleanup(log_patch.stop)

    def records(self):
        return [json.loads(line) for line in self.log_path.read_text().splitlines()]

    async def test_single_sample_records_result_and_original_fields(self):
        item = sample()
        self.assertEqual(await reward.custom_rm(None, item), 1.0)
        self.assertEqual(self.records(), [{
            "reward": 1.0, "label": item.label, "response": item.response,
            "metadata": item.metadata, "expected_answer": "2", "extracted_answer": "2",
            "error_type": None, "message": "",
        }])

    async def test_batch_appends_ordered_success_wrong_and_malformed_records(self):
        items = [sample(), sample(r"\boxed{1}"), sample("no answer"), sample(None)]
        self.assertEqual(await reward.custom_rm(None, items), [1.0, 0.0, 0.0, 0.0])
        self.assertEqual([r["error_type"] for r in self.records()],
                         [None, "wrong_answer", "no_boxed_answer", "invalid_response"])
        await reward.custom_rm(None, sample())
        self.assertEqual(len(self.records()), 5)

    async def test_disabled_logging_does_not_open_a_file(self):
        with patch.object(reward, "REWARD_LOG_PATH", None), \
             patch("builtins.open", side_effect=AssertionError("logging is disabled")):
            self.assertEqual(await reward.custom_rm(None, sample()), 1.0)
        self.assertFalse(self.log_path.exists())

    async def test_ordinary_log_error_does_not_change_reward(self):
        with patch("builtins.open", side_effect=OSError("log unavailable")), \
             self.assertLogs(reward.logger, level="WARNING"):
            self.assertEqual(await reward.custom_rm(None, sample()), 1.0)

    async def test_ordinary_scoring_error_is_logged_as_zero(self):
        with patch.object(reward, "_score_math_logic_puzzle_sample", side_effect=ValueError("bad sample")), \
             self.assertLogs(reward.logger, level="WARNING"):
            self.assertEqual(await reward.custom_rm(None, sample()), 0.0)
        self.assertEqual(self.records()[0]["error_type"], "reward_exception")
        self.assertEqual(self.records()[0]["message"], "bad sample")

    async def test_log_timeout_propagates(self):
        with patch("builtins.open", side_effect=TimeoutError("log deadline")):
            with self.assertRaises(RewardTimeout):
                await reward.custom_rm(None, sample())


class TestScoringTimeouts(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        log_patch = patch.object(reward, "REWARD_LOG_PATH", None)
        log_patch.start()
        self.addCleanup(log_patch.stop)

    async def test_plain_named_wrapped_and_shared_timeouts_propagate(self):
        for factory in (lambda: TimeoutError("deadline"), lambda: ReadTimeout("read deadline"),
                        wrapped_timeout, lambda: RewardTimeout("already classified")):
            for is_batch in (False, True):
                with self.subTest(exception=factory, batch=is_batch):
                    with patch.object(reward, "_score_math_logic_puzzle_sample", side_effect=factory()):
                        with self.assertRaises(RewardTimeout):
                            await reward.custom_rm(None, [sample()] if is_batch else sample())

    async def test_answer_extraction_timeout_is_not_zero_reward(self):
        with patch.object(reward, "_extract_last_boxed_answer", side_effect=wrapped_timeout()):
            with self.assertRaises(RewardTimeout):
                await reward.custom_rm(None, sample())

    async def test_cancellation_propagates(self):
        with patch.object(reward, "_score_math_logic_puzzle_sample", side_effect=asyncio.CancelledError()):
            with self.assertRaises(asyncio.CancelledError):
                await reward.custom_rm(None, sample())


if __name__ == "__main__":
    unittest.main()
