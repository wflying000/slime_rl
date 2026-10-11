"""Regression tests for timeout propagation in CPU-only reward scoring."""

import asyncio
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

# Match the existing reward tests: torch is only needed by Sample annotations.
if "torch" not in sys.modules:
    sys.modules["torch"] = MagicMock()

from recipe.rl_reward.timeout_utils import RewardTimeout
from recipe.zebralogicbench import zebralogicbench_reward as reward


class ReadTimeout(Exception):
    pass


def wrapped_timeout():
    error = RuntimeError("wrapped scoring failure")
    error.__cause__ = TimeoutError("underlying deadline")
    return error


class TestRewardTimeoutPropagation(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        log_patch = patch.object(reward, "REWARD_LOG_PATH", None)
        log_patch.start()
        self.addCleanup(log_patch.stop)

    def sample(self, mode="grid"):
        label = "A" if mode == "mc" else {"header": ["House", "Name"], "rows": [["1", "Alice"]]}
        return SimpleNamespace(response='{"solution": {"House 1": {"Name": "Alice"}}}',
                               label=label, metadata={"mode": mode})

    async def test_grid_and_mc_timeouts_are_not_zero_rewards(self):
        for mode, scorer in (("grid", "_score_grid"), ("mc", "_score_mc")):
            for factory in (lambda: TimeoutError("deadline"), lambda: ReadTimeout("read deadline"),
                            wrapped_timeout, lambda: RewardTimeout("already classified")):
                with self.subTest(mode=mode, exception=factory):
                    with patch.object(reward, scorer, side_effect=factory()):
                        with self.assertRaises(RewardTimeout):
                            await reward.custom_rm(None, self.sample(mode))

    async def test_ordinary_scoring_errors_still_return_zero(self):
        for mode, scorer in (("grid", "_score_grid"), ("mc", "_score_mc")):
            with self.subTest(mode=mode), patch.object(reward, scorer, side_effect=ValueError("bad answer")):
                self.assertEqual(await reward.custom_rm(None, self.sample(mode)), 0.0)

    async def test_json_decoding_error_with_timeout_cause_propagates(self):
        error = json.JSONDecodeError("wrapped failure", "{}", 0)
        error.__cause__ = TimeoutError("parse deadline")
        with patch.object(reward.json, "loads", side_effect=error):
            with self.assertRaises(RewardTimeout):
                await reward.custom_rm(None, self.sample())

    async def test_label_decoding_timeout_propagates(self):
        sample = self.sample()
        sample.label = json.dumps(sample.label)
        with patch.object(reward.json, "loads", side_effect=TimeoutError("label deadline")):
            with self.assertRaises(RewardTimeout):
                await reward.custom_rm(None, sample)

    async def test_logging_timeout_propagates(self):
        with patch.object(reward, "REWARD_LOG_PATH", "/unused/zebra/reward.jsonl"), \
             patch.object(reward.os, "makedirs", side_effect=TimeoutError("log deadline")):
            with self.assertRaises(RewardTimeout):
                await reward.custom_rm(None, self.sample())

    async def test_ordinary_log_failure_preserves_reward(self):
        with patch.object(reward, "REWARD_LOG_PATH", "/unused/zebra/reward.jsonl"), \
             patch.object(reward.os, "makedirs", side_effect=OSError("log unavailable")):
            self.assertEqual(await reward.custom_rm(None, self.sample()), 1.0)

    async def test_cancellation_is_not_silenced(self):
        with patch.object(reward, "_score_grid", side_effect=asyncio.CancelledError()):
            with self.assertRaises(asyncio.CancelledError):
                await reward.custom_rm(None, self.sample())


if __name__ == "__main__":
    unittest.main()
