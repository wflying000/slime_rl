"""Regression tests for the rollout-56 SymPy hang and process cleanup."""

import asyncio
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from recipe.nemotron_rl_reasoning_gym import reasoning_gym_reward as reward


# Captured from the stuck RolloutManager on 2026-09-12 (source_index 13024).
HANGING_ANSWER = "(3/4)*cos((2*x - 9)**4) + C"
HANGING_LABEL = "3*cos(2*x - 9)**4/4 + C"


class TestCascadeShortCircuit(unittest.TestCase):
    def test_exact_and_numeric_matches_skip_symbolic_work(self):
        with patch.object(reward, "_math_match", side_effect=AssertionError("unnecessary SymPy")):
            self.assertEqual(reward._cascade_score("x + 1", "x + 1"), 1.0)
            self.assertEqual(reward._cascade_score("2.0", "2"), 1.0)
            self.assertEqual(reward._cascade_score(r"\text{42}", "42"), 1.0)


class TestScoringSubprocess(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.env_patch = patch.dict(os.environ, {
            "REASONING_GYM_REWARD_TIMEOUT_S": "2",
            "REASONING_GYM_REWARD_WORKERS": "2",
        })
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.log_patch = patch.object(reward, "REWARD_LOG_PATH", None)
        self.log_patch.start()
        self.addCleanup(self.log_patch.stop)
        self.processes = []
        self.create_process = asyncio.create_subprocess_exec

    async def record_process(self, *args, **kwargs):
        process = await self.create_process(*args, **kwargs)
        self.processes.append(process)
        return process

    async def test_real_hang_times_out_without_blocking_loop_and_next_sample_scores(self):
        sample = SimpleNamespace(
            response=f"<answer>{HANGING_ANSWER}</answer>",
            label=HANGING_LABEL,
            metadata={"source_dataset": "intermediate_integration", "source_index": 13024},
            index=13024,
            remove_sample=False,
        )
        ticks = 0

        async def heartbeat():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.05)
                ticks += 1

        ticker = asyncio.create_task(heartbeat())
        started = time.monotonic()
        try:
            with patch.object(asyncio, "create_subprocess_exec", self.record_process):
                self.assertEqual(await reward.reasoning_gym_reward(None, sample), 0.0)
        finally:
            ticker.cancel()
            await asyncio.gather(ticker, return_exceptions=True)
        self.assertLess(time.monotonic() - started, 8)
        self.assertGreaterEqual(ticks, 5)
        self.assertTrue(sample.metadata["reward_timeout"])
        self.assertTrue(sample.remove_sample)
        self.assertEqual(len(self.processes), 1)
        self.assertIsNotNone(self.processes[0].returncode)

        next_sample = SimpleNamespace(
            response="<answer>(x + 1)**2</answer>",
            label="x**2 + 2*x + 1",
            metadata={},
            remove_sample=False,
        )
        self.assertEqual(await reward.reasoning_gym_reward(None, next_sample), 1.0)
        self.assertFalse(next_sample.remove_sample)
        self.assertNotIn("reward_timeout", next_sample.metadata)

    async def test_correct_task_specific_and_wrong_answers(self):
        self.assertEqual(await reward._score_answer_in_subprocess(
            "Hello World", "unused", {"source_dataset": "needle_haystack", "ground_truth": "helloworld"}
        ), 1.0)
        self.assertEqual(await reward._score_answer_in_subprocess("x+2", "x+1", {}), 0.0)

    async def test_concurrency_is_bounded_and_cancellation_reaps_workers(self):
        with patch.object(asyncio, "create_subprocess_exec", self.record_process):
            tasks = [asyncio.create_task(reward._score_answer_in_subprocess(
                HANGING_ANSWER, HANGING_LABEL, {}
            )) for _ in range(4)]
            try:
                for _ in range(100):
                    if len(self.processes) >= 2:
                        break
                    await asyncio.sleep(0.01)
                await asyncio.sleep(0.1)
                self.assertEqual(len(self.processes), 2)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        self.assertTrue(all(process.returncode is not None for process in self.processes))


if __name__ == "__main__":
    unittest.main()
