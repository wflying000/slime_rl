"""Regression coverage for nested timeouts and the scoring worker protocol."""

import asyncio
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from recipe.nemotron_rl_reasoning_gym import reasoning_gym_reward as reward
from recipe.nemotron_rl_reasoning_gym import reasoning_gym_reward_worker as worker
from recipe.rl_reward.timeout_utils import RewardTimeout


def wrapped_timeout():
    error = RuntimeError("wrapped scoring failure")
    error.__cause__ = TimeoutError("symbolic deadline")
    return error


class TestNestedTimeouts(unittest.TestCase):
    def test_symbolic_timeout_cannot_become_a_wrong_answer(self):
        for factory in (lambda: TimeoutError("deadline"), wrapped_timeout):
            with self.subTest(exception=factory):
                sympy = ModuleType("sympy")
                sympy.sympify = Mock(side_effect=factory())
                sympy.Symbol = Mock(side_effect=factory())
                with patch.dict(sys.modules, {"sympy": sympy}):
                    with self.assertRaises(RewardTimeout):
                        reward._math_match("x + 1", "x")
                    with self.assertRaises(RewardTimeout):
                        reward._score_simple_integration("x", {"variable": "x", "integrand": "1"})

    def test_task_timeout_does_not_fall_back_to_cascade(self):
        with patch.dict(reward._TASK_SCORERS, {"test_timeout": Mock(side_effect=TimeoutError("deadline"))}), \
             patch.object(reward, "_cascade_score", side_effect=AssertionError("must not fall back")):
            with self.assertRaises(RewardTimeout):
                reward._score_answer("x", "x", {"source_dataset": "test_timeout"})

    def test_ordinary_task_error_still_uses_cascade(self):
        with patch.dict(reward._TASK_SCORERS, {"test_error": Mock(side_effect=ValueError("bad data"))}), \
             patch.object(reward, "_cascade_score", return_value=0.5):
            self.assertEqual(reward._score_answer("x", "x", {"source_dataset": "test_error"}), 0.5)

    def test_task_parser_timeout_is_not_a_partial_reward(self):
        with patch.object(reward, "_format_sig_figs", side_effect=TimeoutError("deadline")):
            with self.assertRaises(RewardTimeout):
                reward._score_power_function("1", {"ground_truth": "1"})
        with patch.object(reward, "_parse_hanoi_move", side_effect=wrapped_timeout()):
            with self.assertRaises(RewardTimeout):
                reward._score_tower_of_hanoi("move", {
                    "num_disks": 1, "num_pegs": 3, "start_peg": 1, "target_peg": 3,
                })

    def test_logging_timeout_propagates(self):
        with patch.object(reward, "REWARD_LOG_PATH", "/unused/reward.jsonl"), \
             patch.object(reward.os, "makedirs", side_effect=TimeoutError("log deadline")):
            with self.assertRaises(RewardTimeout):
                reward._append_reward_log("answer", "label", "answer", 1.0, {})

    def test_worker_reports_nested_timeout_explicitly(self):
        for exception in (RewardTimeout("deadline"), wrapped_timeout()):
            with self.subTest(exception=type(exception).__name__):
                output = io.StringIO()
                request = io.StringIO(json.dumps({"answer": "x", "label": "x", "metadata": {}}))
                with patch.object(sys, "stdin", request), patch.object(sys, "stdout", output), \
                     patch.object(reward, "_score_answer", side_effect=exception):
                    worker.main()
                result = json.loads(output.getvalue())
                self.assertIs(result["timeout"], True)
                self.assertNotIn("reward", result)

    def test_worker_script_imports_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(Path(worker.__file__).resolve())],
                input=json.dumps({"answer": "42", "label": "42", "metadata": {}}),
                capture_output=True, text=True, cwd=directory, timeout=5,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"reward": 1.0})


class TestInvalidTimeoutSamples(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        log_patch = patch.object(reward, "REWARD_LOG_PATH", None)
        log_patch.start()
        self.addCleanup(log_patch.stop)

    def sample(self):
        return SimpleNamespace(
            response="<answer>42</answer>", label="42", metadata={},
            response_length=3, loss_mask=[1, 1, 1], remove_sample=False,
        )

    def assert_invalid(self, sample, timeout=True):
        self.assertTrue(sample.metadata["reward_timeout" if timeout else "reward_error"])
        self.assertTrue(sample.remove_sample)
        self.assertEqual(sample.loss_mask, [0, 0, 0])

    async def test_worker_timeout_protocol_excludes_sample(self):
        process = SimpleNamespace(returncode=0, communicate=AsyncMock(return_value=(
            json.dumps({"timeout": True, "error": "inner deadline"}).encode(), b"",
        )))
        sample = self.sample()
        with patch.object(asyncio, "create_subprocess_exec", AsyncMock(return_value=process)):
            self.assertEqual(await reward.reasoning_gym_reward(None, sample), 0.0)
        self.assert_invalid(sample)

    async def test_wrapped_or_shared_timeout_excludes_sample(self):
        for exception in (TimeoutError("deadline"), RewardTimeout("deadline"), wrapped_timeout()):
            with self.subTest(exception=type(exception).__name__):
                sample = self.sample()
                with patch.object(reward, "_score_answer_in_subprocess", AsyncMock(side_effect=exception)):
                    self.assertEqual(await reward.reasoning_gym_reward(None, sample), 0.0)
                self.assert_invalid(sample)

    async def test_worker_failure_is_distinct_from_timeout(self):
        sample = self.sample()
        with patch.object(reward, "_score_answer_in_subprocess", AsyncMock(side_effect=reward.RewardWorkerError("worker failed"))):
            self.assertEqual(await reward.reasoning_gym_reward(None, sample), 0.0)
        self.assert_invalid(sample, timeout=False)
        self.assertNotIn("reward_timeout", sample.metadata)

    async def test_configuration_error_is_not_silenced(self):
        with patch.object(reward, "_score_answer_in_subprocess", AsyncMock(side_effect=ValueError("bad timeout"))):
            with self.assertRaises(ValueError):
                await reward.reasoning_gym_reward(None, self.sample())

    async def test_cancellation_propagates(self):
        with patch.object(reward, "_score_answer_in_subprocess", AsyncMock(side_effect=asyncio.CancelledError())):
            with self.assertRaises(asyncio.CancelledError):
                await reward.reasoning_gym_reward(None, self.sample())


if __name__ == "__main__":
    unittest.main()
