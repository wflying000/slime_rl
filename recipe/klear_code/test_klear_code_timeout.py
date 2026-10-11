"""Klear Code uses the same process deadline and invalid-sample contract as Guru-RL."""

import asyncio
from contextlib import ExitStack
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cloudpickle
import psutil

from recipe.rl_reward import isolated_reward as isolated
from recipe.rl_reward import joint_reward as joint
from recipe.rl_reward.timeout_utils import RewardTimeout

cloudpickle.register_pickle_by_value(sys.modules[__name__])


def make_sample(**metadata):
    return SimpleNamespace(
        metadata={"data_type": "klear_code", **metadata},
        response="print(1)", label=json.dumps({"inputs": [""], "outputs": ["1"]}),
        response_length=3, loss_mask=[1, 1, 1], remove_sample=False, index=0,
    )


async def controlled_code_route(args, sample, **kwargs):
    module = importlib.import_module("recipe.klear_code.klear_code_reward")
    stage = sample.metadata.get("blocked_stage")

    def block(*unused_args, **unused_kwargs):
        pids = {"worker": os.getpid()}
        if stage == "_score_sync":
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                start_new_session=True,
            )
            pids["child"] = child.pid
        Path(sample.metadata["marker"]).write_text(json.dumps(pids))
        while True:
            time.sleep(0.02)

    with ExitStack() as stack:
        stack.enter_context(patch.object(module, "_score_sync", return_value=1.0))
        if stage:
            stack.enter_context(patch.object(module, stage, side_effect=block))
        return await joint._score_klear_code(args, sample, **kwargs)


class TestKlearCodeTimeout(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = patch.dict(os.environ, {"JOINT_REWARD_WORKERS": "1", "CODE_REWARD_LOG_PATH": ""})
        env.start()
        self.addCleanup(env.stop)
        self.reward = importlib.import_module("recipe.klear_code.klear_code_reward")
        log = patch.object(self.reward, "REWARD_LOG_PATH", None)
        log.start()
        self.addCleanup(log.stop)

    async def asyncTearDown(self):
        await isolated.close_reward_workers()

    async def test_correct_and_wrong_code_keep_their_rewards(self):
        cases = [
            {"functional": "assert True", "timeout": 7},
            {"functional": ["assert True", "assert True"], "timeout": 7},
            {"inputs": [""], "outputs": ["1"], "timeout": 7},
            {"pytest": "def test_ok(): assert True", "timeout": 7},
        ]
        for truth in cases:
            for passed in [True, False]:
                with self.subTest(truth=truth, passed=passed):
                    sample = make_sample()
                    sample.label = json.dumps(truth)
                    with patch.object(self.reward, "code_executor", return_value=(passed, "1")) as execute:
                        self.assertEqual(await self.reward.klear_code_reward(None, sample), float(passed))
                    self.assertFalse(sample.remove_sample)
                    if "pytest" not in truth:
                        call = execute.call_args
                        self.assertEqual(call.kwargs.get("timeout", call.args[2] if len(call.args) > 2 else None), 7)

    async def test_execution_timeouts_are_not_wrong_answers(self):
        for truth in [{"functional": "assert True"}, {"functional": ["assert True"]},
                      {"inputs": [""], "outputs": ["1"]}, {"pytest": "def test_ok(): pass"}]:
            with self.subTest(truth=truth):
                sample = make_sample()
                sample.label = json.dumps(truth)
                with patch.object(self.reward, "code_executor", side_effect=TimeoutError("test execution timeout")):
                    with self.assertRaises(RewardTimeout):
                        await self.reward.klear_code_reward(None, sample)

    async def test_logging_timeout_propagates(self):
        with patch.object(self.reward, "REWARD_LOG_PATH", str(Path(self.tmp.name) / "reward.jsonl")), \
             patch("builtins.open", side_effect=TimeoutError("log write timeout")):
            with self.assertRaises(RewardTimeout):
                self.reward._append_reward_log("print(1)", 1.0, {})

    async def test_shared_deadline_covers_entire_sample_and_reaps_children(self):
        with patch.dict(joint._REWARD_SCORERS, {"klear_code": controlled_code_route}):
            for stage, deadline in [("_extract_code", 0.25), ("_score_sync", 0.5), ("_append_reward_log", 0.25)]:
                with self.subTest(stage=stage):
                    # Import the real scorer before starting the short deadline.
                    self.assertEqual(await joint.joint_reward(SimpleNamespace(joint_reward_sample_timeout_s=30), make_sample()), 1.0)
                    marker = Path(self.tmp.name) / stage
                    sample = make_sample(blocked_stage=stage, marker=str(marker))
                    with patch.dict(os.environ, {"JOINT_REWARD_SAMPLE_TIMEOUT_S": str(deadline)}):
                        started = time.monotonic()
                        pending = asyncio.create_task(joint.joint_reward(SimpleNamespace(), sample))
                        ticks = 0
                        while not pending.done():
                            await asyncio.sleep(0.02)
                            ticks += 1
                        self.assertEqual(await pending, 0.0)
                    elapsed = time.monotonic() - started
                    self.assertLess(elapsed, deadline + 3)
                    self.assertGreater(ticks, 2, "rollout event loop must remain responsive")
                    self.assertTrue(marker.exists(), "must reach scorer before timing out")
                    self.assertTrue(sample.metadata["reward_timeout"])
                    self.assertTrue(sample.remove_sample)
                    self.assertEqual(sample.loss_mask, [0, 0, 0])
                    for pid in json.loads(marker.read_text()).values():
                        try:
                            self.assertEqual(psutil.Process(pid).status(), psutil.STATUS_ZOMBIE)
                        except psutil.NoSuchProcess:
                            pass
                    self.assertFalse(isolated._LIVE_PIDS)
                    recovered = make_sample()
                    self.assertEqual(await joint.joint_reward(SimpleNamespace(joint_reward_sample_timeout_s=30), recovered), 1.0)
                    self.assertFalse(recovered.remove_sample)


if __name__ == "__main__":
    unittest.main()
