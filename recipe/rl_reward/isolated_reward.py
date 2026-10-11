"""Reusable CPU workers with a parent-enforced wall-clock reward deadline.

Only IPC and process supervision run on the rollout event loop. Synchronous
scorers, imports, network calls, and their logging all run in a child process.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import math
import os
import signal
import struct
import sys
from pathlib import Path
from weakref import WeakKeyDictionary

import cloudpickle
import psutil

from .timeout_utils import RewardTimeout, is_invalid_reward, mark_invalid_reward

logger = logging.getLogger(__name__)
MAX_FRAME_BYTES = 64 * 1024 * 1024
_POOLS = WeakKeyDictionary()
_LIVE_PIDS: set[int] = set()


def kill_worker_tree(pid: int) -> None:
    """Kill this worker's descendants, including sandboxes with their own PGID."""
    try:
        root = psutil.Process(pid)
        root.suspend()
        descendants = []
        # Freeze descendants before killing so a sandbox cannot fork another
        # child between enumeration and termination.
        for _ in range(2):
            descendants = root.children(recursive=True)
            for child in descendants:
                try:
                    child.suspend()
                except psutil.NoSuchProcess:
                    pass
        for child in reversed(descendants):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        root.kill()
    except psutil.NoSuchProcess:
        pass
    finally:
        _LIVE_PIDS.discard(pid)


@atexit.register
def _cleanup_workers() -> None:
    for pid in list(_LIVE_PIDS):
        try:
            kill_worker_tree(pid)
        except Exception:
            pass


class _Worker:
    def __init__(self):
        self.process = None

    async def start(self):
        if self.process is not None and self.process.returncode is None:
            return
        child_env = os.environ.copy()
        child_env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                         OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        # Include the checkout even when its caller did not export PYTHONPATH.
        root = str(Path(__file__).resolve().parents[2])
        child_env["PYTHONPATH"] = root + os.pathsep + child_env.get("PYTHONPATH", "")
        start = asyncio.create_task(asyncio.create_subprocess_exec(
            sys.executable, "-u", str(Path(__file__).with_name("reward_worker.py")),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=None, start_new_session=True, env=child_env,
        ))
        try:
            self.process = await asyncio.shield(start)
        except asyncio.CancelledError:
            # Reap a process created concurrently with cancellation.
            self.process = await start
            _LIVE_PIDS.add(self.process.pid)
            await self.close()
            raise
        _LIVE_PIDS.add(self.process.pid)

    async def close(self):
        process, self.process = self.process, None
        if process is None:
            return
        kill_worker_tree(process.pid)
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            logger.error("Reward worker %s did not reap within 5s after SIGKILL", process.pid)
        if process.stdin is not None:
            process.stdin.close()

    async def call(self, scorer, args, sample, kwargs):
        await self.start()
        payload = cloudpickle.dumps((scorer, args, sample, kwargs))
        if len(payload) > MAX_FRAME_BYTES:
            raise ValueError("Reward worker request exceeds 64 MiB")
        self.process.stdin.write(struct.pack("!I", len(payload)) + payload)
        await self.process.stdin.drain()
        size = struct.unpack("!I", await self.process.stdout.readexactly(4))[0]
        if size > MAX_FRAME_BYTES:
            raise ValueError("Reward worker response exceeds 64 MiB")
        return cloudpickle.loads(await self.process.stdout.readexactly(size))


class _Pool:
    def __init__(self, workers: int):
        self.workers = [_Worker() for _ in range(workers)]
        self.available = asyncio.LifoQueue()
        for worker in self.workers:
            self.available.put_nowait(worker)

    async def call(self, scorer, args, sample, kwargs):
        worker = await self.available.get()
        try:
            result = await worker.call(scorer, args, sample, kwargs)
            if result["invalid"]:
                # A timed-out executor thread may still be running even after
                # its scorer returns. Never reuse that worker.
                await worker.close()
            return result
        except BaseException:
            await worker.close()
            raise
        finally:
            self.available.put_nowait(worker)

    async def close(self):
        await asyncio.gather(*(worker.close() for worker in self.workers))


def get_timeout_s(args) -> float:
    value = getattr(args, "joint_reward_sample_timeout_s", None)
    if value is None:
        value = getattr(args, "joint_reward_sample_timeout_seconds", None)
    if value is None:
        value = os.getenv("JOINT_REWARD_SAMPLE_TIMEOUT_S",
                          os.getenv("JOINT_REWARD_SAMPLE_TIMEOUT_SECONDS", "240"))
    timeout_s = float(value)
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Reward timeout must be finite and positive; disabling it is not supported")
    return timeout_s


async def close_reward_workers() -> None:
    pool = _POOLS.pop(asyncio.get_running_loop(), None)
    if pool is not None:
        await pool.close()


async def score_with_timeout(scorer, args, sample, **kwargs):
    timeout_s = get_timeout_s(args)
    workers = int(os.getenv("JOINT_REWARD_WORKERS", "8"))
    if workers <= 0:
        raise ValueError("JOINT_REWARD_WORKERS must be positive")
    if is_invalid_reward(sample):
        metadata = getattr(sample, "metadata", None)
        timed_out = isinstance(metadata, dict) and bool(metadata.get("reward_timeout"))
        mark_invalid_reward(sample, timeout=timed_out, reason="sample already has an invalid reward")
        return 0.0
    loop = asyncio.get_running_loop()
    if loop not in _POOLS:
        _POOLS[loop] = _Pool(workers)
    try:
        result = await asyncio.wait_for(_POOLS[loop].call(scorer, args, sample, kwargs), timeout_s)
        # Preserve mutations such as timeout flags and OPD teacher logprobs.
        for name, value in result["updates"].items():
            setattr(sample, name, value)
        if result["invalid"]:
            mark_invalid_reward(sample, timeout=result["timeout"], reason=result["reason"])
            logger.warning("Reward invalid: index=%s timeout=%s reason=%s",
                           getattr(sample, "index", None), result["timeout"], result["reason"])
            return 0.0
        return result["reward"]
    except (asyncio.TimeoutError, RewardTimeout) as exc:
        mark_invalid_reward(sample, timeout=True, reason=f"reward deadline exceeded ({timeout_s}s): {exc}")
        logger.warning("Reward deadline exceeded: index=%s timeout=%ss", getattr(sample, "index", None), timeout_s)
        return 0.0
    except Exception as exc:
        mark_invalid_reward(sample, timeout=False, reason=f"reward worker failed: {type(exc).__name__}: {exc}")
        logger.exception("Reward worker failed; excluding sample index=%s", getattr(sample, "index", None))
        return 0.0
