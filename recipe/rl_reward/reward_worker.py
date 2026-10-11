"""Private binary IPC worker for isolated_reward; one sample at a time."""

from __future__ import annotations

import asyncio
import ctypes
import inspect
import math
import os
import signal
import struct
import sys

import cloudpickle

from recipe.rl_reward.timeout_utils import is_invalid_reward, is_timeout_error


async def _score(request):
    scorer, args, sample, kwargs = request
    try:
        value = scorer(args, sample, **kwargs)
        reward = await value if inspect.isawaitable(value) else value
        if isinstance(reward, (int, float)) and not math.isfinite(reward):
            raise ValueError("non-finite reward")
        metadata = getattr(sample, "metadata", None) or {}
        return dict(reward=reward, invalid=is_invalid_reward(sample),
                    timeout=bool(metadata.get("reward_timeout")),
                    reason=metadata.get("reward_invalid_reason", "inner scorer marked sample invalid"),
                    updates={name: getattr(sample, name) for name in
                             ("metadata", "remove_sample", "loss_mask", "teacher_log_probs")
                             if hasattr(sample, name)})
    except BaseException as exc:
        return dict(reward=0.0, invalid=True, timeout=is_timeout_error(exc),
                    reason=f"{type(exc).__name__}: {exc}"[:2000], updates={})


async def main():
    # Reward libraries may write directly to fd 1; keep those bytes out of IPC.
    output = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    if sys.platform == "linux":
        parent = os.getppid()
        ctypes.CDLL(None).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
        if os.getppid() != parent:
            return
    while True:
        header = sys.stdin.buffer.read(4)
        if not header:
            return
        if len(header) != 4:
            raise EOFError("incomplete reward request header")
        size = struct.unpack("!I", header)[0]
        if size > 64 * 1024 * 1024:
            raise ValueError("oversized reward request")
        payload = sys.stdin.buffer.read(size)
        if len(payload) != size:
            raise EOFError("incomplete reward request")
        result = await _score(cloudpickle.loads(payload))
        response = cloudpickle.dumps(result)
        output.write(struct.pack("!I", len(response)) + response)
        # Do not await hung executor shutdown on failure. The parent destroys
        # this process tree immediately after receiving the invalid result.


if __name__ == "__main__":
    asyncio.run(main())
