"""Slime ``custom_rm`` for code: pure correctness, no format requirement.

Returns ``1.0`` iff the code extracted from ``sample.response`` passes **all**
test cases; ``0.0`` otherwise. There is no reward shaping for ``<think>`` /
``</think>`` or for the presence of a ```python``` block label — any Python
code found in the response is used.

Code extraction order:
1. The last ``python``-labelled fenced block (``` ```python ... ``` ```).
2. The last fenced code block with any / no language tag.
3. The full response stripped, if it already looks like plain code.

Ground-truth (``sample.label`` JSON or ``sample.metadata['ground_truth']``):

* ``{"functional": "assert f(...) ..."}`` — single assertion block.
* ``{"functional": ["assert ...", ...]}`` — list of assertions; each is run
  independently and all must pass.
* ``{"pytest": "def test_..."}`` — a pytest file.
* ``{"inputs": [...], "outputs": [...]}`` — stdin/stdout pairs.
* Optional ``timeout`` (seconds).

Usage:
    --custom-rm-path recipe.rl_reward.joint_reward.joint_reward
    sample.metadata["data_type"] = "klear_code"

Like Guru-RL, the whole sample (including extraction and logging) runs in the
shared killable reward worker under JOINT_REWARD_SAMPLE_TIMEOUT_S. Individual
code executions retain their own test-case timeout and propagate timeouts to
the shared worker so invalid samples are excluded from training.
"""

from __future__ import annotations

from recipe.rl_reward.timeout_utils import raise_if_timeout

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from slime.utils.types import Sample

from .code_r1_reward import (
    BASE_IMPORTS,
    BASE_LEETCODE_IMPORTS,
    _DEFAULT_TIMEOUT_SECONDS,
    code_exec_firejail as code_executor,
)

logger = logging.getLogger(__name__)

# Set CODE_REWARD_LOG_PATH to enable per-sample reward logging.
REWARD_LOG_PATH: str | None = os.environ.get("CODE_REWARD_LOG_PATH") or None

# Per-sample test-case concurrency; sample concurrency is managed by
# JOINT_REWARD_WORKERS in the shared reward worker pool.
MAX_WORKERS = 2

_PY_FENCE_RE = re.compile(r"```(?:python|py)\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
_ANY_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)\n```", re.DOTALL)


def _extract_code(response: str) -> str | None:
    if not response:
        return None

    matches = _PY_FENCE_RE.findall(response)
    if matches:
        return matches[-1]

    matches = _ANY_FENCE_RE.findall(response)
    if matches:
        return matches[-1]

    stripped = response.strip()
    if stripped:
        return stripped
    return None


def _resolve_ground_truth(sample: Sample) -> dict[str, Any]:
    raw: Any = None
    if isinstance(sample.label, str) and sample.label.strip():
        raw = sample.label
    else:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        raw = metadata.get("ground_truth")

    if raw is None:
        raise ValueError(
            "code sample must provide ground truth via sample.label or "
            "sample.metadata['ground_truth']"
        )

    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, dict):
        return raw
    raise ValueError(f"unsupported ground_truth type: {type(raw).__name__}")


def _run_all_functional(solution_code: str, test_cases: Any, timeout: int) -> bool:
    prelude = "\n\n".join([BASE_IMPORTS, BASE_LEETCODE_IMPORTS, solution_code])

    if isinstance(test_cases, str):
        success, _ = code_executor(prelude + "\n\n" + test_cases, timeout=timeout)
        return bool(success)

    if isinstance(test_cases, list):
        if not test_cases:
            return False
        programs = [prelude + "\n\n" + case for case in test_cases]
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(programs))) as ex:
            futures = [ex.submit(code_executor, prog, None, timeout) for prog in programs]
            all_passed = True
            try:
                for fut in as_completed(futures):
                    if fut.cancelled():
                        continue
                    success, _ = fut.result()
                    if not success:
                        all_passed = False
                        for pending in futures:
                            pending.cancel()
            finally:
                for pending in futures:
                    if pending.cancelled() or pending.done():
                        continue
                    pending.cancel()
        return all_passed

    raise ValueError(f"unsupported functional test_cases shape: {type(test_cases).__name__}")


def _run_pytest(solution_code: str, pytest_src: Any, timeout: int) -> bool:
    del timeout  # code_exec_firejail controls its own timeout in pytest mode
    prelude = "\n\n".join([BASE_IMPORTS, BASE_LEETCODE_IMPORTS, solution_code])
    success, _ = code_executor(prelude, pytest=pytest_src)
    return bool(success)


def _run_all_stdio(solution_code: str, inputs: list[Any], outputs: list[Any], timeout: int) -> bool:
    if not inputs or len(inputs) != len(outputs):
        return False

    def _one(stdin: Any, expected: Any) -> bool:
        if isinstance(stdin, list):
            stdin = ", ".join(map(str, stdin))
        if isinstance(expected, list):
            expected = ", ".join(map(str, expected))
        success, output = code_executor(code=solution_code, stdin=stdin, timeout=timeout)
        if not success:
            return False
        return output.strip() == str(expected).strip()

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(inputs))) as ex:
        futures = [ex.submit(_one, i, o) for i, o in zip(inputs, outputs)]
        all_passed = True
        try:
            for fut in as_completed(futures):
                if fut.cancelled():
                    continue
                if not fut.result():
                    all_passed = False
                    for pending in futures:
                        pending.cancel()
        finally:
            for pending in futures:
                if pending.cancelled() or pending.done():
                    continue
                pending.cancel()
    return all_passed


def _score_sync(solution_code: str, ground_truth: dict[str, Any]) -> float:
    timeout = int(ground_truth.get("timeout", _DEFAULT_TIMEOUT_SECONDS))

    try:
        if "functional" in ground_truth:
            ok = _run_all_functional(solution_code, ground_truth["functional"], timeout)
        elif "pytest" in ground_truth:
            ok = _run_pytest(solution_code, ground_truth["pytest"], timeout)
        elif "inputs" in ground_truth and "outputs" in ground_truth:
            ok = _run_all_stdio(
                solution_code,
                list(ground_truth["inputs"]),
                list(ground_truth["outputs"]),
                timeout,
            )
        else:
            raise ValueError(f"unsupported test format: {list(ground_truth.keys())}")
    except Exception as exc:
        raise_if_timeout(exc)
        logger.warning(f"klear_code scoring crashed: {exc}; returning 0.0")
        return 0.0

    return 1.0 if ok else 0.0


def _append_reward_log(response: str, reward: float, metadata: dict[str, Any] | None) -> None:
    if REWARD_LOG_PATH is None:
        return
    record = {"reward": reward, "response": response, "metadata": metadata or {}}
    try:
        os.makedirs(os.path.dirname(REWARD_LOG_PATH), exist_ok=True)
        with open(REWARD_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        raise_if_timeout(exc)
        logger.warning(f"failed to append klear_code reward log: {exc}")


async def klear_code_reward(args, sample: Sample, **kwargs) -> float:
    del args, kwargs

    ground_truth = _resolve_ground_truth(sample)
    solution_code = _extract_code(sample.response or "")

    if not solution_code:
        reward = 0.0
    else:
        # The parent enforces the sample deadline and kills this worker and
        # its code-execution subprocesses on timeout, just as for Guru-RL.
        reward = _score_sync(solution_code, ground_truth)

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    _append_reward_log(sample.response or "", reward, metadata)
    return reward


custom_rm = klear_code_reward
