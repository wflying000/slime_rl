"""Code scoring logic ported from KlearReasoner.

Source: ``KlearReasoner/verl/utils/reward_score/code_r1_compute.py``. Only
differences from upstream:

* Imports reference this slime example package (``reward_config``,
  ``code_r1_reward``) instead of verl paths.
* Type hint refinements to satisfy Pyright.
"""

from __future__ import annotations

import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

from .code_r1_reward import (
    BASE_IMPORTS,
    BASE_LEETCODE_IMPORTS,
    _DEFAULT_TIMEOUT_SECONDS,
    code_exec_firejail as code_executor,
)
from .reward_config import ScoringConfig

MAX_WORKERS = 2
MAX_TEST_CASES = 16
ERROR_MSG_PREFIX = "Execution failed: "

scoring_config = ScoringConfig()
CODE_EXTRACTION_PATTERN = re.compile(r"</think>.*```python\n(.*?)\n```", re.DOTALL)


def extract_code(content: str) -> str | None:
    """Pull out the Python code block after ``</think>`` (klear's exact rule)."""
    match = CODE_EXTRACTION_PATTERN.search(content)
    return match.group(1) if match else None


def execute_code(code: str, input_data: str, expected_output: str, timeout: int):
    success, output = code_executor(code=code, stdin=input_data, timeout=timeout)
    return success, output, input_data, expected_output


def sample_test_cases(inputs: list, outputs: list, max_samples: int):
    """Randomly sample up to ``max_samples`` (input, output) pairs (klear logic)."""
    assert len(inputs) == len(outputs)
    indices = random.sample(range(len(inputs)), min(len(inputs), max_samples))
    sampled_inputs: list[str] = []
    sampled_outputs: list[str] = []

    for idx in indices:
        in_data = inputs[idx]
        out_data = outputs[idx]

        if isinstance(in_data, list):
            in_data = ", ".join(map(str, in_data))
        if isinstance(out_data, list):
            out_data = ", ".join(map(str, out_data))

        sampled_inputs.append(in_data)
        sampled_outputs.append(out_data)

    return sampled_inputs, sampled_outputs


def _compute_score(solution: str, ground_truth, extra_info) -> tuple[float, str]:
    log_messages: list[str] = []
    solution_code = extract_code(solution)

    if not solution_code:
        log_messages.append("─" * 20 + " Invalid Code Format " + "─" * 20)
        log_messages.append(solution)
        return scoring_config.format_error_score, "\n".join(log_messages)

    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)

    timeout = ground_truth.get("timeout", _DEFAULT_TIMEOUT_SECONDS)
    max_test_cases = ground_truth.get("max_test_cases", MAX_TEST_CASES)
    log_messages.append("─" * 20 + " Valid Code Detected " + "─" * 20)

    _start_time = time.time()

    if "pytest" in ground_truth or "functional" in ground_truth:
        solution_code = "\n\n".join([BASE_IMPORTS, BASE_LEETCODE_IMPORTS, solution_code])

        if "functional" in ground_truth:
            test_cases = ground_truth["functional"]

            if isinstance(test_cases, str):
                all_result: list[float] = []
                success, _output = code_executor(solution_code + "\n\n" + test_cases, timeout=timeout)
                all_result.append(scoring_config.correct_score if success else scoring_config.incorrect_score)

            elif isinstance(test_cases, list):
                selected = random.sample(test_cases, min(max_test_cases, len(test_cases)))
                test_programs = [solution_code + "\n\n" + case for case in selected]

                with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(test_programs))) as executor:
                    futures = [executor.submit(code_executor, prog, None, timeout) for prog in test_programs]
                    all_result = []
                    for future in as_completed(futures):
                        success, _output = future.result()
                        all_result.append(scoring_config.correct_score if success else scoring_config.incorrect_score)
            else:
                raise ValueError("Unsupported functional test format")
        else:
            all_result = []
            success, _output = code_executor(solution_code, pytest=ground_truth["pytest"])
            all_result.append(scoring_config.correct_score if success else scoring_config.incorrect_score)

    elif "inputs" in ground_truth and "outputs" in ground_truth:
        inputs, outputs = sample_test_cases(
            ground_truth["inputs"],
            ground_truth["outputs"],
            max_test_cases,
        )

        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(inputs))) as executor:
            futures = [executor.submit(execute_code, solution_code, i, o, timeout) for i, o in zip(inputs, outputs)]
            all_result = []
            for future in as_completed(futures):
                success, output, _stdin, stdout = future.result()
                if not success or output.strip() != stdout.strip():
                    all_result.append(scoring_config.incorrect_score)
                else:
                    all_result.append(scoring_config.correct_score)

    elif "inputs_file" in ground_truth and "outputs_file" in ground_truth:
        with open(ground_truth["inputs_file"]) as f:
            inputs = [f.read()]
        with open(ground_truth["outputs_file"]) as f:
            outputs = [f.read()]

        all_result = []
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(inputs))) as executor:
            futures = [executor.submit(execute_code, solution_code, i, o, timeout) for i, o in zip(inputs, outputs)]
            for future in as_completed(futures):
                success, output, _stdin, stdout = future.result()
                if not success or output.strip() != stdout.strip():
                    all_result.append(scoring_config.incorrect_score)
                else:
                    all_result.append(scoring_config.correct_score)

    else:
        raise ValueError(f"Unsupported test format: {list(ground_truth.keys())}")

    if not all_result:
        return scoring_config.incorrect_score, "\n".join(log_messages)
    return sum(all_result) * 1.0 / len(all_result), "\n".join(log_messages)


def compute_score(
    solution: str,
    ground_truth: Any,
    extra_info: Any = None,
    debug: bool = False,
    is_longcot: bool = False,
) -> float:
    """Score a code response. Semantics identical to klear's ``code_r1_compute``."""
    del debug

    if is_longcot:
        if "</think>" not in solution:
            return scoring_config.wo_eos_think
        if "<think>" not in solution:
            return scoring_config.wo_bos_think

    if extra_info is None:
        extra_info = []
    if isinstance(extra_info, np.ndarray):
        extra_info = extra_info.item()

    score, _log = _compute_score(solution=solution, ground_truth=ground_truth, extra_info=extra_info)
    return score
