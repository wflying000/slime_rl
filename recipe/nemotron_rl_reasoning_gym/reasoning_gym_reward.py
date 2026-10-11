# coding: utf-8
"""Reward function for Nemotron-RL-ReasoningGym RLVR training.

Implements scoring logic adapted from the reasoning-gym project, with
task-specific verifiers for datasets that have custom scoring and a
cascade fallback (string → float → math match) for the rest.

Usage:
    --custom-rm-path examples.nemotron_rl_reasoning_gym.reasoning_gym_reward.reasoning_gym_reward
"""

from __future__ import annotations

from recipe.rl_reward.timeout_utils import (
    RewardTimeout,
    is_timeout_error,
    mark_invalid_reward,
    raise_if_timeout,
)

import asyncio
import json
import logging
import math
import os
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
from weakref import WeakKeyDictionary

if TYPE_CHECKING:
    from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# Set REASONING_GYM_REWARD_LOG_PATH to enable per-sample reward logging.
REWARD_LOG_PATH: str | None = os.environ.get("REASONING_GYM_REWARD_LOG_PATH") or None

# ---------------------------------------------------------------------------
# Reward logging
# ---------------------------------------------------------------------------


def _append_reward_log(
    response: str,
    label: str,
    answer: str,
    reward: float,
    metadata: dict[str, Any] | None,
) -> None:
    if REWARD_LOG_PATH is None:
        return
    record = {
        "reward": reward,
        "label": label,
        "answer": answer,
        "response": response,
        "metadata": metadata or {},
    }
    try:
        os.makedirs(os.path.dirname(REWARD_LOG_PATH), exist_ok=True)
        with open(REWARD_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        raise_if_timeout(exc)
        logger.warning(f"failed to append reasoning_gym reward log: {exc}")


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

# Find delimiters separately: adjacent unbounded whitespace expressions in the
# old regex backtracked for hours on a generated 38K-character whitespace run.
_ANSWER_TAG_OPEN_RE = re.compile(r"<answer>", re.IGNORECASE)
_ANSWER_TAG_CLOSE_RE = re.compile(r"</answer>$", re.IGNORECASE)
_ANSWER_END_RE = re.compile(r"<\|im_end\|>|<\|endoftext\|>", re.IGNORECASE)


def _extract_tagged_answer(text: str) -> Optional[str]:
    start = _ANSWER_TAG_OPEN_RE.search(text)
    if start is None:
        return None
    stop = _ANSWER_END_RE.search(text, start.end())
    answer = text[start.end() : stop.start() if stop else len(text)].strip()
    # Preserve the old optional closing tag, EOF fallback and first-opening-tag
    # precedence without a wildcard/whitespace backtracking combination.
    return _ANSWER_TAG_CLOSE_RE.sub("", answer).strip()


_ANSWER_LINE_RE = re.compile(
    r"(?im)^\s*(?:\*\*[ \t]*)?(?:final\s+)?answer[ \t]*[:：][ \t]*(?:\*\*[ \t]*)?(.+?)\s*$"
)
_ANSWER_IS_RE = re.compile(r"(?is)\b(?:final\s+)?answer\s+is\s+([^\n]+)")


def _strip_chain_of_thought(text: str) -> str:
    """Remove <think>...</think> block and trailing special tokens."""
    if not text:
        return ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    text = text.replace("<|im_end|>", "").replace("</s>", "")
    return text


def _strip_boxed_answer(text: str) -> Optional[str]:
    """Extract the last LaTeX \boxed{...} answer, handling nested braces."""
    last_answer = None
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = match.end()
        depth = 1
        pos = start
        while pos < len(text) and depth > 0:
            char = text[pos]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            last_answer = text[start : pos - 1]
    return _normalize_extracted_answer(last_answer) if last_answer is not None else None


def _normalize_extracted_answer(answer: str) -> str:
    """Clean common wrappers and trailing punctuation from extracted answers."""
    answer = answer.strip()
    answer = answer.replace("<|im_end|>", "").replace("<|endoftext|>", "").replace("</s>", "").strip()
    # Scan only the suffix; regex search from every position is quadratic on
    # long whitespace/markup runs that are followed by ordinary text.
    end = len(answer)
    pos = end
    while pos and answer[pos - 1] in "*_`":
        pos -= 1
    if pos < end:
        while pos and answer[pos - 1] in " \t":
            pos -= 1
        answer = answer[:pos]
    # Python's legacy $ anchor also matched immediately before a final newline.
    end = len(answer) - int(answer.endswith("\n"))
    pos = end
    while pos and answer[pos - 1] in ".。,;:!":
        pos -= 1
    if pos < end:
        # Match the old (?<=\S) behavior, including punctuation-only strings.
        if pos == 0 or answer[pos - 1].isspace():
            pos += 1
        answer = answer[:pos] + answer[end:]
    return answer.strip()


def _extract_answer(response: str) -> str:
    """Extract the final answer from model response.

    Priority:
    1. Content inside <answer>...</answer> tags
    2. Content inside the last \boxed{...}
    3. Last answer line like "Answer: ..." or "Final Answer: ..."
    4. Last phrase like "answer is ..." or "final answer is ..."
    5. Full response after stripping chain-of-thought
    """
    text = _strip_chain_of_thought(response)

    # Try <answer>...</answer> tags first
    tagged_answer = _extract_tagged_answer(text)
    if tagged_answer is not None:
        return _normalize_extracted_answer(tagged_answer)

    boxed_answer = _strip_boxed_answer(text)
    if boxed_answer:
        return boxed_answer

    answer_line_matches = _ANSWER_LINE_RE.findall(text)
    if answer_line_matches:
        answer = _normalize_extracted_answer(answer_line_matches[-1])
        if answer:
            return answer

    answer_is_matches = _ANSWER_IS_RE.findall(text)
    if answer_is_matches:
        answer = _normalize_extracted_answer(answer_is_matches[-1])
        if answer:
            return answer

    return _normalize_extracted_answer(text)


def _resolve_label(sample: Sample) -> str:
    """Resolve ground truth label: sample.label, then metadata['ground_truth']."""
    if isinstance(sample.label, str) and sample.label.strip():
        return sample.label.strip()

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    gt = metadata.get("ground_truth")
    if isinstance(gt, str) and gt.strip():
        return gt.strip()

    raise ValueError(
        "sample must provide a non-empty label via sample.label or sample.metadata['ground_truth']"
    )


# ---------------------------------------------------------------------------
# Generic cascade matchers (from reasoning-gym scoring.py)
# ---------------------------------------------------------------------------


def _strip_latex(s: str) -> str:
    """Remove common LaTeX wrappers and normalise whitespace."""
    s = re.sub(r"^\\\((.*)\\\)$", r"\1", s.strip())
    s = re.sub(r"\\\((.*?)\\\)", r"\1", s)
    s = re.sub(r"\\(?:text|mathrm)\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\\\+", " ", s)
    s = re.sub(r"~", " ", s)
    s = re.sub(r"\\", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _string_match(predicted: str, expected: str) -> float:
    """Case-insensitive exact string comparison after stripping whitespace."""
    try:
        return 1.0 if predicted.lower().strip() == expected.lower().strip() else 0.0
    except Exception as exc:
        raise_if_timeout(exc)
        return 0.0


def _float_match(
    predicted: str,
    expected: str,
    rel_tol: float = 0.01,
    abs_tol: float = 0.01,
) -> float:
    """Numeric comparison with configurable tolerance."""
    try:
        a = float(predicted)
        b = float(expected)
        return 1.0 if abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol) else 0.0
    except Exception as exc:
        raise_if_timeout(exc)
        return 0.0


def _math_match(predicted: str, expected: str) -> float:
    """Symbolic math verification via sympy.

    Checks if the derivative of the predicted expression equals the
    expected, or if symbolic simplification shows equivalence.
    """
    try:
        import sympy

        # Try direct symbolic equivalence
        p = sympy.sympify(predicted)
        e = sympy.sympify(expected)
        if sympy.simplify(p - e) == 0:
            return 1.0
    except Exception as exc:
        raise_if_timeout(exc)
        pass
    return 0.0


def _cascade_score(answer: str, expected: str) -> float:
    """Apply multi-step scoring cascade (string → float → math).

    The cascade can only upgrade — if an earlier step already returned
    a near-perfect score (>= 0.99) it is returned immediately.
    """
    best = 0.0
    # Evaluate lazily: constructing a tuple of scores runs SymPy even when
    # the first string comparison already matches.
    for score_fn in (
        lambda: _string_match(answer, expected),
        lambda: _string_match(_strip_latex(answer), _strip_latex(expected)),
        lambda: _float_match(answer, expected),
        lambda: _math_match(answer, expected),
        lambda: _math_match(
            re.sub(r"\\mathrm\{([^}]*)\}", r"\\text{\1}", answer),
            re.sub(r"\\mathrm\{([^}]*)\}", r"\\text{\1}", expected),
        ),
    ):
        score = score_fn()
        if score >= 0.99:
            return score
        best = max(best, score)
    return best


# ---------------------------------------------------------------------------
# Task-specific scorers
# ---------------------------------------------------------------------------


def _score_needle_haystack(answer: str, metadata: dict) -> float:
    """Case-insensitive, no-space match for needle_haystack."""
    if not isinstance(answer, str):
        return 0.0
    correct = metadata.get("ground_truth", "")
    answer = answer.replace(" ", "").strip().lower()
    correct = correct.strip().lower()
    return 1.0 if answer == correct else 0.0


def _score_cryptarithm(answer: str, metadata: dict) -> float:
    """Parse letter=digit mapping and verify arithmetic constraints."""
    if not isinstance(answer, str):
        return 0.0
    try:
        predicted_mapping = {}
        for pair in answer.split(","):
            letter, digit_str = pair.strip().split("=")
            letter = letter.strip()
            predicted_mapping[letter] = int(digit_str.strip())
    except (ValueError, AttributeError) as exc:
        raise_if_timeout(exc)
        return 0.0

    words_letters = metadata.get("words_letters", [])
    result_letters = metadata.get("result_letters", "")
    allow_leading_zero = metadata.get("allow_leading_zero", False)

    is_valid, _ = _verify_cryptarithm(
        predicted_mapping, words_letters, result_letters, allow_leading_zero
    )
    return 1.0 if is_valid else 0.01


def _verify_cryptarithm(
    mapping: dict[str, int],
    words_letters: list[str],
    result_letters: str,
    allow_leading_zero: bool,
) -> tuple[bool, str]:
    """Validate cryptarithm letter-to-digit mapping."""
    all_puzzle_letters = set()
    for word in words_letters:
        all_puzzle_letters.update(word)
    all_puzzle_letters.update(result_letters)

    if set(mapping.keys()) != all_puzzle_letters:
        return False, "letter set mismatch"

    for letter, digit in mapping.items():
        if not isinstance(digit, int) or digit < 0 or digit > 9:
            return False, f"invalid digit for {letter}"

    if len(set(mapping.values())) != len(mapping.values()):
        return False, "duplicate digit assignments"

    if not allow_leading_zero:
        for word in words_letters:
            if word and mapping.get(word[0]) == 0:
                return False, f"leading zero in {word}"
        if result_letters and mapping.get(result_letters[0]) == 0:
            return False, "leading zero in result"

    try:
        word_numbers = []
        for word in words_letters:
            number_str = "".join(str(mapping[letter]) for letter in word)
            word_numbers.append(int(number_str))
        result_number_str = "".join(str(mapping[letter]) for letter in result_letters)
        result_number = int(result_number_str)
        if sum(word_numbers) != result_number:
            return False, "arithmetic not satisfied"
    except (KeyError, ValueError) as e:
        raise_if_timeout(e)
        return False, str(e)

    return True, ""


def _score_simple_integration(answer: str, metadata: dict) -> float:
    """Verify integration by checking derivative of user's answer."""
    if not answer:
        return 0.0
    try:
        import sympy

        var = metadata["variable"]
        x = sympy.Symbol(var)
        user_expr = sympy.parse_expr(answer, local_dict={var: x, "C": sympy.Symbol("C")})
        derivative = sympy.diff(user_expr, x)
        integrand = sympy.parse_expr(metadata["integrand"], local_dict={var: x})
        if sympy.simplify(derivative - integrand) == 0:
            return 1.0
    except Exception as exc:
        raise_if_timeout(exc)
        pass
    return 0.0


def _score_tower_of_hanoi(answer: str, metadata: dict) -> float:
    """Validate Tower of Hanoi move sequence."""
    if not isinstance(answer, str) or not answer.strip():
        return 0.0

    moves = [line.strip() for line in answer.strip().splitlines() if line.strip()]
    num_disks = metadata["num_disks"]
    num_pegs = metadata["num_pegs"]
    start_peg = metadata["start_peg"]
    target_peg = metadata["target_peg"]

    peg_state = {peg: [] for peg in range(1, num_pegs + 1)}
    for disk in range(num_disks, 0, -1):
        peg_state[start_peg].append(disk)

    for move in moves:
        try:
            disk, from_peg, to_peg = _parse_hanoi_move(move)
        except Exception as exc:
            raise_if_timeout(exc)
            return 0.0

        if not _validate_hanoi_move(peg_state, disk, from_peg, to_peg):
            return 0.0

        peg_state[from_peg].pop()
        peg_state[to_peg].append(disk)

    expected_final = list(range(num_disks, 0, -1))
    if peg_state[target_peg] != expected_final:
        return 0.05

    optimal_moves = metadata.get("solution_length", len(moves))
    user_moves = len(moves)
    if user_moves <= optimal_moves:
        return 1.0
    return optimal_moves / user_moves


def _parse_hanoi_move(move: str) -> tuple[int, int, int]:
    pattern = r"Move disk (\d+) from Peg (\d+) to Peg (\d+)"
    m = re.search(pattern, move)
    if not m:
        raise ValueError(f"Unexpected move format: '{move}'")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _validate_hanoi_move(
    peg_state: dict[int, list[int]], disk: int, from_peg: int, to_peg: int
) -> bool:
    if not peg_state.get(from_peg) or peg_state[from_peg][-1] != disk:
        return False
    if peg_state.get(to_peg) and peg_state[to_peg][-1] < disk:
        return False
    return True


def _score_palindrome_generation(answer: str, metadata: dict) -> float:
    """Validate palindrome with correct letters."""
    if not isinstance(answer, str) or not answer.strip():
        return 0.0
    answer = answer.strip().lower()
    expected_letters = metadata["letters"]

    if answer != answer[::-1]:
        return 0.02
    if sorted(answer) != sorted(expected_letters):
        return 0.05
    return 1.0


def _score_power_function(answer: str, metadata: dict) -> float:
    """Compare answers to 3 significant figures."""
    oracle_answer = metadata.get("ground_truth", "")
    if not answer:
        return 0.0
    try:
        user_answer = _format_sig_figs(Decimal(answer), 3)
        oracle = _format_sig_figs(Decimal(oracle_answer), 3)
        return 1.0 if user_answer == oracle else 0.01
    except Exception as exc:
        raise_if_timeout(exc)
        return 0.01


def _format_sig_figs(x: Decimal, sig: int) -> Decimal:
    if x.is_zero():
        return Decimal("0." + "0" * (sig - 1))
    exp = x.adjusted()
    shift = sig - exp - 1
    rounded = x.quantize(Decimal("1e{}".format(-shift)))
    return Decimal(rounded)


def _score_shortest_path(answer: str, metadata: dict) -> float:
    """Validate shortest path in matrix."""
    if not isinstance(answer, str) or not answer.strip():
        return 0.0

    oracle_answer = metadata.get("ground_truth", "").strip()
    answer = answer.strip()

    if answer == oracle_answer:
        return 1.0

    matrix = metadata.get("matrix", [])
    answer_path = answer.split()
    oracle_path = oracle_answer.split()

    if _is_valid_path(matrix, answer_path):
        if len(answer_path) == len(oracle_path):
            return 1.0
        return 0.5

    return 0.0


def _is_valid_path(matrix: list[list[str]], path: list[str]) -> bool:
    ROWS, COLS = len(matrix), len(matrix[0])
    DIRS = {"right": (0, 1), "down": (1, 0), "left": (0, -1), "up": (-1, 0)}

    start_r = start_c = end_r = end_c = None
    for r in range(ROWS):
        for c in range(COLS):
            if matrix[r][c] == "*":
                start_r, start_c = r, c
            elif matrix[r][c] == "#":
                end_r, end_c = r, c

    if start_r is None or end_r is None:
        return False

    r, c = start_r, start_c
    for direction in path:
        if direction not in DIRS:
            return False
        dr, dc = DIRS[direction]
        r, c = r + dr, c + dc
        if not (0 <= r < ROWS and 0 <= c < COLS):
            return False
        if matrix[r][c] == "X":
            return False

    return (r, c) == (end_r, end_c)


def _score_string_insertion(answer: str, metadata: dict) -> float:
    """Direct match with eval fallback for list-of-char format."""
    oracle_answer = metadata.get("ground_truth", "")
    if isinstance(answer, str):
        if answer == oracle_answer:
            return 1.0
        try:
            evaluated = "".join(eval(answer))
            if evaluated == oracle_answer:
                return 0.1
        except Exception as exc:
            raise_if_timeout(exc)
            pass
    return 0.0


def _score_manipulate_matrix(answer: str, metadata: dict) -> float:
    """Matrix match with format normalization."""
    oracle_answer = metadata.get("ground_truth", "").strip()
    if not answer or not answer.strip():
        return 0.0
    answer = answer.strip()
    if answer == oracle_answer:
        return 1.0

    # Normalize whitespace in matrix rows
    try:
        normalized = "\n".join(
            " ".join(row.strip().split()) for row in answer.strip().split("\n")
        ).strip()
        if normalized == oracle_answer:
            return 1.0
        if oracle_answer in normalized:
            return len(oracle_answer) / len(normalized)
    except Exception as exc:
        raise_if_timeout(exc)
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Task dispatcher
# ---------------------------------------------------------------------------

_TASK_SCORERS: dict[str, Any] = {
    "needle_haystack": _score_needle_haystack,
    "cryptarithm": _score_cryptarithm,
    "simple_integration": _score_simple_integration,
    "tower_of_hanoi": _score_tower_of_hanoi,
    "palindrome_generation": _score_palindrome_generation,
    "power_function": _score_power_function,
    "shortest_path": _score_shortest_path,
    "string_insertion": _score_string_insertion,
    "manipulate_matrix": _score_manipulate_matrix,
}


def _score_answer(answer: str, label: str, metadata: dict) -> float:
    """Score an answer using task-specific or cascade scoring."""
    if not answer:
        return 0.0

    dataset = metadata.get("source_dataset", "")
    scorer = _TASK_SCORERS.get(dataset)

    if scorer is not None:
        try:
            return float(scorer(answer, metadata))
        except Exception as e:
            raise_if_timeout(e)
            logger.debug(f"Task-specific scorer for {dataset} failed: {e}, falling back to cascade")
            return _cascade_score(answer, label)

    return _cascade_score(answer, label)


# ---------------------------------------------------------------------------
# Bounded scoring subprocesses
# ---------------------------------------------------------------------------

_SCORING_SEMAPHORES = WeakKeyDictionary()


class RewardWorkerError(RuntimeError):
    """A scoring subprocess failed without producing a usable reward."""


async def _score_answer_in_subprocess(answer: str, label: str, metadata: dict) -> float:
    """Keep synchronous verifiers off the event loop and enforce a hard timeout.

    A thread/executor timeout cannot stop SymPy. Each subprocess handles one
    answer, and is killed and reaped on timeout or caller cancellation.
    Queueing time is covered by joint_reward's outer sample timeout.
    """
    timeout_s = float(os.environ.get("REASONING_GYM_REWARD_TIMEOUT_S", "10"))
    workers = int(os.environ.get("REASONING_GYM_REWARD_WORKERS", "8"))
    if not math.isfinite(timeout_s) or timeout_s <= 0 or workers <= 0:
        raise ValueError("ReasoningGym reward timeout and worker count must be positive")
    loop = asyncio.get_running_loop()
    if loop not in _SCORING_SEMAPHORES:
        _SCORING_SEMAPHORES[loop] = asyncio.Semaphore(workers)

    payload = json.dumps({"answer": answer, "label": label, "metadata": metadata}).encode()
    async with _SCORING_SEMAPHORES[loop]:
        child_env = os.environ.copy()
        child_env.update(
            CUDA_VISIBLE_DEVICES="",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
        )
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(__file__).with_name("reasoning_gym_reward_worker.py")),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=child_env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(payload), timeout=timeout_s)
        except Exception as exc:
            raise_if_timeout(exc)
            raise
        finally:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.communicate()

        if process.returncode:
            raise RewardWorkerError(
                f"Scoring worker exited with {process.returncode}: {stderr.decode(errors='replace')[-2000:]}"
            )
        try:
            result = json.loads(stdout)
            if not isinstance(result, dict):
                raise TypeError("Scoring worker returned a non-object result")
            if result.get("timeout") is True:
                raise RewardTimeout(str(result.get("error", "Scoring worker timed out")))
            reward = float(result["reward"])
            if not math.isfinite(reward):
                raise ValueError("non-finite reward")
        except (ValueError, TypeError, KeyError) as exc:
            raise_if_timeout(exc)
            raise RewardWorkerError("Scoring worker returned an invalid reward") from exc
        return reward


# ---------------------------------------------------------------------------
# Main reward entrypoint
# ---------------------------------------------------------------------------


async def reasoning_gym_reward(args, sample: Sample) -> float:
    """Slime-compatible reward function for Nemotron-RL-ReasoningGym.

    Extracts the model's final answer from <answer> tags (or after
    chain-of-thought), then scores it against the ground truth label.

    Uses task-specific verifiers from reasoning-gym where available,
    and a cascade scoring fallback (string → float → math match) otherwise.

    Args:
        args: Training args (unused).
        sample: Sample with .response, .label, and .metadata containing
                "source_dataset" and "ground_truth".

    Returns:
        float: Reward in [0.0, 1.0].
    """
    label = _resolve_label(sample)
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}

    answer = _extract_answer(sample.response)
    if not answer:
        return 0.0

    try:
        reward = await _score_answer_in_subprocess(answer, label, metadata)
    except (Exception, RewardTimeout) as exc:
        timed_out = is_timeout_error(exc)
        if not timed_out and not isinstance(exc, RewardWorkerError):
            raise
        # Match joint_reward's invalid-sample contract: exclude the sample
        # from NGRPO statistics and zero its training loss mask.
        mark_invalid_reward(sample, timeout=timed_out, reason=str(exc))
        metadata = sample.metadata
        logger.warning(
            "ReasoningGym scoring %s; marking sample invalid: index=%s dataset=%s timeout=%ss error=%s",
            "timed out" if timed_out else "failed",
            getattr(sample, "index", None),
            metadata.get("source_dataset"),
            os.environ.get("REASONING_GYM_REWARD_TIMEOUT_S", "10"),
            exc,
        )
        reward = 0.0

    if reward < 0.01:
        logger.debug(
            f"reasoning_gym reward=0: dataset={metadata.get('source_dataset')}, "
            f"label={label[:80]!r}, answer={answer[:80]!r}"
        )

    _append_reward_log(sample.response or "", label, answer, reward, metadata)
    return reward


custom_rm = reasoning_gym_reward
