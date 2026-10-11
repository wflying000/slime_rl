"""Reward function for Guru-RL-92k RLVR training.

Dispatches reward computation according to ``sample.metadata["data_source"]``:

- ``logic__zebra_puzzle_dataset``: cell-level grid accuracy
- ``logic__graph_logical_dataset``: exact string match
- ``logic__ordering_puzzle_dataset``: strict list match or edit-distance partial credit
- ``stem_web``: LLM judge verification (requires STEM_LLM_JUDGE_URL env var)
- ``table__hitab`` / ``table__multihier``: numeric comparison with tolerance

Adapted from Reasoning360 verl/utils/reward_score/ implementations.

Usage:
    --custom-rm-path examples.guru_rl.guru_rl_reward.guru_rl_reward
"""

from __future__ import annotations

from recipe.rl_reward.timeout_utils import raise_if_timeout

import ast
import json
import logging
import os
import re
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# Set GURU_RL_REWARD_LOG_PATH to enable per-sample reward logging.
REWARD_LOG_PATH: str | None = os.environ.get("GURU_RL_REWARD_LOG_PATH") or None

# ---------------------------------------------------------------------------
# Reward logging
# ---------------------------------------------------------------------------


def _append_reward_log(
    response: str,
    label: Any,
    reward: float,
    metadata: dict[str, Any] | None,
) -> None:
    if REWARD_LOG_PATH is None:
        return
    record = {
        "reward": reward,
        "label": label,
        "response": response,
        "metadata": metadata or {},
    }
    try:
        os.makedirs(os.path.dirname(REWARD_LOG_PATH), exist_ok=True)
        with open(REWARD_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning(f"failed to append guru_rl reward log: {exc}")


# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _strip_chain_of_thought(text: str) -> str:
    """Remove <think>...</think> block and trailing special tokens."""
    if not text:
        return ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    text = text.replace("<|im_end|>", "")
    return text


def _extract_answer_tag(response: str) -> str | None:
    """Extract content from the last <answer>...</answer> tag."""
    matches = list(re.finditer(_ANSWER_TAG_RE, response))
    if matches:
        return matches[-1].group(1).strip()
    return None


def _resolve_label(sample: Sample) -> Any:
    """Resolve ground truth from sample.label or metadata."""
    if sample.label is not None and sample.label != "":
        return sample.label
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    gt = metadata.get("ground_truth")
    if gt is not None and gt != "":
        return gt
    raise ValueError("sample must provide a non-empty label via sample.label or sample.metadata['ground_truth']")


# ---------------------------------------------------------------------------
# Logic: Zebra Puzzle — cell-level grid accuracy
# ---------------------------------------------------------------------------


def _extract_zebra_solution(solution_str: str) -> dict | None:
    """Extract predicted grid arrangement from <answer> tags."""
    content = _extract_answer_tag(solution_str)
    if content is None:
        return None
    try:
        return ast.literal_eval(content)
    except (SyntaxError, ValueError):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None
    except Exception:
        return None


def _compute_zebra_accuracy(answer: dict, ground_truth: dict) -> float:
    """Compare grid cell-by-cell accuracy between predicted and ground truth."""
    if not isinstance(answer, dict):
        return 0.0
    num_rows = len(ground_truth["rows"])
    num_cols = len(ground_truth["header"])
    correct_cells = 0
    for i in range(num_rows):
        for j in range(num_cols):
            try:
                if answer["rows"][i][j] == ground_truth["rows"][i][j]:
                    correct_cells += 1
            except (IndexError, KeyError):
                pass
    return correct_cells / (num_rows * num_cols)


def _score_zebra_puzzle(response: str, label: Any, metadata: dict) -> float:
    predicted = _extract_zebra_solution(response)
    if predicted is None:
        return 0.0
    try:
        return _compute_zebra_accuracy(predicted, label)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Logic: Graph Logical — exact string match
# ---------------------------------------------------------------------------


def _extract_graph_solution(solution_str: str) -> str | None:
    """Extract single-word answer from <answer> tags."""
    content = _extract_answer_tag(solution_str)
    if content is None:
        return None
    if re.search(r"^[A-Za-z]+$", content):
        return content
    return None


def _score_graph_logical(response: str, label: Any, metadata: dict) -> float:
    ground_truth = str(label).lower()
    solution = _extract_graph_solution(response)
    if solution is None:
        return 0.0
    return 1.0 if solution.lower() == ground_truth else 0.0


# ---------------------------------------------------------------------------
# Logic: Ordering Puzzle — strict list match or edit-distance partial credit
# ---------------------------------------------------------------------------


def _extract_ordering_solution(solution_str: str) -> list | None:
    """Extract ordered list from <answer> tags."""
    content = _extract_answer_tag(solution_str)
    if content is None:
        return None
    # Try regex extraction of a list
    bird_pattern = r"\[[\s'\"]*([^\],]+)[\s'\"]*(?:,[\s'\"]*([^\],]+)[\s'\"]*)*\]"
    bird_match = re.search(bird_pattern, content)
    if bird_match:
        bird_list_str = bird_match.group(0)
        items = re.findall(r"['\"]?([\w\s]+)['\"]?", bird_list_str)
        items = [item.lower().strip() for item in items if item.strip() and item.strip() not in ["[", "]"]]
        return items
    # Fallback: try safe eval
    try:
        fixed = re.sub(r"(\[|\s*,\s*)(\w+)(\s*,|\])", r'\1"\2"\3', content)
        return eval(fixed)  # noqa: S307
    except Exception:
        return None


def _compute_edit_distance(list1: list, list2: list) -> int:
    """Compute edit distance between two lists."""
    m, n = len(list1), len(list2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if list1[i - 1] == list2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def _score_ordering_puzzle(response: str, label: Any, metadata: dict) -> float:
    target = label if isinstance(label, list) else list(label)
    predicted = _extract_ordering_solution(response)
    if predicted is None:
        return 0.0
    if not isinstance(predicted, list):
        return 0.0
    if predicted == target:
        return 1.0
    # Non-strict: partial credit via edit distance
    edit_distance = _compute_edit_distance(predicted, target)
    max_dist = max(len(predicted), len(target))
    return max(0.0, 1.0 - edit_distance / max_dist) if max_dist > 0 else 0.0


# ---------------------------------------------------------------------------
# STEM: Web — LLM judge verification
# ---------------------------------------------------------------------------


def _last_boxed_only_string(string: str) -> str | None:
    """Extract the last \\boxed{...} or \\fbox{...} content."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if left_brace_idx is None or right_brace_idx is None:
        return None
    return string[left_brace_idx + 1 : right_brace_idx].strip()


def _llm_judge(question: str, student: str, reference: str) -> float:
    """Call external LLM judge to verify answer equivalence."""
    import requests

    url_base = os.getenv("STEM_LLM_JUDGE_URL")
    if not url_base:
        raise EnvironmentError("STEM_LLM_JUDGE_URL not set")
    url = url_base.rstrip("/") + "/v1/chat/completions"

    prompt = (
        f"User: ### Question: {question}\n\n"
        f"### Ground Truth Answer: {reference}\n\n"
        f"### Student Answer: {student}\n\n"
        "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
        "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
        'If the student\'s answer is correct, output "Final Decision: Yes". '
        'If the student\'s answer is incorrect, output "Final Decision: No". Assistant:'
    )

    payload = {
        "model": "TIGER-Lab/general-verifier",
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.0,
    }

    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    return float("Final Decision: Yes" in text)


def _score_stem_web(response: str, label: Any, metadata: dict) -> float:
    """Score STEM web answer using LLM judge."""
    # Extract boxed answer
    solution_str = _strip_chain_of_thought(response)
    boxed = _last_boxed_only_string(solution_str)
    if boxed is None:
        return 0.0

    # Get question from extra_info
    extra_info = metadata.get("extra_info", {})
    question = extra_info.get("question", "")
    if not question:
        logger.warning("stem_web: no question found in metadata.extra_info")
        return 0.0

    try:
        return _llm_judge(question, boxed, str(label))
    except Exception as e:
        raise_if_timeout(e)
        logger.warning(f"stem_web LLM judge error: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Table: HiTab / MultiHier — numeric comparison with tolerance
# ---------------------------------------------------------------------------


def _remove_boxed(s: str) -> str:
    """Remove \\boxed{} wrapper from a string."""
    if s.startswith("\\boxed{") and s.endswith("}"):
        return s[len("\\boxed{") : -1]
    if s.startswith("\\boxed "):
        return s[len("\\boxed ") :]
    return s


def _drop_latex_text(answer: str) -> str:
    """Remove \\text{} from LaTeX answers like '20 \\text{to} 39'."""
    answer = re.sub(r"\\text\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\\\\", r"", answer)
    return answer


def _math_equal(a: float, b: float, tolerance: float = 1e-3) -> bool:
    """Compare two floats with tolerance."""
    if a == b:
        return True
    try:
        return abs(a - b) < tolerance * max(1, abs(a), abs(b))
    except Exception:
        return False


def _is_equiv(str1: str, str2: str) -> bool:
    """String equivalence after stripping whitespace and normalizing."""
    if str1 is None or str2 is None:
        return False
    return str1.strip() == str2.strip()


def _check_single_answer(answer: str, ground_truth: str) -> bool:
    """Check a single numeric answer against ground truth."""
    try:
        nanswer = answer.replace(",", "").replace("%", " / 100").replace("$", "").replace(":", "/").replace("\\", "")
        nanswer = float(eval(nanswer))  # noqa: S307
        try:
            ngt = ground_truth.replace(",", "").replace("%", " / 100").replace("$", "").replace(":", "/").replace("\\", "")
            ngt = float(eval(ngt))  # noqa: S307
            return _math_equal(nanswer, ngt)
        except Exception:
            return _math_equal(nanswer, float(ground_truth))
    except Exception:
        return _is_equiv(answer, ground_truth)


def _score_table(response: str, label: Any, metadata: dict) -> float:
    """Score table QA answer using numeric comparison with tolerance."""
    model_output = str(response).lower()
    ground_truth = str(label).lower()

    solution_str = model_output.split("</think>")[-1] if "</think>" in model_output else model_output

    # Try extracting \boxed{} answer
    boxed_str = _last_boxed_only_string(solution_str)
    if boxed_str is not None:
        answer = _remove_boxed(boxed_str)
        answer = _drop_latex_text(answer)
    else:
        answer = solution_str.strip()

    # Single or multiple answers
    if "|" not in ground_truth:
        return float(_check_single_answer(answer, ground_truth))

    # Multiple answers in format "ans1|ans2|ans3"
    try:
        gt_sorted = sorted([ans.strip() for ans in ground_truth.split("|")])
        ans_sorted = sorted([ans.strip() for ans in answer.split("|")])
        if len(gt_sorted) != len(ans_sorted):
            return 0.0
        for gt, res in zip(gt_sorted, ans_sorted):
            if not _check_single_answer(res, gt):
                return 0.0
        return 1.0
    except Exception as e:
        logger.debug(f"table multi-answer scoring error: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_TASK_SCORERS: dict[str, Any] = {
    "logic__zebra_puzzle_dataset": _score_zebra_puzzle,
    "logic__graph_logical_dataset": _score_graph_logical,
    "logic__ordering_puzzle_dataset": _score_ordering_puzzle,
    "stem_web": _score_stem_web,
    "table__hitab": _score_table,
    "table__multihier": _score_table,
}


def _score_answer(answer: str, label: Any, metadata: dict) -> float:
    """Score an answer using task-specific scoring based on data_source."""
    if not answer:
        return 0.0

    data_source = metadata.get("data_source", "")
    scorer = _TASK_SCORERS.get(data_source)

    if scorer is not None:
        try:
            return float(scorer(answer, label, metadata))
        except Exception as e:
            logger.debug(f"Task-specific scorer for {data_source} failed: {e}")
            return 0.0

    logger.warning(f"Unknown data_source: {data_source!r}, defaulting to string match")
    return 1.0 if str(answer).strip().lower() == str(label).strip().lower() else 0.0


# ---------------------------------------------------------------------------
# Main reward entrypoint
# ---------------------------------------------------------------------------


async def guru_rl_reward(args, sample: Sample, **kwargs) -> float:
    """Slime-compatible reward function for Guru-RL-92k.

    Extracts the model's final answer, then scores it against the ground
    truth label using task-specific verifiers dispatched by data_source.

    Args:
        args: Training args (unused).
        sample: Sample with .response, .label, and .metadata containing
                "data_source" and "ground_truth".

    Returns:
        float: Reward in [0.0, 1.0].
    """
    del args, kwargs

    label = _resolve_label(sample)
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}

    # For logic tasks, use full response (they extract from <answer> tags)
    # For stem/table, strip chain-of-thought first
    data_source = metadata.get("data_source", "")
    if data_source.startswith("logic__"):
        response = sample.response
    else:
        response = _strip_chain_of_thought(sample.response)

    reward = _score_answer(response, label, metadata)

    if reward < 0.01:
        logger.debug(
            f"guru_rl reward=0: data_source={data_source}, "
            f"label={str(label)[:80]!r}, response={response[:80]!r}"
        )

    _append_reward_log(sample.response or "", label, reward, metadata)
    return reward


custom_rm = guru_rl_reward
