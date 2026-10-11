"""Reward function for ZebraLogicBench RL training.

Scores model responses against ground-truth grid solutions using cell-level
accuracy (matching evalscope's ``process_results`` logic). A puzzle is fully
solved only when every cell matches; partial credit is the fraction of
correctly predicted cells.

For MC-mode samples, uses exact string match against the answer.

Usage:
    --custom-rm-path examples.zebralogicbench.zebralogicbench_reward.zebralogicbench_reward
"""

from __future__ import annotations

from recipe.rl_reward.timeout_utils import raise_if_timeout

import json
import logging
import os
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# Set ZEBRALOGICBENCH_REWARD_LOG_PATH to enable per-sample reward logging.
REWARD_LOG_PATH: str | None = os.environ.get("ZEBRALOGICBENCH_REWARD_LOG_PATH") or None


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
        raise_if_timeout(exc)
        logger.warning(f"failed to append zebralogicbench reward log: {exc}")


# ---------------------------------------------------------------------------
# Answer extraction — find the last complete JSON object in model output
# ---------------------------------------------------------------------------


def _extract_last_complete_json(s: str) -> dict | None:
    """Extract the last complete JSON object from model output.

    Uses a stack-based parser to handle nested braces, then validates
    with json.loads. Returns None if no valid JSON found.
    """
    stack: list[int] = []
    last_json_start: int | None = None
    last_json_str: str | None = None

    for i, char in enumerate(s):
        if char == "{":
            if not stack:
                last_json_start = i
            stack.append(i)
        elif char == "}":
            if stack:
                stack.pop()
                if not stack and last_json_start is not None:
                    candidate = s[last_json_start : i + 1]
                    last_json_str = candidate
                    last_json_start = None

    if last_json_str:
        try:
            return json.loads(last_json_str)
        except json.JSONDecodeError as exc:
            raise_if_timeout(exc)
            return None
    return None


# ---------------------------------------------------------------------------
# Grid-mode scoring (matches evalscope process_results)
# ---------------------------------------------------------------------------


def _score_grid(response: str, ground_truth: dict) -> float:
    """Score a grid-mode puzzle response.

    Compares the predicted solution table against the reference table
    cell-by-cell. Returns the fraction of correctly predicted cells.

    A cell is considered correct when the prediction matches the ground
    truth case-insensitively after stripping whitespace. Missing cells
    or None values count as incorrect.
    """
    prediction_obj = _extract_last_complete_json(response)
    if prediction_obj is None:
        return 0.0

    reference_obj = ground_truth
    num_houses = len(reference_obj["rows"])
    columns = reference_obj["header"]
    total_cells = (len(columns) - 1) * num_houses

    # Build reference table: {house_name: {column: value}}
    reference_table: dict[str, dict[str, str]] = {}
    for i in range(num_houses):
        reference_table[f"House {i + 1}"] = {
            columns[j]: reference_obj["rows"][i][j] for j in range(1, len(columns))
        }

    prediction_table = prediction_obj.get("solution", None)
    if not prediction_table:
        return 0.0

    correct_cells = 0
    for house, col_map in reference_table.items():
        for column, truth in col_map.items():
            if house not in prediction_table or column not in prediction_table[house]:
                continue

            truth_cell = truth.lower().strip()
            pred_val = prediction_table[house][column]
            if pred_val is None:
                continue

            if isinstance(pred_val, list):
                if not pred_val:
                    continue
                predicted_cell = str(pred_val[0]).lower().strip()
            elif isinstance(pred_val, str):
                predicted_cell = pred_val.lower().strip()
            else:
                predicted_cell = str(pred_val).lower().strip()

            if truth_cell == predicted_cell:
                correct_cells += 1

    return correct_cells / total_cells if total_cells > 0 else 0.0


# ---------------------------------------------------------------------------
# MC-mode scoring
# ---------------------------------------------------------------------------


def _score_mc(response: str, ground_truth: str) -> float:
    """Score an MC-mode response using exact match."""
    # Try extracting JSON answer first
    prediction_obj = _extract_last_complete_json(response)
    if prediction_obj is not None:
        pred_str = str(prediction_obj).lower().strip()
        if pred_str == ground_truth.lower().strip():
            return 1.0

    # Fallback: check if ground_truth appears in response
    gt = ground_truth.lower().strip()
    resp_lower = response.lower()
    if gt in resp_lower:
        return 1.0

    return 0.0


# ---------------------------------------------------------------------------
# Resolve label
# ---------------------------------------------------------------------------


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
# Main reward entrypoint
# ---------------------------------------------------------------------------


async def zebralogicbench_reward(args: Any, sample: Sample, **kwargs) -> float:
    """Slime-compatible reward function for ZebraLogicBench.

    Dispatches to grid-mode or MC-mode scoring based on metadata["mode"].

    Args:
        args: Training args (unused).
        sample: Sample with .response, .label, and .metadata containing
                "ground_truth", "mode", and optionally "size".

    Returns:
        float: Reward in [0.0, 1.0].
    """
    del args, kwargs

    label = _resolve_label(sample)
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    mode = metadata.get("mode", "grid")
    response = sample.response

    if not response:
        return 0.0

    try:
        if mode == "mc":
            reward = _score_mc(response, str(label))
        else:
            # Grid mode: label is a JSON string of the solution
            if isinstance(label, str):
                ground_truth = json.loads(label)
            else:
                ground_truth = label
            reward = _score_grid(response, ground_truth)
    except Exception as e:
        raise_if_timeout(e)
        logger.debug(f"zebralogicbench_reward error: {e}")
        reward = 0.0

    if reward < 0.01:
        logger.debug(
            f"zebralogicbench reward=0: mode={mode}, "
            f"label={str(label)[:80]!r}, response={response[:80]!r}"
        )

    _append_reward_log(response, label, reward, metadata)
    return reward


custom_rm = zebralogicbench_reward
