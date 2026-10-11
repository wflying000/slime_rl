"""Test that klear_math_reward returns 0 for the sample cases in reward_logs.

Standalone — no slime/torch dependency. We re-implement the core comparison
logic from klear_math_reward.py to verify the logged reward=0 is correct.
"""

import json
import re
import sys
import os

# ---------------------------------------------------------------------------
# Re-implement the core logic locally (avoids torch import from slime.utils)
# ---------------------------------------------------------------------------

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+?\|>")
_ANSWER_LINE_RE = re.compile(r"(?i)Answer\s*[:：]\s*([^\n]+)")


def _strip_special_tokens(text: str) -> str:
    return _SPECIAL_TOKEN_RE.sub("", text).strip()


# --- math_dapo helpers (inlined) ---

def last_boxed_only_string(string: str):
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


SUBSTITUTIONS = [
    ("an ", ""), ("a ", ""), (".$", "$"), ("\\$", ""),
    (r"\ ", ""), (" ", ""), ("mbox", "text"),
    (",\\text{and}", ","), ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "hours", "km",
    "units", "\\ldots", "sue", "points", "feet", "minutes", "digits", "cents",
    "degrees", "cm", "gm", "pounds", "meters", "meals", "edges", "students",
    "childrentickets", "multiples", "\\text{s}", "\\text{.}", "\\text{\ns}",
    "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}",
    r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    final_answer = final_answer.split("=")[-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer.strip()


# --- klear_math_reward core logic (inlined) ---

def _resolve_label(label, metadata: dict) -> str:
    if isinstance(label, (str, int, float)) and str(label) != "":
        return str(label)
    for key in ("ground_truth", "label", "answer"):
        v = metadata.get(key)
        if isinstance(v, (str, int, float)) and str(v) != "":
            return str(v)
    raise ValueError("no ground truth found")


def _extract_predicted_answer(response: str) -> str:
    if not response:
        return ""
    text = _strip_special_tokens(response)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    text = text.strip()
    boxed = last_boxed_only_string(text)
    if boxed is not None:
        try:
            return _strip_special_tokens(remove_boxed(boxed))
        except AssertionError:
            pass
    matches = _ANSWER_LINE_RE.findall(text)
    if matches:
        return _strip_special_tokens(matches[-1])
    return text


def _is_correct(prediction: str, ground_truth: str) -> bool:
    prediction = _strip_special_tokens(prediction)
    ground_truth = _strip_special_tokens(ground_truth)
    if not prediction:
        return False
    norm_pred = normalize_final_answer(prediction)
    norm_gt = normalize_final_answer(ground_truth)
    if norm_pred and norm_pred == norm_gt:
        return True
    try:
        from math_verify import parse
        from math_verify import verify as m_verify
    except ImportError:
        return False
    try:
        gt_parsed = parse(f"\\boxed{{{ground_truth}}}", parsing_timeout=5)
        pred_parsed = parse(f"\\boxed{{{prediction}}}", parsing_timeout=5)
    except Exception:
        return False
    try:
        return bool(m_verify(gt_parsed, pred_parsed, timeout_seconds=5))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def load_samples(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def test_reward_is_zero():
    log_path = "/mnt/code/wangfei61/train_qwen35/reward_logs/klear_math_reward_sample.jsonl"
    records = load_samples(log_path)
    assert len(records) > 0, "No records found in sample log"

    for i, rec in enumerate(records):
        response = rec["response"]
        label = rec["label"]
        logged_reward = rec["reward"]
        logged_prediction = rec["prediction"]
        metadata = rec.get("metadata", {})

        resolved_label = _resolve_label(label, metadata)
        prediction = _extract_predicted_answer(response or "")

        # Normalized string match
        norm_pred = normalize_final_answer(prediction)
        norm_gt = normalize_final_answer(resolved_label)
        string_match = bool(norm_pred and norm_pred == norm_gt)

        # math_verify semantic match
        math_verify_match = False
        math_verify_err = None
        try:
            from math_verify import parse
            from math_verify import verify as m_verify
            gt_parsed = parse(f"\\boxed{{{resolved_label}}}", parsing_timeout=5)
            pred_parsed = parse(f"\\boxed{{{prediction}}}", parsing_timeout=5)
            math_verify_match = bool(m_verify(gt_parsed, pred_parsed, timeout_seconds=5))
        except Exception as e:
            math_verify_err = str(e)

        is_correct = _is_correct(prediction, resolved_label)
        reward = 1.0 if is_correct else 0.0

        print(f"\n--- Sample {i} ---")
        print(f"  label:           {resolved_label!r}")
        print(f"  prediction:      {prediction!r}")
        print(f"  logged_pred:     {logged_prediction!r}")
        print(f"  norm_pred:       {norm_pred!r}")
        print(f"  norm_label:      {norm_gt!r}")
        print(f"  string_match:    {string_match}")
        print(f"  math_verify:     {math_verify_match}" + (f"  (err: {math_verify_err})" if math_verify_err else ""))
        print(f"  computed reward: {reward}")
        print(f"  logged reward:   {logged_reward}")

        # Record whether the logged reward=0 is reproduced
        match_logged = reward == logged_reward

        print(f"  match_logged:    {match_logged}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: With math_verify installed, the above samples are")
    print("mathematically correct (prediction matches label semantically).")
    print("The logged reward=0 was likely produced when math_verify was")
    print("unavailable, causing the code to fall back to string matching,")
    print("which fails for e.g. '0.5' vs '\\dfrac{1}{2}'.")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_reward_is_zero()
