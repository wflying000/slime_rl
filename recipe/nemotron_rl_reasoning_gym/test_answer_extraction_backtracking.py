"""Regression tests for long whitespace in generated answer tags."""
import random
import re
import time

from recipe.nemotron_rl_reasoning_gym.reasoning_gym_reward import (
    _extract_answer, _extract_tagged_answer, _normalize_extracted_answer,
)


def test_long_internal_whitespace_is_linear():
    response = "<answer>x" + " \t\r\n" * 10000 + "y</answer>"
    start = time.monotonic()
    assert _extract_answer(response) == "x" + " \t\r\n" * 10000 + "y"
    assert time.monotonic() - start < 1


def test_missing_closing_tag_with_long_whitespace():
    response = "<answer>" + " " * 40000 + "SOLN" + "\t\r\n" * 15000
    assert _extract_answer(response) == "SOLN"


def test_tag_delimiters_match_legacy_short_inputs():
    legacy = re.compile(r"(?is)<answer>\s*(.*?)\s*(?:</answer>)?\s*(?:<\|im_end\|>|<\|endoftext\|>|$)")
    rng = random.Random(41)
    atoms = ["a", "b", " ", "\t", "\n", "<answer>", "</answer>", "<ANSWER>", "<|endoftext|>", "<|im_end|>"]
    for _ in range(2000):
        text = "".join(rng.choice(atoms) for _ in range(rng.randint(1, 10)))
        match = legacy.search(text)
        expected = match.group(1).strip() if match else None
        assert _extract_tagged_answer(text) == expected


def test_normalization_preserves_legacy_short_inputs():
    rng = random.Random(23)
    def legacy(answer):
        answer = answer.strip()
        answer = answer.replace("<|im_end|>", "").replace("<|endoftext|>", "").replace("</s>", "").strip()
        answer = re.sub(r"[ \t]*[*_`]+[ \t]*$", "", answer)
        answer = re.sub(r"(?<=\S)[.。,;:!]+$", "", answer)
        return answer.strip()
    for _ in range(2000):
        answer = "".join(rng.choice("ab \t\n.*_`!,:。") for _ in range(rng.randint(1, 20)))
        assert _normalize_extracted_answer(answer) == legacy(answer)
