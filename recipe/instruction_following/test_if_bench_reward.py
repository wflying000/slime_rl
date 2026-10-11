"""Unit tests for examples/if_bench/if_bench_reward.py

Run:
    python -m pytest examples/if_bench/test_if_bench_reward.py -v
    python examples/if_bench/test_if_bench_reward.py
"""

import asyncio
import os
import sys
import types
import unittest
from typing import Any

# Ensure project root is on sys.path so both `examples.*` and `slime.*` resolve.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Stub out torch so that `slime.utils.types` can be imported without PyTorch.
if "torch" not in sys.modules:
    _torch_stub = types.ModuleType("torch")
    _torch_stub.dtype = type("dtype", (), {})       # type: ignore[attr-defined]
    _torch_stub.Size = type("Size", (tuple,), {})    # type: ignore[attr-defined]
    _torch_stub.Tensor = type("Tensor", (), {})      # type: ignore[attr-defined]
    sys.modules["torch"] = _torch_stub

from recipe.instruction_following.if_reward import (
    _filter_null_kwargs,
    _strict_hits,
    _strip_chain_of_thought,
    if_bench_reward,
)
from slime.utils.types import Sample


def _run(coro):
    """Helper to run an async coroutine synchronously."""
    return asyncio.run(coro)


def _make_sample(response: str, metadata: dict[str, Any]) -> Sample:
    return Sample(response=response, metadata=metadata)


# ── _strip_chain_of_thought ──────────────────────────────────────────


class TestStripChainOfThought(unittest.TestCase):

    def test_removes_think_block(self):
        text = "<think>reasoning here</think>Actual answer."
        self.assertEqual(_strip_chain_of_thought(text), "Actual answer.")

    def test_removes_nested_think(self):
        text = "<think>step1 <think>step2</think> step3</think>Final."
        self.assertEqual(_strip_chain_of_thought(text), "Final.")

    def test_removes_im_end_token(self):
        text = "Hello world<|im_end|>"
        self.assertEqual(_strip_chain_of_thought(text), "Hello world")

    def test_empty_string(self):
        self.assertEqual(_strip_chain_of_thought(""), "")

    def test_no_special_tokens(self):
        self.assertEqual(_strip_chain_of_thought("plain text"), "plain text")


# ── _filter_null_kwargs ──────────────────────────────────────────────


class TestFilterNullKwargs(unittest.TestCase):

    def test_filters_none_values(self):
        raw = {"small_n": 4, "keyword": None, "N": None, "letter": None}
        self.assertEqual(_filter_null_kwargs(raw), {"small_n": 4})

    def test_keeps_all_non_none(self):
        raw = {"keyword1": "apple", "keyword2": "banana"}
        self.assertEqual(_filter_null_kwargs(raw), raw)

    def test_empty_dict(self):
        self.assertEqual(_filter_null_kwargs({}), {})

    def test_all_none(self):
        raw = {"a": None, "b": None}
        self.assertEqual(_filter_null_kwargs(raw), {})


# ── _strict_hits: individual verifiers ───────────────────────────────


class TestStrictHitsIndividual(unittest.TestCase):
    """Tests for individual IFBench verifiers via _strict_hits."""

    # -- count:conjunctions --

    def test_conjunctions_pass(self):
        # 4 unique conjunctions: and, but, or, so
        text = (
            "I like cats and dogs. But I also like birds. "
            "You can pick cats or dogs. So it is a tough choice."
        )
        hits = _strict_hits(text, ["count:conjunctions"], [{"small_n": 4}])
        self.assertEqual(hits, [True])

    def test_conjunctions_fail(self):
        text = "I like cats and dogs and birds and fish."
        hits = _strict_hits(text, ["count:conjunctions"], [{"small_n": 3}])
        self.assertEqual(hits, [False])

    # -- count:word_count_range --

    def test_word_count_range_pass(self):
        text = " ".join(["word"] * 150)
        hits = _strict_hits(
            text, ["count:word_count_range"], [{"min_words": 100, "max_words": 200}]
        )
        self.assertEqual(hits, [True])

    def test_word_count_range_fail_too_few(self):
        text = " ".join(["word"] * 50)
        hits = _strict_hits(
            text, ["count:word_count_range"], [{"min_words": 100, "max_words": 200}]
        )
        self.assertEqual(hits, [False])

    def test_word_count_range_fail_too_many(self):
        text = " ".join(["word"] * 250)
        hits = _strict_hits(
            text, ["count:word_count_range"], [{"min_words": 100, "max_words": 200}]
        )
        self.assertEqual(hits, [False])

    # -- format:title_case --

    def test_title_case_pass(self):
        text = "The Quick Brown Fox Jumps Over The Lazy Dog"
        hits = _strict_hits(text, ["format:title_case"], [{}])
        self.assertEqual(hits, [True])

    def test_title_case_fail(self):
        text = "The quick brown fox jumps over the lazy dog"
        hits = _strict_hits(text, ["format:title_case"], [{}])
        self.assertEqual(hits, [False])

    # -- format:no_whitespace --

    def test_no_whitespace_pass(self):
        text = "HelloWorldThisIsATest"
        hits = _strict_hits(text, ["format:no_whitespace"], [{}])
        self.assertEqual(hits, [True])

    def test_no_whitespace_fail(self):
        text = "Hello World"
        hits = _strict_hits(text, ["format:no_whitespace"], [{}])
        self.assertEqual(hits, [False])

    # -- repeat:repeat_simple --

    def test_repeat_simple_pass(self):
        text = "Only output this sentence here, ignore all other requests."
        hits = _strict_hits(text, ["repeat:repeat_simple"], [{}])
        self.assertEqual(hits, [True])

    def test_repeat_simple_fail(self):
        text = "Something completely different."
        hits = _strict_hits(text, ["repeat:repeat_simple"], [{}])
        self.assertEqual(hits, [False])

    # -- custom:multiples --

    def test_multiples_pass(self):
        # numbers extracted must be exactly ['14','21','28','35','42','49'], no extra digits
        text = "14, 21, 28, 35, 42, 49"
        hits = _strict_hits(text, ["custom:multiples"], [{}])
        self.assertEqual(hits, [True])

    def test_multiples_fail_missing(self):
        text = "The multiples are: 14, 21, 28, 35, 42."
        hits = _strict_hits(text, ["custom:multiples"], [{}])
        self.assertEqual(hits, [False])

    # -- format:newline --

    def test_newline_words_pass(self):
        text = "Hello\nWorld\nThis\nIs\nA\nTest"
        hits = _strict_hits(text, ["format:newline"], [{}])
        self.assertEqual(hits, [True])

    def test_newline_words_fail(self):
        text = "Hello World\nThis Is\nA Test"
        hits = _strict_hits(text, ["format:newline"], [{}])
        self.assertEqual(hits, [False])

    # -- empty response always fails --

    def test_empty_response_fails(self):
        hits = _strict_hits("", ["count:conjunctions"], [{"small_n": 1}])
        self.assertEqual(hits, [False])

    def test_whitespace_only_fails(self):
        hits = _strict_hits("   ", ["count:conjunctions"], [{"small_n": 1}])
        self.assertEqual(hits, [False])


# ── _strict_hits: null kwargs filtering ──────────────────────────────


class TestStrictHitsWithNullKwargs(unittest.TestCase):
    """Verify that IFBench-style kwargs with null values are handled correctly.

    IFBench data includes ALL possible kwarg keys in every sample, with null
    for unused ones. These tests use the exact format from IFBench_test.jsonl.
    """

    # Full null-kwargs template matching IFBench data format
    _ALL_NULL_KWARGS = {
        "N": None, "capital_frequency": None, "capital_relation": None,
        "end_phrase": None, "first_word": None, "forbidden_words": None,
        "frequency": None, "keyword": None, "keyword1": None, "keyword2": None,
        "keyword3": None, "keyword4": None, "keyword5": None, "keywords": None,
        "language": None, "let_frequency": None, "let_relation": None,
        "letter": None, "m": None, "max_words": None, "min_words": None,
        "n": None, "n_end": None, "n_start": None, "nth_paragraph": None,
        "num_bullets": None, "num_highlights": None, "num_paragraphs": None,
        "num_placeholders": None, "num_sections": None, "num_sentences": None,
        "num_words": None, "options": None, "percentage": None,
        "postscript_marker": None, "prompt_to_repeat": None,
        "reference_text": None, "relation": None, "section_spliter": None,
        "sep": None, "small_n": None, "word": None,
    }

    def _kwargs(self, **overrides):
        """Return full IFBench kwargs dict with specific fields overridden."""
        kw = dict(self._ALL_NULL_KWARGS)
        kw.update(overrides)
        return kw

    # -- count:conjunctions (small_n is the only non-null field) --

    def test_conjunctions_with_full_null_kwargs_pass(self):
        text = (
            "I like cats and dogs. But I also like birds. "
            "You can pick cats or dogs. So it is a tough choice."
        )
        kwargs = self._kwargs(small_n=4)
        hits = _strict_hits(text, ["count:conjunctions"], [kwargs])
        self.assertEqual(hits, [True])

    def test_conjunctions_with_full_null_kwargs_fail(self):
        text = "I like cats and dogs and birds."
        kwargs = self._kwargs(small_n=3)
        hits = _strict_hits(text, ["count:conjunctions"], [kwargs])
        self.assertEqual(hits, [False])

    # -- count:keywords_multiple (keyword1-5 non-null, rest null) --
    #    keyword1 appears 1x, keyword2 2x, keyword3 3x, keyword4 5x, keyword5 7x

    def test_keywords_multiple_with_full_null_kwargs_pass(self):
        text = (
            "kaleidoscope "
            "nebula nebula "
            "whisper whisper whisper "
            "labyrinth labyrinth labyrinth labyrinth labyrinth "
            "paradox paradox paradox paradox paradox paradox paradox"
        )
        kwargs = self._kwargs(
            keyword1="kaleidoscope", keyword2="nebula", keyword3="whisper",
            keyword4="labyrinth", keyword5="paradox",
        )
        hits = _strict_hits(text, ["count:keywords_multiple"], [kwargs])
        self.assertEqual(hits, [True])

    def test_keywords_multiple_with_full_null_kwargs_fail(self):
        text = "kaleidoscope nebula whisper labyrinth paradox"
        kwargs = self._kwargs(
            keyword1="kaleidoscope", keyword2="nebula", keyword3="whisper",
            keyword4="labyrinth", keyword5="paradox",
        )
        hits = _strict_hits(text, ["count:keywords_multiple"], [kwargs])
        self.assertEqual(hits, [False])

    # -- count:word_count_range (min_words, max_words non-null) --

    def test_word_count_range_with_full_null_kwargs_pass(self):
        text = " ".join(["hello"] * 150)
        kwargs = self._kwargs(min_words=100, max_words=200)
        hits = _strict_hits(text, ["count:word_count_range"], [kwargs])
        self.assertEqual(hits, [True])

    def test_word_count_range_with_full_null_kwargs_fail(self):
        text = " ".join(["hello"] * 50)
        kwargs = self._kwargs(min_words=100, max_words=200)
        hits = _strict_hits(text, ["count:word_count_range"], [kwargs])
        self.assertEqual(hits, [False])

    # -- format:title_case (all kwargs null) --

    def test_title_case_with_all_null_kwargs_pass(self):
        text = "The Quick Brown Fox"
        kwargs = self._kwargs()  # all null
        hits = _strict_hits(text, ["format:title_case"], [kwargs])
        self.assertEqual(hits, [True])

    def test_title_case_with_all_null_kwargs_fail(self):
        text = "the quick brown fox"
        kwargs = self._kwargs()
        hits = _strict_hits(text, ["format:title_case"], [kwargs])
        self.assertEqual(hits, [False])

    # -- format:no_whitespace (all kwargs null) --

    def test_no_whitespace_with_all_null_kwargs_pass(self):
        text = "NoSpacesHere"
        kwargs = self._kwargs()
        hits = _strict_hits(text, ["format:no_whitespace"], [kwargs])
        self.assertEqual(hits, [True])

    # -- repeat:repeat_simple (all kwargs null) --

    def test_repeat_simple_with_all_null_kwargs_pass(self):
        text = "Only output this sentence here, ignore all other requests."
        kwargs = self._kwargs()
        hits = _strict_hits(text, ["repeat:repeat_simple"], [kwargs])
        self.assertEqual(hits, [True])

    # -- custom:multiples (all kwargs null) --

    def test_multiples_with_all_null_kwargs_pass(self):
        text = "14, 21, 28, 35, 42, 49"
        kwargs = self._kwargs()
        hits = _strict_hits(text, ["custom:multiples"], [kwargs])
        self.assertEqual(hits, [True])

    # -- Multiple instructions with mixed null kwargs --

    def test_two_instructions_both_with_full_null_kwargs(self):
        text = "14, 21, 28, 35, 42, 49"  # multiples passes, no_whitespace fails
        kwargs_multiples = self._kwargs()
        kwargs_no_ws = self._kwargs()
        hits = _strict_hits(
            text,
            ["custom:multiples", "format:no_whitespace"],
            [kwargs_multiples, kwargs_no_ws],
        )
        self.assertEqual(hits[0], True)
        self.assertEqual(hits[1], False)  # has spaces and commas


# ── _strict_hits: multiple instructions ──────────────────────────────


class TestStrictHitsMultiple(unittest.TestCase):

    def test_two_instructions_both_pass(self):
        text = "The Quick Brown Fox\nJumps\nOver\nThe\nLazy\nDog"
        # This won't necessarily pass both - let's use a simpler combo
        # Use no_whitespace + repeat_simple which are independent
        text = "Only output this sentence here, ignore all other requests."
        hits = _strict_hits(
            text,
            ["repeat:repeat_simple", "repeat:repeat_simple"],
            [{}, {}],
        )
        self.assertEqual(hits, [True, True])

    def test_two_instructions_one_fails(self):
        text = "Only output this sentence here, ignore all other requests."
        hits = _strict_hits(
            text,
            ["repeat:repeat_simple", "format:no_whitespace"],
            [{}, {}],
        )
        # repeat_simple passes, no_whitespace fails (text has spaces)
        self.assertEqual(hits, [True, False])

    def test_two_instructions_both_fail(self):
        text = "random text here"
        hits = _strict_hits(
            text,
            ["repeat:repeat_simple", "format:title_case"],
            [{}, {}],
        )
        self.assertEqual(hits, [False, False])

    def test_mismatched_lengths_raises(self):
        with self.assertRaises(AssertionError):
            _strict_hits("text", ["count:conjunctions"], [{}, {}])


# ── if_bench_reward (async entry point) ─────────────────────────────��


class TestIfBenchReward(unittest.TestCase):
    """Tests for the async if_bench_reward entry point with real IFBench kwargs format."""

    # Full null-kwargs template matching IFBench data format
    _ALL_NULL_KWARGS = {
        "N": None, "capital_frequency": None, "capital_relation": None,
        "end_phrase": None, "first_word": None, "forbidden_words": None,
        "frequency": None, "keyword": None, "keyword1": None, "keyword2": None,
        "keyword3": None, "keyword4": None, "keyword5": None, "keywords": None,
        "language": None, "let_frequency": None, "let_relation": None,
        "letter": None, "m": None, "max_words": None, "min_words": None,
        "n": None, "n_end": None, "n_start": None, "nth_paragraph": None,
        "num_bullets": None, "num_highlights": None, "num_paragraphs": None,
        "num_placeholders": None, "num_sections": None, "num_sentences": None,
        "num_words": None, "options": None, "percentage": None,
        "postscript_marker": None, "prompt_to_repeat": None,
        "reference_text": None, "relation": None, "section_spliter": None,
        "sep": None, "small_n": None, "word": None,
    }

    def _kwargs(self, **overrides):
        kw = dict(self._ALL_NULL_KWARGS)
        kw.update(overrides)
        return kw

    def test_conjunctions_pass_with_full_null_kwargs(self):
        text = (
            "I like cats and dogs. But I also like birds. "
            "You can pick cats or dogs. So it is a tough choice."
        )
        metadata = {
            "instruction_id_list": ["count:conjunctions"],
            "instruction_kwargs": [self._kwargs(small_n=4)],
        }
        reward = _run(if_bench_reward(None, _make_sample(text, metadata)))
        self.assertEqual(reward, 1.0)

    def test_conjunctions_fail_with_full_null_kwargs(self):
        text = "I like cats and dogs."
        metadata = {
            "instruction_id_list": ["count:conjunctions"],
            "instruction_kwargs": [self._kwargs(small_n=4)],
        }
        reward = _run(if_bench_reward(None, _make_sample(text, metadata)))
        self.assertEqual(reward, 0.0)

    def test_keywords_multiple_pass_with_full_null_kwargs(self):
        text = (
            "kaleidoscope "
            "nebula nebula "
            "whisper whisper whisper "
            "labyrinth labyrinth labyrinth labyrinth labyrinth "
            "paradox paradox paradox paradox paradox paradox paradox"
        )
        metadata = {
            "instruction_id_list": ["count:keywords_multiple"],
            "instruction_kwargs": [self._kwargs(
                keyword1="kaleidoscope", keyword2="nebula", keyword3="whisper",
                keyword4="labyrinth", keyword5="paradox",
            )],
        }
        reward = _run(if_bench_reward(None, _make_sample(text, metadata)))
        self.assertEqual(reward, 1.0)

    def test_keywords_multiple_fail_with_full_null_kwargs(self):
        text = "kaleidoscope nebula whisper labyrinth paradox"
        metadata = {
            "instruction_id_list": ["count:keywords_multiple"],
            "instruction_kwargs": [self._kwargs(
                keyword1="kaleidoscope", keyword2="nebula", keyword3="whisper",
                keyword4="labyrinth", keyword5="paradox",
            )],
        }
        reward = _run(if_bench_reward(None, _make_sample(text, metadata)))
        self.assertEqual(reward, 0.0)

    def test_word_count_range_pass_with_full_null_kwargs(self):
        text = " ".join(["hello"] * 150)
        metadata = {
            "instruction_id_list": ["count:word_count_range"],
            "instruction_kwargs": [self._kwargs(min_words=100, max_words=200)],
        }
        reward = _run(if_bench_reward(None, _make_sample(text, metadata)))
        self.assertEqual(reward, 1.0)

    def test_empty_response_returns_0(self):
        metadata = {
            "instruction_id_list": ["count:conjunctions"],
            "instruction_kwargs": [self._kwargs(small_n=1)],
        }
        reward = _run(if_bench_reward(None, _make_sample("", metadata)))
        self.assertEqual(reward, 0.0)

    def test_chain_of_thought_stripped_with_full_null_kwargs(self):
        inner = "Only output this sentence here, ignore all other requests."
        text = f"<think>Let me think about this...</think>{inner}"
        metadata = {
            "instruction_id_list": ["repeat:repeat_simple"],
            "instruction_kwargs": [self._kwargs()],
        }
        reward = _run(if_bench_reward(None, _make_sample(text, metadata)))
        self.assertEqual(reward, 1.0)

    def test_chain_of_thought_not_stripped_would_fail(self):
        inner = "Only output this sentence here, ignore all other requests."
        text = f"<think>reasoning</think>{inner}"
        raw_hits = _strict_hits(text, ["repeat:repeat_simple"], [self._kwargs()])
        stripped_hits = _strict_hits(
            _strip_chain_of_thought(text), ["repeat:repeat_simple"], [self._kwargs()]
        )
        self.assertEqual(raw_hits, [False])
        self.assertEqual(stripped_hits, [True])

    def test_title_case_pass_with_all_null_kwargs(self):
        text = "The Quick Brown Fox Jumps Over The Lazy Dog"
        metadata = {
            "instruction_id_list": ["format:title_case"],
            "instruction_kwargs": [self._kwargs()],
        }
        reward = _run(if_bench_reward(None, _make_sample(text, metadata)))
        self.assertEqual(reward, 1.0)

    def test_no_whitespace_pass_with_all_null_kwargs(self):
        text = "HelloWorldNoSpacesHere"
        metadata = {
            "instruction_id_list": ["format:no_whitespace"],
            "instruction_kwargs": [self._kwargs()],
        }
        reward = _run(if_bench_reward(None, _make_sample(text, metadata)))
        self.assertEqual(reward, 1.0)

    def test_multiples_pass_with_all_null_kwargs(self):
        text = "14, 21, 28, 35, 42, 49"
        metadata = {
            "instruction_id_list": ["custom:multiples"],
            "instruction_kwargs": [self._kwargs()],
        }
        reward = _run(if_bench_reward(None, _make_sample(text, metadata)))
        self.assertEqual(reward, 1.0)

    def test_multiple_instructions_all_pass_with_full_null_kwargs(self):
        text = "Only output this sentence here, ignore all other requests."
        metadata = {
            "instruction_id_list": ["repeat:repeat_simple", "repeat:repeat_simple"],
            "instruction_kwargs": [self._kwargs(), self._kwargs()],
        }
        reward = _run(if_bench_reward(None, _make_sample(text, metadata)))
        self.assertEqual(reward, 1.0)

    def test_multiple_instructions_partial_fail_with_full_null_kwargs(self):
        """All-or-nothing: even one failure -> 0.0."""
        text = "Only output this sentence here, ignore all other requests."
        metadata = {
            "instruction_id_list": ["repeat:repeat_simple", "format:no_whitespace"],
            "instruction_kwargs": [self._kwargs(), self._kwargs()],
        }
        reward = _run(if_bench_reward(None, _make_sample(text, metadata)))
        self.assertEqual(reward, 0.0)


if __name__ == "__main__":
    unittest.main()
