"""Three-state content rule precheck for instruction-following rewards.

The rule precheck is deliberately three-state:

- ``pass_high_conf``: safe to reward without an LLM, currently limited to
  copy-only tasks whose hard constraints have already passed.
- ``fail_high_conf``: safe to reject without an LLM, for empty, repetitive,
  placeholder/refusal, or clearly malformed responses.
- ``unsure``: call the LLM judge. This includes most semantic content tasks,
  including format-heavy, math, table/SQL, translation, target-language, and
  structured-output prompts.

This keeps precision high by avoiding broad rule-only positives, while keeping
recall by sending low-confidence rule failures to the semantic judge.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from . import rule_content_features as _features

RULE_PASS_HIGH_CONF = "pass_high_conf"
RULE_FAIL_HIGH_CONF = "fail_high_conf"
RULE_UNSURE = "unsure"


@dataclass(frozen=True)
class RuleContentResult:
    """Structured output from the content precheck."""

    score: float | None
    verdict: str
    confidence: float
    reason: str
    task_type: str
    needs_llm: bool
    heuristic_score: float | None = None
    heuristic_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_HIGH_CONF_FAIL_REASONS = {
    "empty_response",
    "empty_content",
    "few_unique_tokens",
    "low_unigram_diversity",
    "low_trigram_diversity",
    "high_trigram_frequency",
    "high_bigram_frequency",
    "unexpected_non_english_script",
    "too_few_language_chars",
    "unnatural_token_shape",
    "refusal_or_placeholder",
    "generic_short_answer",
    "too_little_substance",
    "numeric_only_without_numeric_task",
}

_LOW_CONF_FAIL_PREFIXES = (
    "too_short:",
)

_LOW_CONF_FAIL_REASONS = {
    "low_prompt_relevance",
    "missing_code_content",
    "missing_math_content",
    "translation_too_short",
    "missing_option_answer",
}

_FORMAT_HEAVY_TASK_TYPES = {
    "format_heavy",
    "structured_output",
    "table_sql",
    "target_language",
    "translation",
    "math",
    "code",
    "repeat_then_answer",
}


def _filtered_kwargs(kwargs_list: list[dict[str, Any] | None] | None) -> list[dict[str, Any]]:
    return _features._normalize_kwargs_list(kwargs_list)


def _contains_prompt_pattern(pattern: str, instruction: str) -> bool:
    return bool(re.search(pattern, instruction.lower(), flags=re.IGNORECASE | re.DOTALL))


def classify_content_task(
    instruction: str,
    instruction_id_list: list[str] | None = None,
    kwargs_list: list[dict[str, Any] | None] | None = None,
) -> str:
    """Classify the semantic task shape for rule/judge routing."""
    instruction_id_list = instruction_id_list or []
    filtered_kwargs = _filtered_kwargs(kwargs_list)
    if _features._is_copy_only_task(instruction_id_list):
        return "copy_only"
    if "combination:repeat_prompt" in instruction_id_list:
        return "repeat_then_answer"
    if "language:response_language" in instruction_id_list:
        return "target_language"
    if _contains_prompt_pattern(r"\btranslate\b|translation|translate the", instruction):
        return "translation"
    if _contains_prompt_pattern(r"\bsql\b|table name:|\|---|\b(?:row|column|columns|table)\b", instruction):
        return "table_sql"
    if _contains_prompt_pattern(r"\b(?:python|javascript|java|code|function|class|debug|transcompile)\b", instruction):
        return "code"
    if _contains_prompt_pattern(
        r"\\\(|\\\[|\$|=|\b(?:calculate|solve|derive|equation|integral|derivative|probability|ratio|maximum|minimum)\b",
        instruction,
    ):
        return "math"
    if _features._is_structured_data_task(instruction_id_list, instruction):
        return "structured_output"
    if _features._is_short_answer_task(instruction_id_list, filtered_kwargs, instruction):
        return "short_answer"
    if any(inst_id in _features._FORMAT_HEAVY_INSTRUCTIONS for inst_id in instruction_id_list):
        return "format_heavy"
    if len(_features._tokenize_words(_features._remove_format_instruction_text(instruction))) <= 6:
        return "short_answer"
    return "general_generation"


def _is_high_conf_fail_reason(reason: str) -> bool:
    if reason in _HIGH_CONF_FAIL_REASONS:
        return True
    return any(reason.startswith(prefix) for prefix in _LOW_CONF_FAIL_PREFIXES) is False and reason.startswith("too_few")


def _is_low_conf_fail_reason(reason: str) -> bool:
    return reason in _LOW_CONF_FAIL_REASONS or any(reason.startswith(prefix) for prefix in _LOW_CONF_FAIL_PREFIXES)


def _safe_rule_pass(task_type: str) -> bool:
    return task_type == "copy_only"


def compute_rule_content_reward(
    response: str,
    instruction: str = "",
    instruction_id_list: list[str] | None = None,
    kwargs_list: list[dict[str, Any] | None] | None = None,
) -> RuleContentResult:
    """Return a high-confidence rule decision or ``unsure`` for LLM routing.

    The heuristic implementation is used as a feature extractor and
    degenerate-response detector. Broad heuristic passes are not treated as
    final positives for semantic tasks.
    """
    instruction_id_list = instruction_id_list or []
    task_type = classify_content_task(instruction, instruction_id_list, kwargs_list)

    if not response or not response.strip():
        return RuleContentResult(
            score=0.0,
            verdict=RULE_FAIL_HIGH_CONF,
            confidence=1.0,
            reason="empty_response",
            task_type=task_type,
            needs_llm=False,
            heuristic_score=0.0,
            heuristic_reason="empty_response",
        )

    heuristic_score, heuristic_reason = _features.compute_rule_content_reward(
        response,
        instruction,
        instruction_id_list,
        kwargs_list,
    )

    if heuristic_score >= 1.0:
        if _safe_rule_pass(task_type):
            return RuleContentResult(
                score=1.0,
                verdict=RULE_PASS_HIGH_CONF,
                confidence=0.98,
                reason=f"safe_rule_pass:{heuristic_reason}",
                task_type=task_type,
                needs_llm=False,
                heuristic_score=heuristic_score,
                heuristic_reason=heuristic_reason,
            )

        confidence = 0.55
        reason = f"rule_pass_requires_semantic_judge:{heuristic_reason}"
        if task_type in _FORMAT_HEAVY_TASK_TYPES:
            confidence = 0.45
            reason = f"format_or_semantic_task_requires_judge:{heuristic_reason}"

        return RuleContentResult(
            score=1.0,
            verdict=RULE_UNSURE,
            confidence=confidence,
            reason=reason,
            task_type=task_type,
            needs_llm=True,
            heuristic_score=heuristic_score,
            heuristic_reason=heuristic_reason,
        )

    if _is_high_conf_fail_reason(heuristic_reason):
        return RuleContentResult(
            score=0.0,
            verdict=RULE_FAIL_HIGH_CONF,
            confidence=0.95,
            reason=f"high_conf_rule_fail:{heuristic_reason}",
            task_type=task_type,
            needs_llm=False,
            heuristic_score=heuristic_score,
            heuristic_reason=heuristic_reason,
        )

    if _is_low_conf_fail_reason(heuristic_reason):
        return RuleContentResult(
            score=0.0,
            verdict=RULE_UNSURE,
            confidence=0.35,
            reason=f"low_conf_rule_fail_requires_judge:{heuristic_reason}",
            task_type=task_type,
            needs_llm=True,
            heuristic_score=heuristic_score,
            heuristic_reason=heuristic_reason,
        )

    return RuleContentResult(
        score=0.0,
        verdict=RULE_UNSURE,
        confidence=0.5,
        reason=f"rule_fail_requires_judge:{heuristic_reason}",
        task_type=task_type,
        needs_llm=True,
        heuristic_score=heuristic_score,
        heuristic_reason=heuristic_reason,
    )
