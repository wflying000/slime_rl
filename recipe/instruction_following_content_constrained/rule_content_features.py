# coding: utf-8
"""Feature extraction and heuristics for content-constrained rewards."""

import collections
import math
import re
import unicodedata
from typing import Any


_COPY_ONLY_INSTRUCTIONS = {
    "copy:copy",
    "copy:copying_simple",
    "copy:copying_multiple",
    "new:copy_span_idx",
    "repeat:repeat_change",
    "repeat:repeat_simple",
    "repeat:repeat_span",
}

_SHORT_ANSWER_INSTRUCTIONS = {
    "custom:multiples",
    "detectable_format:constrained_response",
    "format:options",
}

_LOW_VERBOSITY_INSTRUCTIONS = {
    "count:words_japanese",
    "count:lowercase_counting",
    "detectable_format:bigram_wrapping",
    "detectable_format:sentence_hyphens",
    "detectable_format:square_brackets",
    "first_word:first_word_answer",
    "first_word:first_word_sent",
    "format:no_whitespace",
    "keywords:no_adjacent_consecutive",
    "last_word:last_word_answer",
    "last_word:last_word_sent",
    "letters:letter_counting",
    "letters:letter_counting2",
    "punctuation:no_comma",
    "punctuation:punctuation_dot",
    "startend:quotation",
}

_FORMAT_HEAVY_INSTRUCTIONS = {
    "count:words_japanese",
    "count:lowercase_counting",
    "custom:multiples",
    "detectable_content:number_placeholders",
    "detectable_content:postscript",
    "detectable_format:bigram_wrapping",
    "detectable_format:constrained_response",
    "detectable_format:json_format",
    "detectable_format:number_bullet_lists",
    "detectable_format:number_highlighted_sections",
    "detectable_format:sentence_hyphens",
    "detectable_format:square_brackets",
    "detectable_format:title",
    "first_word:first_word_answer",
    "first_word:first_word_sent",
    "format:list",
    "format:newline",
    "format:no_whitespace",
    "format:options",
    "keywords:letter_frequency",
    "last_word:last_word_answer",
    "last_word:last_word_sent",
    "letters:letter_counting",
    "letters:letter_counting2",
    "punctuation:no_comma",
    "punctuation:punctuation_dot",
    "startend:quotation",
}

_SKIP_PROMPT_RELEVANCE_INSTRUCTIONS = _FORMAT_HEAVY_INSTRUCTIONS | {
    "detectable_content:number_placeholders",
    "detectable_content:postscript",
}

_NUMERIC_ANSWER_INSTRUCTIONS = {
    "count:numbers",
    "custom:multiples",
}

_CODELIKE_INSTRUCTIONS = {
    "detectable_format:json_format",
    "custom:csv_city",
    "custom:csv_special_character",
    "custom:csv_quotes",
}

_WRAPPER_MARKER_PATTERNS = (
    (r"^```(?:json|jsonc|python|py|javascript|js|typescript|ts|sql|markdown|md)?\s*", ""),
    (r"\s*```$", ""),
    (r"^<<\s*", ""),
    (r"\s*>>$", ""),
)

_INLINE_STRIP_PATTERNS = (
    r"\*{1,6}",
)

_FORMAT_INSTRUCTION_PATTERNS = (
    r"\bIn your (?:entire )?response\b.*?(?:\.|\n|$)",
    r"\bYour (?:entire )?response\b.*?(?:\.|\n|$)",
    r"\bYour answer\b.*?(?:\.|\n|$)",
    r"\bAnswer with\b.*?(?:\.|\n|$)",
    r"\bInclude keyword\b.*?(?:\.|\n|$)",
    r"\bInclude keywords\b.*?(?:\.|\n|$)",
    r"\bDo not include keyword\b.*?(?:\.|\n|$)",
    r"\bThere should be\b.*?(?:\.|\n|$)",
    r"\bParagraphs\b.*?(?:\.|\n|$)",
    r"\bHighlight at least\b.*?(?:\.|\n|$)",
    r"\bFinish your response\b.*?(?:\.|\n|$)",
    r"\bStart and end your response\b.*?(?:\.|\n|$)",
    r"\bThe first word\b.*?(?:\.|\n|$)",
    r"\bThe last word\b.*?(?:\.|\n|$)",
    r"\bThe response must contain\b.*?(?:\.|\n|$)",
    r"\bWrap (?:your entire response|every word bigram)\b.*?(?:\.|\n|$)",
    r"\bAll sentences must be connected\b.*?(?:\.|\n|$)",
    r"\bEnclose every word\b.*?(?:\.|\n|$)",
    r"\bGive two different responses\b.*?(?:\.|\n|$)",
    r"\bFirst repeat the request\b.*?(?:\.|\n|$)",
    r"\bRepeat the request\b.*?(?:\.|\n|$)",
    r"\bRepeat the phrase\b.*?(?:\.|\n|$)",
    r"\bOnly use unique words\b.*?(?:\.|\n|$)",
    r"\bOnly output this sentence\b.*?(?:\.|\n|$)",
    r"\bNo two adjacent words\b.*?(?:\.|\n|$)",
    r"\bNo other words should follow\b.*?(?:\.|\n|$)",
    r"\bWrite 3 paragraphs\b.*?(?:\.|\n|$)",
)

_FORMAT_KEYWORDS = {
    "answer",
    "capital",
    "comma",
    "delimiter",
    "divider",
    "entire",
    "exact",
    "finish",
    "format",
    "highlight",
    "keyword",
    "letter",
    "line",
    "lowercase",
    "markdown",
    "paragraph",
    "phrase",
    "punctuation",
    "quote",
    "repeat",
    "response",
    "sentence",
    "uppercase",
    "word",
    "words",
}

_TASK_KEYWORDS = {
    "ad",
    "advise",
    "analyze",
    "answer",
    "calculate",
    "classify",
    "compare",
    "convert",
    "correct",
    "create",
    "define",
    "derive",
    "describe",
    "design",
    "determine",
    "develop",
    "draft",
    "elaborate",
    "explain",
    "find",
    "generate",
    "identify",
    "implement",
    "improve",
    "list",
    "name",
    "provide",
    "propose",
    "rewrite",
    "solve",
    "translate",
    "verify",
    "what",
    "when",
    "where",
    "which",
    "why",
    "write",
}

_PROMPT_STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "answer",
    "before",
    "being",
    "below",
    "could",
    "every",
    "following",
    "from",
    "have",
    "into",
    "must",
    "need",
    "only",
    "please",
    "provided",
    "question",
    "request",
    "response",
    "should",
    "that",
    "their",
    "there",
    "these",
    "this",
    "those",
    "through",
    "using",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "write",
    "your",
}

_CODE_CUES = {
    "class",
    "code",
    "csv",
    "css",
    "def",
    "django",
    "function",
    "html",
    "import",
    "json",
    "javascript",
    "program",
    "python",
    "return",
    "sql",
}

_MATH_CUES = {
    "area",
    "calculate",
    "derivative",
    "determine",
    "equation",
    "find",
    "function",
    "integral",
    "limit",
    "maximum",
    "minimum",
    "probability",
    "ratio",
    "solve",
}

_FILLER_PHRASES = (
    "as an ai",
    "i cannot",
    "i can't",
    "i am unable",
    "i'm unable",
    "cannot answer",
    "can't answer",
    "do not have enough",
    "does not exist",
    "unable to",
    "unfortunately",
    "not enough information",
    "no information",
    "lorem ipsum",
)

_GENERIC_SHORT_ANSWERS = {
    "impossible",
    "no",
    "none",
    "unknown",
    "yes",
}

_SCRIPT_HINTS = {
    "ar": ("arabic",),
    "fa": ("arabic",),
    "he": ("hebrew",),
    "hi": ("devanagari",),
    "ja": ("hiragana", "katakana", "cjk"),
    "ko": ("hangul",),
    "ru": ("cyrillic",),
    "uk": ("cyrillic",),
    "zh": ("cjk",),
}


def _filter_null_kwargs(kwargs: dict) -> dict:
    """Filter null values and normalize whole-number floats to ints."""
    result = {}
    for k, v in kwargs.items():
        if v is None:
            continue
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        result[k] = v
    return result


def _normalize_kwargs_list(kwargs_list: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [_filter_null_kwargs(kwargs or {}) for kwargs in kwargs_list or []]


def _char_script(char: str) -> str | None:
    if not char.isalpha():
        return None
    name = unicodedata.name(char, "")
    if "CJK UNIFIED" in name or "CJK COMPATIBILITY" in name:
        return "cjk"
    for script in (
        "HIRAGANA",
        "KATAKANA",
        "HANGUL",
        "CYRILLIC",
        "ARABIC",
        "HEBREW",
        "DEVANAGARI",
        "GREEK",
        "THAI",
        "TAMIL",
        "TELUGU",
        "SINHALA",
    ):
        if script in name:
            return script.lower()
    if "LATIN" in name:
        return "latin"
    return "other"


def _letter_chars(text: str) -> list[str]:
    return [ch for ch in text if ch.isalpha()]


def _script_counts(text: str) -> collections.Counter[str]:
    return collections.Counter(script for ch in text if (script := _char_script(ch)))


def _contains_cjk_or_compact_script(text: str) -> bool:
    scripts = _script_counts(text)
    return any(scripts.get(script, 0) for script in ("cjk", "hiragana", "katakana", "hangul", "thai"))


def _has_expected_language_script(text: str, kwargs_list: list[dict[str, Any]]) -> bool:
    scripts = _script_counts(text)
    if not scripts:
        return False
    for kwargs in kwargs_list:
        language = str(kwargs.get("language") or "").lower()
        if not language:
            continue
        expected = _SCRIPT_HINTS.get(language)
        if expected and any(scripts.get(script, 0) >= 2 for script in expected):
            return True
    return False


def _is_english_required_task(instruction_id_list: list[str], instruction: str) -> bool:
    lowered = instruction.lower()
    return (
        "change_case:english_capital" in instruction_id_list
        or "change_case:english_lowercase" in instruction_id_list
        or bool(re.search(r"\b(?:entire response|response) should be in english\b", lowered))
    )


def _violates_required_english(response: str, instruction_id_list: list[str], instruction: str) -> bool:
    if not _is_english_required_task(instruction_id_list, instruction):
        return False
    counts = _script_counts(response)
    non_latin = sum(count for script, count in counts.items() if script != "latin")
    latin = counts.get("latin", 0)
    return non_latin > max(2, latin)


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"\w+(?:[-']\w+)*|[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text.lower(), flags=re.UNICODE)


def _tokenize_alpha_words(text: str) -> list[str]:
    return re.findall(r"\w+(?:[-']\w+)*|[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text.lower(), flags=re.UNICODE)


def _strip_structural_markers(text: str) -> str:
    cleaned = text
    for pattern, replacement in _WRAPPER_MARKER_PATTERNS:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE | re.DOTALL)
    for pattern in _INLINE_STRIP_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.DOTALL)
    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _strip_copied_prompt_prefix(text: str, kwargs_list: list[dict[str, Any]]) -> str:
    stripped = text.strip()
    lowered = stripped.lower()
    for kwargs in kwargs_list:
        prompt_to_repeat = kwargs.get("prompt_to_repeat")
        if not prompt_to_repeat:
            continue
        prompt = str(prompt_to_repeat).strip()
        prompt_lower = prompt.lower()
        if lowered.startswith(prompt_lower):
            stripped = stripped[len(prompt):].strip()
            lowered = stripped.lower()
            break
    return stripped


def _remove_format_instruction_text(instruction: str) -> str:
    cleaned = instruction
    for pattern in _FORMAT_INSTRUCTION_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _extract_prompt_terms(instruction: str) -> set[str]:
    content_instruction = _remove_format_instruction_text(instruction)
    terms = set()
    for token in _tokenize_alpha_words(content_instruction):
        if len(token) <= 3:
            continue
        if token in _FORMAT_KEYWORDS:
            continue
        if token in _PROMPT_STOPWORDS:
            continue
        if token.isdigit():
            continue
        terms.add(token)
    return terms


def _extract_task_cues(instruction: str) -> set[str]:
    lowered = instruction.lower()
    cues = {cue for cue in _TASK_KEYWORDS if re.search(rf"\b{re.escape(cue)}\b", lowered)}
    if re.search(r"\bq(?:uestion)?\s*:", lowered) or "options:" in lowered:
        cues.add("answer")
    return cues


def _expected_min_words(
    instruction_id_list: list[str],
    kwargs_list: list[dict[str, Any]],
    instruction: str,
) -> int:
    min_words = 8
    has_exact_short_shape = False
    for inst_id, kwargs in zip(instruction_id_list, kwargs_list):
        if inst_id in _SHORT_ANSWER_INSTRUCTIONS:
            has_exact_short_shape = True
            min_words = min(min_words, 3)
        elif inst_id == "length_constraints:number_words":
            num_words = int(kwargs.get("num_words") or 0)
            relation = kwargs.get("relation")
            if relation == "at least":
                min_words = max(min_words, min(40, max(5, num_words // 3)))
            elif relation == "less than":
                min_words = min(min_words, max(3, min(12, num_words // 20)))
        elif inst_id == "count:word_count_range":
            min_words = max(min_words, min(40, int(kwargs.get("min_words") or 0)))
        elif inst_id == "length_constraints:number_sentences":
            num_sentences = int(kwargs.get("num_sentences") or 0)
            relation = kwargs.get("relation")
            if relation == "at least":
                min_words = max(min_words, min(60, num_sentences * 3))
            elif relation == "less than":
                min_words = min(min_words, max(3, num_sentences * 2))
        elif inst_id == "count:counting_composition":
            n_sent = int(kwargs.get("n_sent") or 0)
            n_words = int(kwargs.get("n_words") or 0)
            if n_sent and n_words:
                min_words = min(min_words, max(3, min(12, n_sent * n_words)))
        elif inst_id == "letters:letter_counting":
            letter_limit = int(kwargs.get("N") or 0)
            if kwargs.get("relation") == "less than" and letter_limit <= 5:
                min_words = 1
                has_exact_short_shape = True
        elif inst_id == "format:no_whitespace":
            min_words = min(min_words, 1)
        elif inst_id in _CODELIKE_INSTRUCTIONS:
            min_words = min(min_words, 2)
        elif inst_id in _LOW_VERBOSITY_INSTRUCTIONS:
            min_words = min(min_words, 4)

    content_instruction = _remove_format_instruction_text(instruction)
    if _contains_cjk_or_compact_script(content_instruction):
        min_words = min(min_words, 6)
    if any(inst_id in _FORMAT_HEAVY_INSTRUCTIONS for inst_id in instruction_id_list):
        min_words = min(min_words, 6)
    if not has_exact_short_shape and len(_tokenize_words(content_instruction)) >= 80:
        min_words = max(min_words, 10)
    return max(1, min_words)


def _is_copy_only_task(instruction_id_list: list[str]) -> bool:
    return any(inst_id in _COPY_ONLY_INSTRUCTIONS for inst_id in instruction_id_list) and not any(
        inst_id == "combination:repeat_prompt" for inst_id in instruction_id_list
    )


def _is_short_answer_task(
    instruction_id_list: list[str],
    kwargs_list: list[dict[str, Any]],
    instruction: str,
) -> bool:
    if any(inst_id in _SHORT_ANSWER_INSTRUCTIONS for inst_id in instruction_id_list):
        return True
    for inst_id, kwargs in zip(instruction_id_list, kwargs_list):
        if inst_id == "letters:letter_counting" and kwargs.get("relation") == "less than":
            return True
        if inst_id == "length_constraints:number_words" and kwargs.get("relation") == "less than":
            try:
                if int(kwargs.get("num_words") or 9999) <= 10:
                    return True
            except (TypeError, ValueError):
                pass
    if any(inst_id in _NUMERIC_ANSWER_INSTRUCTIONS for inst_id in instruction_id_list):
        return True
    content_instruction = _remove_format_instruction_text(instruction).lower()
    return bool(
        re.search(
            r"\b(?:answer|respond|reply)\s+(?:with|using|in)\s+(?:yes|no|true|false|one word|a single word)\b|"
            r"\b(?:yes/no|true/false|single word|one word)\b",
            content_instruction,
        )
    )


def _check_repetition(tokens: list[str], allow_short: bool) -> tuple[bool, str]:
    total_tokens = len(tokens)
    unique_tokens = len(set(tokens))
    if total_tokens == 0:
        return False, "empty_content"

    if allow_short and total_tokens <= 4:
        return True, "short_content_ok"

    if total_tokens >= 5 and unique_tokens <= 2:
        return False, "few_unique_tokens"

    if total_tokens < 15:
        if total_tokens >= 5 and unique_tokens / total_tokens < 0.6:
            return False, "low_unigram_diversity"
        return True, "short_diversity_ok"

    trigrams = ["_".join(tokens[i : i + 3]) for i in range(total_tokens - 2)]
    if trigrams:
        diversity_ratio = len(set(trigrams)) / len(trigrams)
        if total_tokens < 30:
            threshold = 0.50
        elif total_tokens < 100:
            threshold = 0.35
        else:
            threshold = 0.20
        if diversity_ratio < threshold:
            return False, "low_trigram_diversity"
        max_freq = max(collections.Counter(trigrams).values())
        if max_freq > max(6, math.ceil(total_tokens * 0.08)):
            return False, "high_trigram_frequency"

    bigrams = ["_".join(tokens[i : i + 2]) for i in range(total_tokens - 1)]
    if bigrams:
        max_bigram = max(collections.Counter(bigrams).values())
        if total_tokens >= 20 and max_bigram > max(8, math.ceil(total_tokens * 0.12)):
            return False, "high_bigram_frequency"

    return True, "diversity_ok"


def _should_skip_min_word_floor(
    content_text: str,
    instruction: str,
    instruction_id_list: list[str],
    allow_short: bool,
) -> bool:
    if allow_short:
        return False
    if _is_structured_data_task(instruction_id_list, instruction) and (
        _has_structured_shape(content_text) or _has_code_shape(content_text)
    ):
        return True
    if "detectable_content:number_placeholders" in instruction_id_list and _has_placeholder_shape(content_text):
        return True
    if "format:no_whitespace" in instruction_id_list and len(_letter_chars(content_text)) >= 8:
        return True
    if _contains_cjk_or_compact_script(content_text) and len(_letter_chars(content_text)) >= 12:
        return True
    return False


def _has_minimum_language_shape(
    text: str,
    tokens: list[str],
    allow_short: bool,
    instruction_id_list: list[str],
    kwargs_list: list[dict[str, Any]],
    instruction: str,
) -> tuple[bool, str]:
    if allow_short and len(tokens) <= 10:
        return True, "short_shape_ok"
    if _violates_required_english(text, instruction_id_list, instruction):
        return False, "unexpected_non_english_script"
    alpha_chars = len(_letter_chars(text))
    if alpha_chars < 3:
        return False, "too_few_language_chars"
    if len(tokens) >= 6:
        if _contains_cjk_or_compact_script(text) or _has_expected_language_script(text, kwargs_list):
            return True, "unicode_shape_ok"
        avg_token_len = sum(len(t) for t in tokens) / len(tokens)
        if avg_token_len < 1.5 and not any(inst_id in _FORMAT_HEAVY_INSTRUCTIONS for inst_id in instruction_id_list):
            return False, "unnatural_token_shape"
    return True, "language_shape_ok"


def _response_addresses_prompt(
    response: str,
    instruction: str,
    instruction_id_list: list[str],
    allow_short: bool,
) -> tuple[bool, str]:
    if not instruction:
        return True, "no_prompt_available"
    if allow_short:
        return True, "short_task_skip_prompt_overlap"
    if any(inst_id in _SKIP_PROMPT_RELEVANCE_INSTRUCTIONS for inst_id in instruction_id_list):
        return True, "format_task_skip_prompt_overlap"

    response_terms = set(_tokenize_alpha_words(response))
    prompt_terms = _extract_prompt_terms(instruction)
    if not prompt_terms:
        return True, "no_prompt_terms"

    overlap = response_terms & prompt_terms
    cue_overlap = overlap & _extract_task_cues(instruction)
    domain_overlap = {term for term in overlap if len(term) >= 5}

    if cue_overlap or len(domain_overlap) >= 2:
        return True, "prompt_overlap_ok"

    if "language:response_language" in instruction_id_list and len(_tokenize_words(response)) >= 8:
        return True, "language_task_skip_prompt_overlap"

    return False, "low_prompt_relevance"


def _is_structured_data_task(instruction_id_list: list[str], instruction: str) -> bool:
    lowered = instruction.lower()
    return any(inst_id in _CODELIKE_INSTRUCTIONS for inst_id in instruction_id_list) or bool(
        re.search(r"\b(?:json|csv|sql|table|array|list)\b", lowered)
    )


def _has_code_shape(response: str) -> bool:
    return bool(
        re.search(
            r"```|def\s+\w+\s*\(|function\s+\w+\s*\(|return\b|class\s+\w+|"
            r"\b(import|from|const|let|var|SELECT|WHERE|JOIN)\b|[{};]",
            response,
            flags=re.IGNORECASE,
        )
    )


def _has_structured_shape(response: str) -> bool:
    stripped = response.strip()
    if re.match(r"^```(?:json|jsonc)?", stripped, flags=re.IGNORECASE):
        return True
    if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
        return True
    if "\n" in stripped and ("," in stripped or "\t" in stripped):
        return True
    return False


def _has_math_shape(response: str) -> bool:
    lowered_response = response.lower()
    return bool(re.search(r"\d|=|<=|>=|\\frac|\\sqrt|therefore|so\b|answer|solution", lowered_response))


def _has_placeholder_shape(response: str) -> bool:
    return bool(re.search(r"\[[^\]\n]{2,80}\]", response))


def _looks_like_numeric_answer(response: str) -> bool:
    stripped = response.strip()
    if not stripped:
        return False
    return bool(re.fullmatch(r"[\s\d.,:/%+\-–—=()$]+", stripped))


def _is_generic_short_response(response: str, instruction_id_list: list[str], allow_short: bool) -> bool:
    if allow_short:
        return False
    normalized = " ".join(_tokenize_words(response))
    return normalized in _GENERIC_SHORT_ANSWERS


def _has_enough_substance(
    response: str,
    instruction: str,
    instruction_id_list: list[str],
    tokens: list[str],
    allow_short: bool,
    kwargs_list: list[dict[str, Any]],
) -> tuple[bool, str]:
    if allow_short:
        return True, "short_task_substance_ok"
    if _is_structured_data_task(instruction_id_list, instruction) and (
        _has_structured_shape(response) or _has_code_shape(response)
    ):
        return True, "structured_substance_ok"
    if "detectable_content:number_placeholders" in instruction_id_list and _has_placeholder_shape(response):
        return True, "placeholder_substance_ok"
    if "format:no_whitespace" in instruction_id_list and len(_letter_chars(response)) >= 8:
        return True, "no_whitespace_substance_ok"
    if _has_expected_language_script(response, kwargs_list) and len(_letter_chars(response)) >= 8:
        return True, "target_language_substance_ok"
    if _contains_cjk_or_compact_script(response) and len(_letter_chars(response)) >= 12:
        return True, "compact_script_substance_ok"
    if (
        any(inst_id in _FORMAT_HEAVY_INSTRUCTIONS for inst_id in instruction_id_list)
        and len(tokens) >= 4
        and len(_letter_chars(response)) >= 4
    ):
        return True, "format_heavy_substance_ok"
    if _is_generic_short_response(response, instruction_id_list, allow_short):
        return False, "generic_short_answer"
    if len(tokens) <= 3 and not _has_math_shape(response) and not _has_code_shape(response):
        return False, "too_little_substance"
    if _looks_like_numeric_answer(response) and not any(
        inst_id in _NUMERIC_ANSWER_INSTRUCTIONS or inst_id == "custom:multiples"
        for inst_id in instruction_id_list
    ):
        return False, "numeric_only_without_numeric_task"
    return True, "substance_ok"


def _check_task_specific_content(
    response: str,
    instruction: str,
    instruction_id_list: list[str],
    allow_short: bool,
    kwargs_list: list[dict[str, Any]],
) -> tuple[bool, str]:
    if allow_short:
        return True, "short_task_skip_task_shape"

    lowered_prompt = instruction.lower()
    lowered_response = response.lower()
    response_tokens = _tokenize_words(response)

    if any(phrase in lowered_response for phrase in _FILLER_PHRASES):
        return False, "refusal_or_placeholder"

    if any(inst_id in _FORMAT_HEAVY_INSTRUCTIONS for inst_id in instruction_id_list):
        return True, "format_heavy_task_shape_ok"

    if _is_structured_data_task(instruction_id_list, instruction) and _has_structured_shape(response):
        return True, "structured_task_shape_ok"

    if "detectable_content:number_placeholders" in instruction_id_list and _has_placeholder_shape(response):
        return True, "placeholder_task_shape_ok"

    if "language:response_language" in instruction_id_list and _has_expected_language_script(response, kwargs_list):
        return True, "target_language_task_shape_ok"

    if any(re.search(rf"\b{re.escape(cue)}\b", lowered_prompt) for cue in _CODE_CUES):
        code_hits = sum(1 for cue in _CODE_CUES if re.search(rf"\b{re.escape(cue)}\b", lowered_response))
        has_code_shape = _has_code_shape(response)
        if not has_code_shape and code_hits == 0 and len(response_tokens) < 20:
            return False, "missing_code_content"

    if any(re.search(rf"\b{re.escape(cue)}\b", lowered_prompt) for cue in _MATH_CUES) or re.search(r"\\\(|\\\[|\$|=", instruction):
        has_math_shape = _has_math_shape(response)
        if not has_math_shape and len(response_tokens) < 20:
            return False, "missing_math_content"

    if re.search(r"\btranslate\b|translation|translate the", lowered_prompt):
        if len(response_tokens) < 2:
            return False, "translation_too_short"

    if "options:" in lowered_prompt or re.search(r"\([a-e]\)", lowered_prompt):
        if not re.search(r"\([a-e]\)|\b[a-e]\b|answer", lowered_response):
            return False, "missing_option_answer"

    return True, "task_shape_ok"


def compute_rule_content_reward(
    response: str,
    instruction: str = "",
    instruction_id_list: list[str] | None = None,
    kwargs_list: list[dict[str, Any]] | None = None,
) -> tuple[float, str]:
    """Rule-based content reward after hard format constraints have passed.

    The checker rejects vacuous/repetitive format-only strings while relaxing
    length and diversity checks when the sample's own constraints require a
    short, copied, or tightly formatted answer.
    """
    instruction_id_list = instruction_id_list or []
    filtered_kwargs = _normalize_kwargs_list(kwargs_list)

    if not response or not response.strip():
        return 0.0, "empty_response"

    if _is_copy_only_task(instruction_id_list):
        return 1.0, "copy_only_rule_task"

    content_text = _strip_copied_prompt_prefix(response, filtered_kwargs)
    content_text = _strip_structural_markers(content_text)
    tokens = _tokenize_words(content_text)
    allow_short = _is_short_answer_task(instruction_id_list, filtered_kwargs, instruction)
    min_words = _expected_min_words(instruction_id_list, filtered_kwargs, instruction)
    if allow_short:
        min_words = min(min_words, 1)

    if len(tokens) < min_words and not _should_skip_min_word_floor(
        content_text,
        instruction,
        instruction_id_list,
        allow_short,
    ):
        return 0.0, f"too_short:{len(tokens)}<{min_words}"

    ok, reason = _has_minimum_language_shape(
        content_text,
        tokens,
        allow_short,
        instruction_id_list,
        filtered_kwargs,
        instruction,
    )
    if not ok:
        return 0.0, reason

    ok, reason = _check_repetition(tokens, allow_short)
    if not ok:
        return 0.0, reason

    ok, reason = _has_enough_substance(
        content_text,
        instruction,
        instruction_id_list,
        tokens,
        allow_short,
        filtered_kwargs,
    )
    if not ok:
        return 0.0, reason

    ok, reason = _check_task_specific_content(
        content_text,
        instruction,
        instruction_id_list,
        allow_short,
        filtered_kwargs,
    )
    if not ok:
        return 0.0, reason

    ok, reason = _response_addresses_prompt(content_text, instruction, instruction_id_list, allow_short)
    if not ok:
        return 0.0, reason

    return 1.0, "content_heuristic_ok"
