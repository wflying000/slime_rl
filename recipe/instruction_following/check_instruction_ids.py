"""
检测训练数据中的 instruction_id 是否在各个 registry 中。

支持三个 registry：
  - IFBench:              examples/if_bench/instructions_registry.py
  - instruction_following: examples/instruction_following/instructions_registry.py  (Google IFEval 格式)
  - 两者的并集

用法:
    python examples/if_bench/check_instruction_ids.py <data.jsonl>
"""

import json
import sys
from collections import Counter

# ── IFBench registry (from examples/if_bench/instructions_registry.py) ──
IFBENCH_IDS = {
    "count:word_count_range", "count:unique_word_count", "ratio:stop_words",
    "ratio:sentence_type", "ratio:sentence_balance", "count:conjunctions",
    "count:person_names", "ratio:overlap", "count:numbers", "words:alphabet",
    "words:vowel", "words:consonants", "sentence:alliteration_increment",
    "words:palindrome", "count:punctuation", "format:parentheses",
    "format:quotes", "words:prime_lengths", "format:options", "format:newline",
    "format:emoji", "ratio:sentence_words", "count:words_japanese",
    "words:start_verb", "words:repeats", "sentence:keyword", "count:pronouns",
    "words:odd_even_syllables", "words:last_first", "words:paragraph_last_first",
    "sentence:increment", "words:no_consecutive", "format:line_indent",
    "format:quote_unquote", "format:list", "format:thesis", "format:sub-bullets",
    "format:no_bullets_bullets", "custom:multiples", "custom:mcq_count_length",
    "custom:reverse_newline", "custom:word_reverse", "custom:character_reverse",
    "custom:sentence_alphabet", "custom:european_capitals_sort", "custom:csv_city",
    "custom:csv_special_character", "custom:csv_quotes", "custom:date_format_list",
    "count:keywords_multiple", "words:keywords_specific_position",
    "words:words_position", "repeat:repeat_change", "repeat:repeat_simple",
    "repeat:repeat_span", "format:title_case", "format:output_template",
    "format:no_whitespace",
}

# ── instruction_following registry (from examples/instruction_following/instructions_registry.py) ──
# 与 Google IFEval 格式一致
INSTRUCTION_FOLLOWING_IDS = {
    "keywords:existence", "keywords:frequency", "keywords:forbidden_words",
    "keywords:letter_frequency", "language:response_language",
    "length_constraints:number_sentences", "length_constraints:number_paragraphs",
    "length_constraints:number_words", "length_constraints:nth_paragraph_first_word",
    "detectable_content:number_placeholders", "detectable_content:postscript",
    "detectable_format:number_bullet_lists", "detectable_format:constrained_response",
    "detectable_format:number_highlighted_sections", "detectable_format:multiple_sections",
    "detectable_format:json_format", "detectable_format:title",
    "combination:two_responses", "combination:repeat_prompt",
    "startend:end_checker", "change_case:capital_word_frequency",
    "change_case:english_capital", "change_case:english_lowercase",
    "punctuation:no_comma", "startend:quotation",
}

ALL_KNOWN_IDS = IFBENCH_IDS | INSTRUCTION_FOLLOWING_IDS


def check_data(filepath: str):
    ifbench_count = Counter()
    if_count = Counter()  # instruction_following
    unknown_count = Counter()

    total_samples = 0
    ifbench_only_samples = 0
    if_only_samples = 0
    mixed_samples = 0
    unknown_samples = 0

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ids = obj.get("metadata", {}).get("instruction_id_list", [])
            total_samples += 1

            sample_sources = set()
            for inst_id in ids:
                if inst_id in IFBENCH_IDS:
                    ifbench_count[inst_id] += 1
                    sample_sources.add("ifbench")
                elif inst_id in INSTRUCTION_FOLLOWING_IDS:
                    if_count[inst_id] += 1
                    sample_sources.add("instruction_following")
                else:
                    unknown_count[inst_id] += 1
                    sample_sources.add("unknown")

            if sample_sources == {"ifbench"}:
                ifbench_only_samples += 1
            elif sample_sources == {"instruction_following"}:
                if_only_samples += 1
            elif len(sample_sources) > 1 and "unknown" not in sample_sources:
                mixed_samples += 1
            else:
                unknown_samples += 1

    # ── Report ──
    print(f"{'='*70}")
    print(f"  Data file: {filepath}")
    print(f"  Total samples: {total_samples}")
    print(f"{'='*70}")

    print(f"\n{'─'*70}")
    print(f"  Sample distribution by source")
    print(f"{'─'*70}")
    print(f"  IFBench only (examples/if_bench):               {ifbench_only_samples:>7d}")
    print(f"  instruction_following only (examples/inst...):   {if_only_samples:>7d}")
    print(f"  Mixed (both benchmarks):                         {mixed_samples:>7d}")
    print(f"  Contains unknown IDs:                            {unknown_samples:>7d}")

    if ifbench_count:
        print(f"\n{'─'*70}")
        print(f"  IFBench instruction IDs  ({sum(ifbench_count.values())} occurrences, {len(ifbench_count)} unique)")
        print(f"{'─'*70}")
        for k, v in ifbench_count.most_common():
            print(f"    {k:50s} {v:>6d}")

    if if_count:
        print(f"\n{'─'*70}")
        print(f"  instruction_following IDs  ({sum(if_count.values())} occurrences, {len(if_count)} unique)")
        print(f"{'─'*70}")
        for k, v in if_count.most_common():
            print(f"    {k:50s} {v:>6d}")

    if unknown_count:
        print(f"\n{'─'*70}")
        print(f"  UNKNOWN instruction IDs  ({sum(unknown_count.values())} occurrences, {len(unknown_count)} unique)")
        print(f"{'─'*70}")
        for k, v in unknown_count.most_common():
            print(f"    {k:50s} {v:>6d}")

    # ── Summary ──
    print(f"\n{'='*70}")
    if not if_count and not unknown_count:
        print("  RESULT: All instruction IDs belong to IFBench. No issues.")
    elif not ifbench_count and not unknown_count:
        print("  RESULT: All instruction IDs belong to instruction_following. No issues.")
    else:
        parts = []
        if ifbench_count:
            parts.append(f"{sum(ifbench_count.values())} IFBench")
        if if_count:
            parts.append(f"{sum(if_count.values())} instruction_following")
        if unknown_count:
            parts.append(f"{sum(unknown_count.values())} UNKNOWN")
        print(f"  RESULT: Mixed data — {', '.join(parts)} occurrences.")
        if unknown_count:
            print(f"  WARNING: {len(unknown_count)} instruction IDs not found in ANY registry!")
    print(f"{'='*70}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <data.jsonl>")
        sys.exit(1)
    check_data(sys.argv[1])
