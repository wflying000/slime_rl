"""Reward for Nemotron structured-output RL data.

Recommended custom RM path:
    --custom-rm-path recipe.nemotron_structured_outputs.structured_outputs_reward.structured_outputs_reward

Environment variables:
    STRUCTURED_OUTPUTS_REWARD_LOG_PATH: JSONL reward log path. Empty disables logging.
"""

from __future__ import annotations

from recipe.rl_reward.timeout_utils import raise_if_timeout

import asyncio
import ast
import copy
import csv
import io
import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum
from typing import Any

import yaml
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError, ValidationError

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


logger = logging.getLogger(__name__)

REWARD_LOG_PATH: str | None = os.environ.get("STRUCTURED_OUTPUTS_REWARD_LOG_PATH") or None


class SchemaType(str, Enum):
    JSON = "json"
    YAML = "yaml"
    XML = "xml"
    TOML = "toml"
    CSV = "csv"


class UnsupportedSchemaTypeError(ValueError):
    pass


@dataclass(frozen=True)
class RewardResult:
    reward: float
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class TextCandidate:
    content: str
    start: int
    end: int
    source: str


def _short_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:200]}"


def _get_metadata(sample: Any) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        raise ValueError("sample.metadata must be a dict")
    return metadata


def _response_text(sample: Any) -> str:
    response = getattr(sample, "response", "")
    text = response if isinstance(response, str) else str(response or "")
    return _extract_response_content(text)


def _extract_response_content(text: str) -> str:
    """Keep generated content after </think> and before <|im_end|>."""
    if not text:
        return ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[0]
    return text.strip()


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:[A-Za-z0-9_+\-.]+)?\s*\n?(.*?)\n?```", stripped, flags=re.DOTALL)
    return match.group(1).strip() if match else stripped


def _iter_code_fences(text: str) -> list[tuple[str | None, str, int, int]]:
    fences: list[tuple[str | None, str, int, int]] = []
    for match in re.finditer(
        r"```(?P<label>[A-Za-z0-9_+\-.]+)?[^\S\r\n]*(?:\r?\n)?(?P<body>.*?)(?:\r?\n)?```",
        text,
        flags=re.DOTALL,
    ):
        label = match.group("label")
        fences.append((label.lower() if label else None, match.group("body").strip(), match.start(), match.end()))
    return fences


def _extract_required_code_fence(text: str, expected_label: str) -> tuple[str | None, RewardResult | None]:
    expected_label = expected_label.lower()
    stripped = text.strip()
    matches = [fence for fence in _iter_code_fences(stripped) if fence[0] == expected_label]
    if not matches:
        return None, RewardResult(
            0.0,
            "missing_code_fence",
            f"Expected exactly one fenced code block labelled {expected_label!r}",
        )
    if len(matches) > 1:
        return None, RewardResult(
            0.0,
            "multiple_code_fences",
            f"Expected exactly one fenced code block labelled {expected_label!r}, got {len(matches)}",
        )

    _, body, start, end = matches[0]
    outside = (stripped[:start] + stripped[end:]).strip()
    if outside:
        return None, RewardResult(
            0.0,
            "extra_text_outside_code_fence",
            f"Expected no text outside the {expected_label!r} fenced code block",
        )
    return body, None


def _expected_response_fence(metadata: dict[str, Any], schema_type: Any) -> str | None:
    fence = metadata.get("expected_response_fence")
    if fence:
        return str(fence).lower()
    if metadata.get("final_output_prompt_version") == "fenced_code_block_v1":
        if metadata.get("response_mode") == "tool_call":
            return "tool_call"
        return str(schema_type).lower()
    return None


def _dedupe_candidates(candidates: list[TextCandidate]) -> list[TextCandidate]:
    seen: set[str] = set()
    deduped: list[TextCandidate] = []
    for candidate in sorted(candidates, key=lambda item: (item.end, len(item.content)), reverse=True):
        content = candidate.content.strip()
        if not content or content in seen:
            continue
        seen.add(content)
        deduped.append(TextCandidate(content, candidate.start, candidate.end, candidate.source))
    return deduped


def _code_fence_candidates(text: str, labels: set[str | None]) -> list[TextCandidate]:
    candidates: list[TextCandidate] = []
    normalized_labels = {label.lower() if isinstance(label, str) else None for label in labels}
    for label, body, start, end in _iter_code_fences(text):
        if label in normalized_labels:
            candidates.append(TextCandidate(body, start, end, "code_fence"))
    return candidates


def _json_balanced_end(text: str, start: int) -> int | None:
    opening = text[start]
    closing = "}" if opening == "{" else "]"
    stack = [closing]
    in_string = False
    escape = False

    for index in range(start + 1, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or char != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return index + 1

    return None


def _json_candidates(text: str) -> list[TextCandidate]:
    candidates: list[TextCandidate] = []
    for match in re.finditer(r"[\{\[]", text):
        end = _json_balanced_end(text, match.start())
        if end is not None:
            candidates.append(TextCandidate(text[match.start() : end], match.start(), end, "json_balanced"))
    return candidates


def _xml_candidates(text: str) -> list[TextCandidate]:
    candidates: list[TextCandidate] = []
    start_tag_pattern = re.compile(r"<(?P<tag>[A-Za-z_][\w:.-]*)(?:\s[^<>]*)?>")
    for match in start_tag_pattern.finditer(text):
        start = match.start()
        tag = match.group("tag")
        if text[match.end() - 2 : match.end()] == "/>":
            candidate = text[start : match.end()]
            try:
                ET.fromstring(candidate)
            except ET.ParseError as exc:
                raise_if_timeout(exc)
                continue
            candidates.append(TextCandidate(candidate, start, match.end(), "xml_element"))
            continue

        close_pattern = re.compile(rf"</{re.escape(tag)}\s*>")
        for close_match in close_pattern.finditer(text, match.end()):
            end = close_match.end()
            candidate = text[start:end]
            try:
                ET.fromstring(candidate)
            except ET.ParseError as exc:
                raise_if_timeout(exc)
                continue
            candidates.append(TextCandidate(candidate, start, end, "xml_element"))
            break
    return candidates


def _line_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        end = offset + len(line)
        spans.append((offset, end, line))
        offset = end
    if text and (not spans or spans[-1][1] < len(text)):
        spans.append((offset, len(text), text[offset:]))
    return spans


def _line_block_candidates(
    text: str,
    *,
    start_pattern: re.Pattern[str],
    row_pattern: re.Pattern[str] | None = None,
    max_candidates: int = 200,
) -> list[TextCandidate]:
    spans = _line_spans(text)
    start_indexes = [
        index
        for index, (_, _, line) in enumerate(spans)
        if start_pattern.search(line.strip())
    ]
    candidates: list[TextCandidate] = []
    for end_index in range(len(spans) - 1, -1, -1):
        for start_index in start_indexes:
            if start_index > end_index:
                continue
            block_lines = [line for _, _, line in spans[start_index : end_index + 1]]
            if row_pattern is not None and not any(row_pattern.search(line.strip()) for line in block_lines):
                continue
            content = "".join(block_lines).strip()
            if not content:
                continue
            candidates.append(
                TextCandidate(
                    content,
                    spans[start_index][0],
                    spans[end_index][1],
                    "line_block",
                )
            )
            if len(candidates) >= max_candidates:
                return candidates
    return candidates


def _yaml_candidates(text: str) -> list[TextCandidate]:
    return _line_block_candidates(
        text,
        start_pattern=re.compile(r"""^(?:[-?]\s+)?[A-Za-z0-9_"'](?:[^:{}\n]{0,120})\s*:"""),
    )


def _toml_candidates(text: str) -> list[TextCandidate]:
    return _line_block_candidates(
        text,
        start_pattern=re.compile(r"^(?:[A-Za-z0-9_.-]+\s*=|\[[^\]]+\]\s*$)"),
    )


def _csv_candidates(text: str) -> list[TextCandidate]:
    return _line_block_candidates(
        text,
        start_pattern=re.compile(r"^[^,\n]+,[^,\n]+"),
        row_pattern=re.compile(r","),
    )


def _structured_candidates(schema_type: Any, response_text: str, expected_fence: str | None) -> list[TextCandidate]:
    try:
        normalized = schema_type if isinstance(schema_type, SchemaType) else SchemaType(str(schema_type).lower())
    except ValueError as exc:
        raise_if_timeout(exc)
        return []

    labels: set[str | None] = {None}
    if expected_fence:
        labels.add(expected_fence.lower())
    labels.add(normalized.value)

    candidates = _code_fence_candidates(response_text, labels)
    if normalized == SchemaType.JSON:
        candidates.extend(_json_candidates(response_text))
    elif normalized == SchemaType.YAML:
        candidates.extend(_yaml_candidates(response_text))
    elif normalized == SchemaType.XML:
        candidates.extend(_xml_candidates(response_text))
    elif normalized == SchemaType.TOML:
        candidates.extend(_toml_candidates(response_text))
    elif normalized == SchemaType.CSV:
        candidates.extend(_csv_candidates(response_text))
    return _dedupe_candidates(candidates)


def _tool_call_candidates(response_text: str, expected_fence: str | None) -> list[TextCandidate]:
    labels = {None, "tool_call", "json"}
    if expected_fence:
        labels.add(expected_fence.lower())
    candidates = _code_fence_candidates(response_text, labels)
    candidates.extend(
        TextCandidate(match.group(1).strip(), match.start(), match.end(), "tool_call_tag")
        for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", response_text, flags=re.DOTALL)
    )
    candidates.extend(_json_candidates(response_text))
    return _dedupe_candidates(candidates)


def strictify_schema(schema: Any) -> Any:
    """Match the reference server's strict OpenAPI-style behavior."""
    schema = copy.deepcopy(schema)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                node["required"] = list(properties)
                node["additionalProperties"] = False
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(schema)
    return schema


def _load_schema(schema_str: Any) -> tuple[dict[str, Any] | None, RewardResult | None]:
    if isinstance(schema_str, dict):
        schema = schema_str
    elif isinstance(schema_str, str):
        try:
            schema = json.loads(schema_str)
        except Exception as exc:
            raise_if_timeout(exc)
            return None, RewardResult(0.0, "schema_error", _short_error(exc))
    else:
        return None, RewardResult(0.0, "schema_error", f"Unsupported schema type: {type(schema_str).__name__}")

    if not isinstance(schema, dict):
        return None, RewardResult(0.0, "schema_error", "schema_str must decode to a JSON object")
    return strictify_schema(schema), None


def _validator_for(schema: dict[str, Any]) -> Draft202012Validator:
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def validate_structured_object(schema_str: Any, response_obj: Any) -> RewardResult:
    schema, error = _load_schema(schema_str)
    if error is not None:
        return error

    try:
        _validator_for(schema).validate(response_obj)
        return RewardResult(1.0)
    except SchemaError as exc:
        raise_if_timeout(exc)
        return RewardResult(0.0, "schema_error", _short_error(exc))
    except ValidationError as exc:
        raise_if_timeout(exc)
        return RewardResult(0.0, "validation_error", _short_error(exc))
    except Exception as exc:
        raise_if_timeout(exc)
        return RewardResult(0.0, "validation_error", _short_error(exc))


def _xml_element_to_obj(element: ET.Element) -> Any:
    children = list(element)
    text = (element.text or "").strip()
    if not children:
        return text if text else None

    result: dict[str, Any] = {}
    for child in children:
        value = _xml_element_to_obj(child)
        if child.tag in result:
            existing = result[child.tag]
            if isinstance(existing, list):
                existing.append(value)
            else:
                result[child.tag] = [existing, value]
        else:
            result[child.tag] = value
    return result


def _parse_xml(content: str, schema: dict[str, Any]) -> dict[str, Any]:
    root = ET.fromstring(content)
    root_obj = _xml_element_to_obj(root)
    properties = schema.get("properties")
    if schema.get("type") == "object" and isinstance(properties, dict):
        if root.tag in properties:
            return {root.tag: root_obj}
        if isinstance(root_obj, dict):
            return root_obj
    return {root.tag: root_obj}


def _coerce_xml_types(data: Any, schema: dict[str, Any]) -> Any:
    schema_type = schema.get("type")

    if isinstance(schema_type, list):
        non_null = [item for item in schema_type if item != "null"]
        if data is None and "null" in schema_type:
            return None
        for candidate in non_null:
            coerced = _coerce_xml_types(data, {**schema, "type": candidate})
            if not (candidate in {"integer", "number", "boolean"} and isinstance(coerced, str)):
                return coerced
        return data

    if schema_type == "object" and isinstance(data, dict):
        properties = schema.get("properties", {})
        return {
            key: _coerce_xml_types(value, properties[key]) if key in properties else value
            for key, value in data.items()
        }

    if schema_type == "array":
        items_schema = schema.get("items", {})
        if isinstance(data, dict) and len(data) == 1:
            data = next(iter(data.values()))
        if not isinstance(data, list):
            data = [data] if data is not None else []
        return [_coerce_xml_types(item, items_schema) for item in data]

    if data is None and schema_type == "string":
        return ""

    if isinstance(data, str):
        try:
            if schema_type == "integer":
                return int(data)
            if schema_type == "number":
                return float(data)
            if schema_type == "boolean":
                lowered = data.lower()
                if lowered in ("true", "1"):
                    return True
                if lowered in ("false", "0"):
                    return False
        except (TypeError, ValueError) as exc:
            raise_if_timeout(exc)
            return data

    return data


def _coerce_csv_scalar(value: str | None, target_type: Any) -> Any:
    if isinstance(target_type, list):
        if (value is None or value == "") and "null" in target_type:
            return None
        for item_type in target_type:
            if item_type == "null":
                continue
            result = _coerce_csv_scalar(value, item_type)
            if not isinstance(result, str) or item_type == "string":
                return result
        return value

    if value is None or value == "":
        return value

    if target_type in {"object", "array"} and isinstance(value, str):
        try:
            return json.loads(value)
        except Exception as exc:
            raise_if_timeout(exc)
            try:
                return ast.literal_eval(value)
            except Exception as exc:
                raise_if_timeout(exc)
                return value

    try:
        if target_type == "integer":
            return int(value)
        if target_type == "number":
            return float(value)
        if target_type == "boolean":
            lowered = value.lower()
            if lowered in ("true", "1"):
                return True
            if lowered in ("false", "0"):
                return False
    except (TypeError, ValueError) as exc:
        raise_if_timeout(exc)
        return value
    return value


def _coerce_csv_types(rows: list[dict[str, str]], schema: dict[str, Any]) -> list[dict[str, Any]]:
    items_schema = schema.get("items", schema)
    properties = items_schema.get("properties", {})
    coerced_rows: list[dict[str, Any]] = []
    for row in rows:
        coerced_rows.append(
            {
                key: _coerce_csv_scalar(value, properties.get(key, {}).get("type", "string"))
                for key, value in row.items()
            }
        )
    return coerced_rows


def parse_structured_text(schema_type: str | SchemaType, content: str, schema: dict[str, Any]) -> Any:
    try:
        normalized = schema_type if isinstance(schema_type, SchemaType) else SchemaType(str(schema_type).lower())
    except ValueError as exc:
        raise_if_timeout(exc)
        raise UnsupportedSchemaTypeError(f"Unsupported schema_type: {schema_type!r}") from exc
    content = _strip_code_fence(content)

    if normalized == SchemaType.JSON:
        return json.loads(content)
    if normalized == SchemaType.YAML:
        return yaml.safe_load(content)
    if normalized == SchemaType.XML:
        return _coerce_xml_types(_parse_xml(content, schema), schema)
    if normalized == SchemaType.TOML:
        return tomllib.loads(content)
    if normalized == SchemaType.CSV:
        rows = _coerce_csv_types(list(csv.DictReader(io.StringIO(content))), schema)
        if schema.get("type") == "object" and len(rows) == 1:
            return rows[0]
        return rows

    raise UnsupportedSchemaTypeError(f"Unsupported schema_type: {schema_type!r}")


def evaluate_structured_output_response(
    schema_type: Any,
    schema_str: Any,
    response_text: str,
    *,
    expected_fence: str | None = None,
) -> RewardResult:
    if not response_text or not response_text.strip():
        return RewardResult(0.0, "empty_response", "No assistant response text")

    schema, error = _load_schema(schema_str)
    if error is not None:
        return error

    first_error: RewardResult | None = None
    if expected_fence:
        fenced_text, error = _extract_required_code_fence(response_text, expected_fence)
        if error is None and fenced_text is not None:
            result = _evaluate_structured_output_candidate(schema_type, schema, fenced_text)
            if result.reward == 1.0:
                return result
            first_error = result
        else:
            first_error = error

    fallback_errors: list[RewardResult] = []
    for candidate in _structured_candidates(schema_type, response_text, expected_fence):
        result = _evaluate_structured_output_candidate(schema_type, schema, candidate.content)
        if result.reward == 1.0:
            return result
        fallback_errors.append(result)

    whole_response_result = _evaluate_structured_output_candidate(schema_type, schema, response_text)
    if whole_response_result.reward == 1.0:
        return whole_response_result
    if first_error is not None and first_error.error_type not in {"missing_code_fence", "extra_text_outside_code_fence"}:
        return first_error
    if fallback_errors:
        return fallback_errors[0]
    return whole_response_result


def _evaluate_structured_output_candidate(schema_type: Any, schema: dict[str, Any], response_text: str) -> RewardResult:
    try:
        response_obj = parse_structured_text(schema_type, response_text, schema)
    except UnsupportedSchemaTypeError as exc:
        raise_if_timeout(exc)
        return RewardResult(0.0, "schema_type_error", _short_error(exc))
    except Exception as exc:
        raise_if_timeout(exc)
        return RewardResult(0.0, "parse_error", _short_error(exc))

    try:
        _validator_for(schema).validate(response_obj)
        return RewardResult(1.0)
    except SchemaError as exc:
        raise_if_timeout(exc)
        return RewardResult(0.0, "schema_error", _short_error(exc))
    except ValidationError as exc:
        raise_if_timeout(exc)
        return RewardResult(0.0, "validation_error", _short_error(exc))
    except Exception as exc:
        raise_if_timeout(exc)
        return RewardResult(0.0, "validation_error", _short_error(exc))


def _loads_json_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception as exc:
        raise_if_timeout(exc)
        return value


def _iter_json_tool_calls(obj: Any) -> list[dict[str, Any]]:
    if not isinstance(obj, dict):
        return []

    if "output" in obj and isinstance(obj["output"], list):
        calls = []
        for item in obj["output"]:
            if isinstance(item, dict) and item.get("type") in {"function_call", "tool_call"}:
                calls.append(item)
        if calls:
            return calls

    if "tool_calls" in obj and isinstance(obj["tool_calls"], list):
        return [item for item in obj["tool_calls"] if isinstance(item, dict)]

    if "function_call" in obj and isinstance(obj["function_call"], dict):
        return [obj["function_call"]]

    if "name" in obj and ("arguments" in obj or "args" in obj):
        return [obj]

    if obj.get("type") in {"function_call", "tool_call"} and "name" in obj:
        return [obj]

    return []


def _canonicalize_json_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments", {})
    else:
        name = call.get("name")
        arguments = call.get("arguments", call.get("args", {}))
    return {"name": name, "arguments": _loads_json_maybe(arguments)}


def _parse_parameter_blocks(function_body: str) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    for match in re.finditer(
        r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", function_body, flags=re.DOTALL
    ):
        key = match.group(1).strip()
        value = match.group(2).strip()
        arguments[key] = _loads_json_maybe(value)
    return arguments


def _parse_tagged_tool_call(block: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(block)
    except Exception as exc:
        raise_if_timeout(exc)
        obj = None
    if isinstance(obj, dict):
        calls = _iter_json_tool_calls(obj)
        if calls:
            return _canonicalize_json_tool_call(calls[0])

    match = re.search(r"<function=([^>\s]+)>\s*(.*?)\s*</function>", block, flags=re.DOTALL)
    if match:
        name = match.group(1).strip()
        body = match.group(2).strip()
        arguments = _parse_parameter_blocks(body)
        if not arguments:
            parsed_body = _loads_json_maybe(body)
            arguments = parsed_body if isinstance(parsed_body, dict) else {}
        return {"name": name, "arguments": arguments}

    match = re.search(r'<invoke\s+name=["\']([^"\']+)["\']\s*>\s*(.*?)\s*</invoke>', block, flags=re.DOTALL)
    if match:
        name = match.group(1).strip()
        arguments = _loads_json_maybe(match.group(2).strip())
        return {"name": name, "arguments": arguments if isinstance(arguments, dict) else {}}

    return None


def _extract_tool_call_payload_from_candidate(
    response_text: str, metadata: dict[str, Any]
) -> tuple[Any | None, RewardResult | None]:
    calls: list[dict[str, Any]] = []

    try:
        obj = json.loads(response_text)
    except Exception as exc:
        raise_if_timeout(exc)
        obj = None
    if obj is not None:
        calls.extend(_canonicalize_json_tool_call(call) for call in _iter_json_tool_calls(obj))

    if not calls:
        call = _parse_tagged_tool_call(response_text)
        if call is not None:
            calls.append(call)

    if not calls:
        return None, RewardResult(0.0, "missing_tool_call", "No function_call item found in assistant response")

    if len(calls) > 1:
        names = ", ".join(str(call.get("name")) for call in calls)
        return None, RewardResult(0.0, "multiple_tool_calls", f"Expected exactly one function_call, got: {names}")

    selected_call = calls[0]
    expected_tool_name = metadata.get("tool_name")
    if expected_tool_name and selected_call.get("name") != expected_tool_name:
        return None, RewardResult(
            0.0,
            "wrong_tool_name",
            f"Expected tool {expected_tool_name!r}, got: {selected_call.get('name')!r}",
        )

    arguments = selected_call.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise_if_timeout(exc)
            return None, RewardResult(0.0, "tool_arguments_parse_error", _short_error(exc))
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return None, RewardResult(
            0.0,
            "tool_arguments_parse_error",
            f"Unsupported arguments type: {type(arguments).__name__}",
        )

    payload_key = metadata.get("tool_payload_key")
    if payload_key:
        if payload_key not in arguments:
            return None, RewardResult(0.0, "missing_tool_payload_key", f"Missing tool payload key: {payload_key}")
        return arguments[payload_key], None

    return arguments, None


def extract_tool_call_payload(response_text: str, metadata: dict[str, Any]) -> tuple[Any | None, RewardResult | None]:
    expected_fence = _expected_response_fence(metadata, metadata.get("schema_type", SchemaType.JSON))
    first_error: RewardResult | None = None
    if expected_fence:
        fenced_text, error = _extract_required_code_fence(response_text, expected_fence)
        if error is None and fenced_text is not None:
            payload, payload_error = _extract_tool_call_payload_from_candidate(fenced_text, metadata)
            if payload_error is None:
                return payload, None
            first_error = payload_error
        else:
            first_error = error

    fallback_errors: list[RewardResult] = []
    for candidate in _tool_call_candidates(response_text, expected_fence):
        payload, error = _extract_tool_call_payload_from_candidate(candidate.content, metadata)
        if error is None:
            return payload, None
        fallback_errors.append(error)

    payload, error = _extract_tool_call_payload_from_candidate(_strip_code_fence(response_text), metadata)
    if error is None:
        return payload, None
    if first_error is not None and first_error.error_type not in {"missing_code_fence", "extra_text_outside_code_fence"}:
        return None, first_error
    if fallback_errors:
        return None, fallback_errors[0]
    return None, error


def score_structured_outputs(sample: Any) -> RewardResult:
    metadata = _get_metadata(sample)
    data_type = metadata.get("data_type")
    if data_type != "nemotron_structured_outputs":
        raise ValueError(f"unsupported data_type: {data_type!r}")

    schema_str = metadata.get("schema_str")
    schema_type = metadata.get("schema_type", SchemaType.JSON)
    response = _response_text(sample)
    expected_fence = _expected_response_fence(metadata, schema_type)

    if metadata.get("response_mode") == "tool_call":
        if not response or not response.strip():
            return RewardResult(0.0, "empty_response", "No assistant response text")
        payload, error = extract_tool_call_payload(response, metadata)
        if error is not None:
            return error
        return validate_structured_object(schema_str, payload)

    return evaluate_structured_output_response(schema_type, schema_str, response, expected_fence=expected_fence)


def _append_reward_log(sample: Any, result: RewardResult) -> None:
    """Append a structured reward record to the JSONL log file."""
    if not REWARD_LOG_PATH:
        return

    try:
        metadata = getattr(sample, "metadata", None)
        metadata = metadata if isinstance(metadata, dict) else {}
        raw_response = getattr(sample, "response", "")
        raw_response = raw_response if isinstance(raw_response, str) else str(raw_response or "")

        record = {
            "reward": result.reward,
            "error_type": result.error_type,
            "error_message": result.error_message,
            "data_type": metadata.get("data_type"),
            "sub_data_type": metadata.get("sub_data_type"),
            "schema_type": metadata.get("schema_type"),
            "response_mode": metadata.get("response_mode"),
            "expected_response_fence": metadata.get("expected_response_fence"),
            "tool_name": metadata.get("tool_name"),
            "tool_payload_key": metadata.get("tool_payload_key"),
            "response": _extract_response_content(raw_response),
            "raw_response": raw_response,
            "label": getattr(sample, "label", ""),
            "metadata": metadata,
        }

        log_dir = os.path.dirname(REWARD_LOG_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(REWARD_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        raise_if_timeout(exc)
        logger.warning("Failed to write Nemotron structured-output reward log: %s", exc)


async def _score_single_sample(sample: Any) -> float:
    try:
        result = score_structured_outputs(sample)
    except Exception as exc:
        raise_if_timeout(exc)
        raise
    _append_reward_log(sample, result)
    return float(result.reward)


async def structured_outputs_reward(args: Any, sample_or_samples: Any, **kwargs) -> float | list[float]:
    """MILES reward entrypoint.

    The data processor writes all reward-critical fields into sample.metadata:
    schema_str, schema_type, response_mode, tool_name, and tool_payload_key.
    """
    del args, kwargs
    if isinstance(sample_or_samples, list):
        return await asyncio.gather(*(_score_single_sample(sample) for sample in sample_or_samples))
    return await _score_single_sample(sample_or_samples)


custom_rm = structured_outputs_reward
