"""Unit tests for Nemotron structured-output reward."""

from __future__ import annotations

import json
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

if "torch" not in sys.modules:
    sys.modules["torch"] = MagicMock()

from slime.utils.types import Sample

from recipe.nemotron_structured_outputs.structured_outputs_reward import (
    _extract_response_content,
    extract_tool_call_payload,
    score_structured_outputs,
    structured_outputs_reward,
)


def _sample(response: str, schema: dict[str, Any], **metadata: Any) -> Sample:
    base_metadata = {
        "data_type": "nemotron_structured_outputs",
        "sub_data_type": "direct_generation",
        "schema_str": json.dumps(schema),
        "schema_type": "json",
    }
    base_metadata.update(metadata)
    return Sample(response=response, label="", metadata=base_metadata)


def _person_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0},
            "active": {"type": "boolean"},
        },
    }


@pytest.mark.asyncio
async def test_json_valid_response_scores_one() -> None:
    sample = _sample('{"name":"Alice","age":30,"active":true}', _person_schema())
    assert await structured_outputs_reward(None, sample) == 1.0


@pytest.mark.asyncio
async def test_json_missing_property_scores_zero_after_strictify() -> None:
    sample = _sample('{"name":"Alice","age":30}', _person_schema())
    assert await structured_outputs_reward(None, sample) == 0.0
    result = score_structured_outputs(sample)
    assert result.error_type == "validation_error"


@pytest.mark.asyncio
async def test_json_extra_property_scores_zero_after_strictify() -> None:
    sample = _sample('{"name":"Alice","age":30,"active":true,"extra":"x"}', _person_schema())
    assert await structured_outputs_reward(None, sample) == 0.0


@pytest.mark.asyncio
async def test_invalid_json_scores_zero() -> None:
    sample = _sample('{"name":', _person_schema())
    result = score_structured_outputs(sample)
    assert result.reward == 0.0
    assert result.error_type == "parse_error"


@pytest.mark.asyncio
async def test_yaml_response() -> None:
    response = "name: Alice\nage: 30\nactive: true\n"
    sample = _sample(response, _person_schema(), schema_type="yaml")
    assert await structured_outputs_reward(None, sample) == 1.0


@pytest.mark.asyncio
async def test_toml_response() -> None:
    response = 'name = "Alice"\nage = 30\nactive = true\n'
    sample = _sample(response, _person_schema(), schema_type="toml")
    assert await structured_outputs_reward(None, sample) == 1.0


@pytest.mark.asyncio
async def test_csv_response_with_type_coercion() -> None:
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "active": {"type": "boolean"},
            },
        },
    }
    sample = _sample("name,age,active\nAlice,30,true\nBob,41,false\n", schema, schema_type="csv")
    assert await structured_outputs_reward(None, sample) == 1.0


@pytest.mark.asyncio
async def test_xml_response_with_type_coercion() -> None:
    schema = {
        "type": "object",
        "properties": {
            "output": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "active": {"type": "boolean"},
                },
            }
        },
    }
    response = "<output><name>Alice</name><age>30</age><active>true</active></output>"
    sample = _sample(response, schema, schema_type="xml")
    assert await structured_outputs_reward(None, sample) == 1.0


@pytest.mark.asyncio
async def test_markdown_code_fence_is_accepted() -> None:
    sample = _sample("```json\n{\"name\":\"Alice\",\"age\":30,\"active\":true}\n```", _person_schema())
    assert await structured_outputs_reward(None, sample) == 1.0


@pytest.mark.asyncio
async def test_expected_json_fence_is_required_for_prompted_data() -> None:
    schema = _person_schema()
    sample = _sample(
        "```json\n{\"name\":\"Alice\",\"age\":30,\"active\":true}\n```",
        schema,
        final_output_prompt_version="fenced_code_block_v1",
        expected_response_fence="json",
    )
    assert await structured_outputs_reward(None, sample) == 1.0

    missing_fence = _sample(
        "{\"name\":\"Alice\",\"age\":30,\"active\":true}",
        schema,
        final_output_prompt_version="fenced_code_block_v1",
        expected_response_fence="json",
    )
    result = score_structured_outputs(missing_fence)
    assert result.reward == 1.0

    extra_text = _sample(
        "Here is the answer:\n```json\n{\"name\":\"Alice\",\"age\":30,\"active\":true}\n```",
        schema,
        final_output_prompt_version="fenced_code_block_v1",
        expected_response_fence="json",
    )
    result = score_structured_outputs(extra_text)
    assert result.reward == 1.0


@pytest.mark.asyncio
async def test_prompted_json_falls_back_to_last_complete_object() -> None:
    response = (
        "Earlier draft:\n"
        "{\"name\":\"Alice\",\"age\":30}\n"
        "Final answer:\n"
        "{\"name\":\"Alice\",\"age\":30,\"active\":true}\n"
    )
    sample = _sample(
        response,
        _person_schema(),
        final_output_prompt_version="fenced_code_block_v1",
        expected_response_fence="json",
    )
    assert await structured_outputs_reward(None, sample) == 1.0


@pytest.mark.asyncio
async def test_prompted_json_falls_back_to_whole_response() -> None:
    sample = _sample(
        "{\"name\":\"Alice\",\"age\":30,\"active\":true}",
        _person_schema(),
        final_output_prompt_version="fenced_code_block_v1",
        expected_response_fence="json",
    )
    assert await structured_outputs_reward(None, sample) == 1.0


@pytest.mark.asyncio
async def test_response_content_after_think_before_im_end() -> None:
    response = '<think>{"bad": true}</think>\n{"name":"Alice","age":30,"active":true}<|im_end|>ignored'
    sample = _sample(response, _person_schema())
    assert await structured_outputs_reward(None, sample) == 1.0


@pytest.mark.asyncio
async def test_tool_call_qwen_json_format() -> None:
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }
    response = '<tool_call>\n{"name":"extract_person","arguments":{"payload":{"name":"Alice","age":30}}}\n</tool_call>'
    sample = _sample(
        response,
        schema,
        sub_data_type="tool_calling_extraction",
        response_mode="tool_call",
        tool_name="extract_person",
        tool_payload_key="payload",
    )
    assert await structured_outputs_reward(None, sample) == 1.0


@pytest.mark.asyncio
async def test_expected_tool_call_fence_is_required_for_prompted_data() -> None:
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }
    response = '```tool_call\n{"name":"extract_person","arguments":{"payload":{"name":"Alice","age":30}}}\n```'
    sample = _sample(
        response,
        schema,
        sub_data_type="tool_calling_extraction",
        response_mode="tool_call",
        tool_name="extract_person",
        tool_payload_key="payload",
        final_output_prompt_version="fenced_code_block_v1",
        expected_response_fence="tool_call",
    )
    assert await structured_outputs_reward(None, sample) == 1.0

    wrong_fence = _sample(
        '```json\n{"name":"extract_person","arguments":{"payload":{"name":"Alice","age":30}}}\n```',
        schema,
        sub_data_type="tool_calling_extraction",
        response_mode="tool_call",
        tool_name="extract_person",
        tool_payload_key="payload",
        final_output_prompt_version="fenced_code_block_v1",
        expected_response_fence="tool_call",
    )
    result = score_structured_outputs(wrong_fence)
    assert result.reward == 1.0


@pytest.mark.asyncio
async def test_prompted_tool_call_falls_back_to_last_complete_json_call() -> None:
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }
    response = (
        "Bad earlier call:\n"
        '{"name":"extract_person","arguments":{"payload":{"name":"Alice"}}}\n'
        "Final call:\n"
        '{"name":"extract_person","arguments":{"payload":{"name":"Alice","age":30}}}'
    )
    sample = _sample(
        response,
        schema,
        sub_data_type="tool_calling_extraction",
        response_mode="tool_call",
        tool_name="extract_person",
        tool_payload_key="payload",
        final_output_prompt_version="fenced_code_block_v1",
        expected_response_fence="tool_call",
    )
    assert await structured_outputs_reward(None, sample) == 1.0


@pytest.mark.asyncio
async def test_tool_call_after_think_before_im_end() -> None:
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }
    response = (
        "<think>reasoning</think>"
        '<tool_call>\n{"name":"extract_person","arguments":{"payload":{"name":"Alice","age":30}}}\n</tool_call>'
        "<|im_end|>ignored"
    )
    sample = _sample(
        response,
        schema,
        sub_data_type="tool_calling_extraction",
        response_mode="tool_call",
        tool_name="extract_person",
        tool_payload_key="payload",
    )
    assert await structured_outputs_reward(None, sample) == 1.0


@pytest.mark.asyncio
async def test_tool_call_qwen35_parameter_format() -> None:
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }
    response = (
        "<tool_call>\n"
        "<function=extract_person>\n"
        "<parameter=payload>\n"
        "{\"name\":\"Alice\",\"age\":30}\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    sample = _sample(
        response,
        schema,
        sub_data_type="tool_calling_extraction",
        response_mode="tool_call",
        tool_name="extract_person",
        tool_payload_key="payload",
    )
    assert await structured_outputs_reward(None, sample) == 1.0


@pytest.mark.asyncio
async def test_openai_style_tool_call() -> None:
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }
    response = json.dumps(
        {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "extract_person",
                        "arguments": json.dumps({"payload": {"name": "Alice", "age": 30}}),
                    },
                }
            ]
        }
    )
    sample = _sample(
        response,
        schema,
        sub_data_type="tool_calling_extraction",
        response_mode="tool_call",
        tool_name="extract_person",
        tool_payload_key="payload",
    )
    assert await structured_outputs_reward(None, sample) == 1.0


@pytest.mark.asyncio
async def test_tool_call_wrong_tool_scores_zero() -> None:
    response = '<tool_call>\n{"name":"wrong","arguments":{"name":"Alice","age":30}}\n</tool_call>'
    sample = _sample(
        response,
        {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}},
        sub_data_type="tool_calling_extraction",
        response_mode="tool_call",
        tool_name="extract_person",
    )
    result = score_structured_outputs(sample)
    assert result.reward == 0.0
    assert result.error_type == "wrong_tool_name"


@pytest.mark.asyncio
async def test_tool_call_missing_payload_scores_zero() -> None:
    response = '<tool_call>\n{"name":"extract_person","arguments":{"other":{"name":"Alice","age":30}}}\n</tool_call>'
    sample = _sample(
        response,
        {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}},
        sub_data_type="tool_calling_extraction",
        response_mode="tool_call",
        tool_name="extract_person",
        tool_payload_key="payload",
    )
    result = score_structured_outputs(sample)
    assert result.reward == 0.0
    assert result.error_type == "missing_tool_payload_key"


@pytest.mark.asyncio
async def test_tool_call_requires_call_wrapper() -> None:
    sample = _sample(
        '{"payload":{"name":"Alice","age":30}}',
        {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}},
        sub_data_type="tool_calling_extraction",
        response_mode="tool_call",
        tool_name="extract_person",
        tool_payload_key="payload",
    )
    result = score_structured_outputs(sample)
    assert result.reward == 0.0
    assert result.error_type == "missing_tool_call"


@pytest.mark.asyncio
async def test_batch_reward() -> None:
    samples = [
        _sample('{"name":"Alice","age":30,"active":true}', _person_schema()),
        _sample('{"name":"Alice","age":30}', _person_schema()),
    ]
    assert await structured_outputs_reward(None, samples) == [1.0, 0.0]


def test_tool_payload_extraction_helper() -> None:
    metadata = {"tool_name": "extract_person", "tool_payload_key": "payload"}
    response = '<tool_call>\n{"name":"extract_person","arguments":{"payload":{"name":"Alice"}}}\n</tool_call>'
    payload, error = extract_tool_call_payload(response, metadata)
    assert error is None
    assert payload == {"name": "Alice"}


def test_extract_response_content() -> None:
    assert _extract_response_content("<think>x</think> answer <|im_end|> trailing") == "answer"
