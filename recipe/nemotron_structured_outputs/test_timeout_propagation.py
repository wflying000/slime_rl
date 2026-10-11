"""Regression tests for timeouts in parsing, validation and reward logging."""

import asyncio
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from recipe.rl_reward.timeout_utils import RewardTimeout

reward = importlib.import_module("recipe.nemotron_structured_outputs.structured_outputs_reward")
SCHEMA = {"type": "object", "properties": {"value": {"type": "integer"}}}


def sample(response='{"value":1}', schema_type="json"):
    return SimpleNamespace(response=response, label="", metadata={
        "data_type": "nemotron_structured_outputs", "schema_type": schema_type,
        "schema_str": SCHEMA,
    })


def wrapped_timeout():
    error = RuntimeError("wrapped validation failure")
    error.__cause__ = TimeoutError("underlying deadline")
    return error


class ReadTimeout(Exception):
    pass


class TestStructuredOutputTimeouts(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        log_patch = patch.object(reward, "REWARD_LOG_PATH", None)
        log_patch.start()
        self.addCleanup(log_patch.stop)

    async def test_entrypoint_propagates_plain_named_wrapped_and_shared_timeouts(self):
        for factory in (lambda: TimeoutError("deadline"), lambda: ReadTimeout("read deadline"),
                        wrapped_timeout, lambda: RewardTimeout("already classified")):
            for batch in (False, True):
                with self.subTest(exception=factory, batch=batch):
                    with patch.object(reward, "score_structured_outputs", side_effect=factory()):
                        with self.assertRaises(RewardTimeout):
                            await reward.custom_rm(None, [sample()] if batch else sample())

    async def test_all_format_parser_timeouts_propagate(self):
        cases = [
            ("json", reward.json, "loads", '{"value":1}'),
            ("yaml", reward.yaml, "safe_load", "value: 1"),
            ("xml", reward.ET, "fromstring", "<root><value>1</value></root>"),
            ("toml", reward.tomllib, "loads", "value = 1"),
            ("csv", reward.csv, "DictReader", "value\n1\n"),
        ]
        for schema_type, parser, method, text in cases:
            with self.subTest(schema_type=schema_type), patch.object(parser, method, side_effect=TimeoutError("parse deadline")):
                with self.assertRaises(RewardTimeout):
                    await reward.custom_rm(None, sample(text, schema_type))

    async def test_schema_loading_timeout_propagates(self):
        item = sample()
        item.metadata["schema_str"] = '{"type":"object"}'
        with patch.object(reward.json, "loads", side_effect=TimeoutError("schema deadline")):
            with self.assertRaises(RewardTimeout):
                await reward.custom_rm(None, item)

    async def test_schema_and_validation_errors_with_timeout_cause_propagate(self):
        for error_type in (reward.SchemaError, reward.ValidationError):
            with self.subTest(error_type=error_type):
                error = error_type("wrapped deadline")
                error.__cause__ = TimeoutError("validation deadline")
                with patch.object(reward, "_validator_for", side_effect=error):
                    with self.assertRaises(RewardTimeout):
                        await reward.custom_rm(None, sample())
                    with self.assertRaises(RewardTimeout):
                        reward.validate_structured_object(SCHEMA, {"value": 1})

    async def test_logging_timeout_propagates(self):
        with patch.object(reward, "REWARD_LOG_PATH", "/unused/structured/reward.jsonl"), \
             patch.object(reward.os, "makedirs", side_effect=TimeoutError("log deadline")):
            with self.assertRaises(RewardTimeout):
                await reward.custom_rm(None, sample())

    async def test_ordinary_log_error_does_not_change_reward(self):
        with patch.object(reward, "REWARD_LOG_PATH", "/unused/structured/reward.jsonl"), \
             patch.object(reward.os, "makedirs", side_effect=OSError("log unavailable")), \
             self.assertLogs(reward.logger, level="WARNING"):
            self.assertEqual(await reward.custom_rm(None, sample()), 1.0)

    async def test_invalid_metadata_still_raises(self):
        item = sample()
        item.metadata = None
        with self.assertRaises(ValueError):
            await reward.custom_rm(None, item)

    async def test_cancellation_propagates(self):
        with patch.object(reward, "score_structured_outputs", side_effect=asyncio.CancelledError()):
            with self.assertRaises(asyncio.CancelledError):
                await reward.custom_rm(None, sample())


class TestNestedParserTimeouts(unittest.TestCase):
    def test_csv_json_timeout_does_not_fall_back_to_python_literal(self):
        with patch.object(reward.json, "loads", side_effect=TimeoutError("JSON deadline")), \
             patch.object(reward.ast, "literal_eval", side_effect=AssertionError("must not fall back")):
            with self.assertRaises(RewardTimeout):
                reward._coerce_csv_scalar("[1]", "array")

    def test_csv_literal_timeout_does_not_return_raw_text(self):
        with patch.object(reward.json, "loads", side_effect=ValueError("not JSON")), \
             patch.object(reward.ast, "literal_eval", side_effect=wrapped_timeout()):
            with self.assertRaises(RewardTimeout):
                reward._coerce_csv_scalar("[1]", "array")

    def test_tool_call_json_timeouts_do_not_trigger_fallbacks(self):
        helpers = [
            lambda: reward._loads_json_maybe('{"value":1}'),
            lambda: reward._parse_tagged_tool_call('{"name":"tool","arguments":{}}'),
            lambda: reward._extract_tool_call_payload_from_candidate('{"name":"tool","arguments":{}}', {}),
        ]
        for helper in helpers:
            with self.subTest(helper=helper), patch.object(reward.json, "loads", side_effect=TimeoutError("tool deadline")):
                with self.assertRaises(RewardTimeout):
                    helper()

    def test_xml_candidate_errors_with_timeout_cause_propagate(self):
        for text in ("<value />", "<value>1</value>"):
            error = reward.ET.ParseError("wrapped deadline")
            error.__cause__ = TimeoutError("XML deadline")
            with self.subTest(text=text), patch.object(reward.ET, "fromstring", side_effect=error):
                with self.assertRaises(RewardTimeout):
                    reward._xml_candidates(text)


if __name__ == "__main__":
    unittest.main()
