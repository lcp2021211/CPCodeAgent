from __future__ import annotations

import json
import unittest

from qwen35_9b_lora.prepare_data import (
    is_strict_positive,
    normalize_tool_call,
    response_message,
)
from qwen35_9b_lora.validate_data import validate_sample


class DataToolTests(unittest.TestCase):
    def test_normalizes_recorded_tool_call_to_openai_shape(self) -> None:
        call = normalize_tool_call(
            {"id": "call-1", "name": "read_file", "arguments": {"path": "x.py"}}
        )
        self.assertEqual(call["function"]["name"], "read_file")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"path": "x.py"})

    def test_tool_only_response_is_a_valid_assistant_target(self) -> None:
        message = response_message(
            {
                "content": "",
                "tool_calls": [
                    {"id": "call-1", "name": "read_file", "arguments": {"path": "x.py"}}
                ],
            }
        )
        self.assertIsNone(message["content"])
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "read_file")

    def test_strict_filter_requires_clean_finish_and_official_resolution(self) -> None:
        self.assertTrue(
            is_strict_positive(
                {"agent": {"status": "succeeded"}, "evaluation": {"resolved": True}}
            )
        )
        self.assertFalse(
            is_strict_positive(
                {"agent": {"status": "budget_exhausted"}, "evaluation": {"resolved": True}}
            )
        )

    def test_validator_accepts_agent_tool_target(self) -> None:
        sample = {
            "messages": [
                {"role": "user", "content": "Inspect x.py"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"x.py"}',
                            },
                        }
                    ],
                },
            ],
            "tools": json.dumps(
                [
                    {
                        "type": "function",
                        "function": {"name": "read_file", "parameters": {"type": "object"}},
                    }
                ]
            ),
        }
        self.assertEqual(validate_sample(sample, "test"), (1, 1))


if __name__ == "__main__":
    unittest.main()

