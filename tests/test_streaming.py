from __future__ import annotations

import io
import tempfile
import unittest
from collections import deque

from rich.console import Console

from cpcodeagent.builtin_tools import build_default_runtime
from cpcodeagent.executor import LocalExecutor
from cpcodeagent.journal import Journal
from cpcodeagent.kernel import Harness
from cpcodeagent.model import (
    ModelError,
    OpenAICompatibleModel,
    ResilientModel,
    ScriptedModel,
    TransientModelError,
)
from cpcodeagent.types import (
    ModelResponse,
    RunEvent,
    RunEventKind,
    RunOutcome,
    RunStatus,
    ToolCall,
    ToolResult,
)
from cpcodeagent.ui import TerminalUI


class _FunctionDelta:
    def __init__(self, name: str | None = None, arguments: str | None = None):
        self.name = name
        self.arguments = arguments


class _ToolDelta:
    def __init__(
        self,
        index: int,
        call_id: str | None = None,
        function: _FunctionDelta | None = None,
    ):
        self.index = index
        self.id = call_id
        self.function = function


class _Delta:
    def __init__(self, content: str | None = None, tool_calls=()):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, delta: _Delta):
        self.delta = delta


class _Usage:
    prompt_tokens = 12
    completion_tokens = 5


class _Chunk:
    def __init__(self, delta: _Delta | None = None, usage=None):
        self.choices = [_Choice(delta)] if delta is not None else []
        self.usage = usage


class _BadRequest(Exception):
    status_code = 400


class _Completions:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.calls: list[dict] = []

    def create(self, **params):
        self.calls.append(params)
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return iter(response)


class _Client:
    def __init__(self, responses):
        self.chat = type("Chat", (), {})()
        self.chat.completions = _Completions(responses)


class _PartialFailureModel:
    name = "partial"

    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools, on_text=None):
        self.calls += 1
        if on_text is not None:
            on_text("partial")
        raise TransientModelError("connection dropped")


class StreamingTests(unittest.TestCase):
    def test_streams_text_and_reassembles_fragmented_tool_calls(self) -> None:
        client = _Client(
            [
                [
                    _Chunk(_Delta(content="hello ")),
                    _Chunk(_Delta(content="world")),
                    _Chunk(
                        _Delta(
                            tool_calls=(
                                _ToolDelta(
                                    0,
                                    "call-1",
                                    _FunctionDelta("write_file", '{"path":"x.txt",'),
                                ),
                            )
                        )
                    ),
                    _Chunk(
                        _Delta(
                            tool_calls=(
                                _ToolDelta(0, function=_FunctionDelta(arguments='"content":"ok"}')),
                            )
                        )
                    ),
                    _Chunk(usage=_Usage()),
                ]
            ]
        )
        model = OpenAICompatibleModel("stream-model", "key", client=client)
        pieces: list[str] = []

        response = model.complete([], [], pieces.append)

        self.assertEqual(pieces, ["hello ", "world"])
        self.assertEqual(response.content, "hello world")
        self.assertEqual(response.prompt_tokens, 12)
        self.assertEqual(response.completion_tokens, 5)
        self.assertEqual(response.tool_calls[0].name, "write_file")
        self.assertEqual(response.tool_calls[0].arguments, {"path": "x.txt", "content": "ok"})
        request = client.chat.completions.calls[0]
        self.assertTrue(request["stream"])
        self.assertEqual(request["stream_options"], {"include_usage": True})

    def test_retries_without_stream_options_when_provider_rejects_them(self) -> None:
        client = _Client([_BadRequest("unsupported"), [_Chunk(_Delta(content="ok"))]])
        model = OpenAICompatibleModel("compatible", "key", client=client)

        response = model.complete([], [])

        self.assertEqual(response.content, "ok")
        self.assertIn("stream_options", client.chat.completions.calls[0])
        self.assertNotIn("stream_options", client.chat.completions.calls[1])

    def test_does_not_retry_after_partial_text_was_shown(self) -> None:
        primary = _PartialFailureModel()
        model = ResilientModel(primary, max_attempts=3, sleeper=lambda _: None)
        pieces: list[str] = []

        with self.assertRaisesRegex(ModelError, "duplicate text"):
            model.complete([], [], pieces.append)

        self.assertEqual(primary.calls, 1)
        self.assertEqual(pieces, ["partial"])

    def test_harness_emits_one_ordered_progress_stream(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            events: list[RunEvent] = []
            model = ScriptedModel(
                [
                    ModelResponse(
                        tool_calls=(ToolCall("read-1", "list_files", {"limit": 2}),)
                    ),
                    ModelResponse(content="done"),
                ]
            )
            harness = Harness(model, build_default_runtime(), event_sink=events.append)

            harness.run("inspect", LocalExecutor(directory), Journal(), "stream-session")

            kinds = [event.kind for event in events]
            self.assertEqual(kinds.count(RunEventKind.MODEL_START), 2)
            self.assertEqual(kinds.count(RunEventKind.MODEL_END), 2)
            self.assertIn(RunEventKind.TEXT_DELTA, kinds)
            self.assertLess(kinds.index(RunEventKind.TOOLS_START), kinds.index(RunEventKind.TOOLS_END))

    def test_terminal_ui_uses_color_and_does_not_repeat_streamed_answer(self) -> None:
        output = io.StringIO()
        ui = TerminalUI(
            Console(file=output, force_terminal=True, color_system="standard", width=100)
        )
        ui.handle(RunEvent(RunEventKind.MODEL_START))
        ui.handle(RunEvent(RunEventKind.TEXT_DELTA, {"text": "hello"}))
        ui.handle(RunEvent(RunEventKind.MODEL_END))
        ui.handle(
            RunEvent(
                RunEventKind.TOOLS_START,
                {"calls": (ToolCall("read-1", "read_file", {"path": "x.py"}),)},
            )
        )
        ui.handle(
            RunEvent(
                RunEventKind.TOOLS_END,
                {"results": (ToolResult("read-1", True, "1\tprint('x')"),)},
            )
        )
        ui.outcome(RunOutcome("s", RunStatus.SUCCEEDED, "hello", 1, 2, turn_id="t"))

        rendered = output.getvalue()
        self.assertIn("\x1b[", rendered)
        self.assertIn("agent", rendered)
        self.assertIn("read_file", rendered)
        self.assertEqual(rendered.count("hello"), 1)


if __name__ == "__main__":
    unittest.main()
