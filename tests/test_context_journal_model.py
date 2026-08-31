from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cpcodeagent.context import ContextEngine
from cpcodeagent.journal import EventKind, Journal
from cpcodeagent.model import ResilientModel, TransientModelError
from cpcodeagent.types import ModelResponse, ToolCall, ToolResult


class TrackingJournal(Journal):
    def __init__(self):
        super().__init__()
        self.after_calls: list[int] = []

    def after(self, seq: int):
        self.after_calls.append(seq)
        return super().after(seq)


class FailingModel:
    name = "failing"

    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools, on_text=None):
        self.calls += 1
        raise TransientModelError("temporary outage")


class WorkingModel:
    name = "fallback"

    def complete(self, messages, tools, on_text=None):
        if on_text is not None:
            on_text("fallback response")
        return ModelResponse(content="fallback response", model=self.name)


class ContextJournalModelTests(unittest.TestCase):
    def test_live_context_consumes_only_events_after_its_projection(self) -> None:
        journal = TrackingJournal()
        journal.append(EventKind.INPUT, {"content": "first", "source": "user"})
        engine = ContextEngine()
        state = engine.rebuild(journal)
        journal.after_calls.clear()

        engine.update(state, journal, "task", "/workspace", "read only")
        journal.append(
            EventKind.MODEL_RESPONSE,
            {"response": ModelResponse(content="second").to_dict()},
        )
        update = engine.update(state, journal, "task", "/workspace", "read only")

        self.assertEqual(journal.after_calls, [0, 0])
        self.assertEqual(state.projected_seq, 1)
        self.assertIn("second", str(update.view.messages))

    def test_half_full_context_mechanically_snips_tool_output(self) -> None:
        journal = Journal()
        call = ToolCall("read-1", "read_file", {"path": "large.txt"})
        journal.append(EventKind.INPUT, {"content": "inspect", "source": "user"})
        journal.append(
            EventKind.MODEL_RESPONSE,
            {"response": ModelResponse(tool_calls=(call,)).to_dict()},
        )
        journal.append(EventKind.TOOL_CALL, {"call": call.to_dict()})
        journal.append(
            EventKind.TOOL_RESULT,
            {"result": ToolResult("read-1", True, "x" * 2_400).to_dict()},
        )
        engine = ContextEngine(
            max_context_tokens=1_000,
            max_tool_output_chars=300,
        )

        update = engine.update(
            engine.rebuild(journal), journal, "task", "/workspace", "read only"
        )

        tool_message = next(
            message for message in update.view.messages if message["role"] == "tool"
        )
        self.assertIn("snipped at 50%", tool_message["content"])
        self.assertEqual(update.compactions, ())

    def test_seventy_percent_context_uses_durable_semantic_summary(self) -> None:
        journal = Journal()
        for index in range(12):
            journal.append(
                EventKind.INPUT,
                {"content": f"request-{index}-" + "r" * 100, "source": "user"},
            )
            journal.append(
                EventKind.MODEL_RESPONSE,
                {
                    "response": ModelResponse(
                        content=f"answer-{index}-" + "a" * 100
                    ).to_dict()
                },
            )
        engine = ContextEngine(max_context_tokens=1_000, keep_recent_blocks=4)
        summary_levels: list[str] = []

        def summarize(level: str, source: str) -> ModelResponse:
            summary_levels.append(level)
            self.assertIn("request-0", source)
            return ModelResponse(
                content="Preserve the early architectural decision.",
                prompt_tokens=11,
                completion_tokens=5,
            )

        state = engine.rebuild(journal)
        update = engine.update(
            state,
            journal,
            "task",
            "/workspace",
            "read only",
            summarizer=summarize,
        )
        engine.commit_compactions(state, journal, update.compactions)

        self.assertEqual(summary_levels, ["summary"])
        self.assertEqual(update.compression_tokens, 16)
        self.assertEqual(len(state.blocks), 4)
        self.assertIn("early architectural decision", update.view.memory_snapshot)
        self.assertIsNotNone(journal.last(EventKind.CONTEXT_COMPACTION))

        rebuilt = engine.rebuild(journal)
        replayed = engine.update(
            rebuilt,
            journal,
            "task",
            "/workspace",
            "read only",
            summarizer=lambda *_: self.fail("durable summary should be replayed"),
        )
        self.assertEqual(replayed.compactions, ())
        self.assertEqual(replayed.view.memory_snapshot, update.view.memory_snapshot)

    def test_emergency_retry_collapses_to_the_tight_recent_tail(self) -> None:
        journal = Journal()
        for index in range(8):
            journal.append(
                EventKind.INPUT,
                {"content": f"request-{index}-" + "x" * 80, "source": "user"},
            )
        engine = ContextEngine(max_context_tokens=10_000, emergency_keep_blocks=2)
        state = engine.rebuild(journal)

        update = engine.update(
            state,
            journal,
            "task",
            "/workspace",
            "read only",
            summarizer=lambda level, _: ModelResponse(content=f"{level} summary"),
            force_emergency=True,
        )

        self.assertEqual([item.level for item in update.compactions], ["emergency"])
        self.assertEqual(len(state.blocks), 2)
        self.assertIn("emergency summary", update.view.memory_snapshot)
        self.assertIn("request-7", str(update.view.messages))

    def test_compaction_is_a_view_and_keeps_raw_events(self) -> None:
        journal = Journal()
        journal.append(EventKind.INPUT, {"content": "initial task", "source": "user"})
        for index in range(8):
            journal.append(
                EventKind.MODEL_RESPONSE,
                {"response": ModelResponse(content=f"analysis-{index}-" + "x" * 180).to_dict()},
            )
            journal.append(
                EventKind.INPUT,
                {"content": f"observation-{index}-" + "y" * 180, "source": "test"},
            )
        before = journal.events

        view = ContextEngine(max_working_chars=900, snapshot_chars=500).build(
            journal, "task", "/workspace", "read only"
        )

        self.assertIsNotNone(view.memory_snapshot)
        self.assertIn("source hash", view.memory_snapshot)
        self.assertEqual(journal.events, before)
        self.assertLess(len(str(view.messages)), len(str(before)) * 2)

    def test_incomplete_trailing_jsonl_record_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            journal = Journal(path)
            journal.append(EventKind.INPUT, {"content": "task"})
            with path.open("ab") as handle:
                handle.write(b'{"seq": 1, "kind": "tool_call"')

            recovered = Journal(path)
            recovered.append(EventKind.FINAL, {"status": "failed"})
            reloaded = Journal(path)

            self.assertEqual(len(reloaded.events), 2)
            self.assertEqual(reloaded.events[0].kind, EventKind.INPUT)
            self.assertEqual(reloaded.events[1].kind, EventKind.FINAL)

    def test_provider_fallback_happens_only_after_retry_budget(self) -> None:
        primary = FailingModel()
        model = ResilientModel(primary, WorkingModel(), max_attempts=2, sleeper=lambda _: None)

        response = model.complete([], [])

        self.assertEqual(primary.calls, 2)
        self.assertEqual(response.model, "fallback")


if __name__ == "__main__":
    unittest.main()
