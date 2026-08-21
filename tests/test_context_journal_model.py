from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cpcodeagent.context import ContextEngine
from cpcodeagent.journal import EventKind, Journal
from cpcodeagent.model import ResilientModel, TransientModelError
from cpcodeagent.types import ModelResponse


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
