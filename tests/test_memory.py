from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from cpcodeagent.builtin_tools import build_default_runtime
from cpcodeagent.cli import interactive_loop
from cpcodeagent.context import ContextEngine
from cpcodeagent.executor import LocalExecutor
from cpcodeagent.journal import EventKind, Journal
from cpcodeagent.kernel import Harness
from cpcodeagent.memory import MemoryManager, MemoryScope, MemoryStore
from cpcodeagent.model import ScriptedModel
from cpcodeagent.session import SessionState, SessionStore
from cpcodeagent.types import ModelResponse


class MemoryTests(unittest.TestCase):
    def test_user_memory_is_shared_while_session_memory_is_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "memory"
            first = MemoryManager(MemoryStore(root, "session-one"))
            second = MemoryManager(MemoryStore(root, "session-two"))

            user_key = first.remember(MemoryScope.USER, "Prefer pytest for Python tests.")
            first.remember(MemoryScope.SESSION, "The current task concerns session recovery.")

            self.assertIn(user_key, second.store.read(MemoryScope.USER))
            self.assertIn("Prefer pytest", second.store.read(MemoryScope.USER))
            self.assertIn("session recovery", first.store.read(MemoryScope.SESSION))
            self.assertEqual(second.store.read(MemoryScope.SESSION), "")

    def test_successful_turn_is_recorded_and_injected_into_the_next_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            workspace.mkdir()
            journal = Journal(Path(directory) / "runs" / "chat.jsonl")
            manager = MemoryManager(MemoryStore(Path(directory) / "memory", "chat"))
            manager.remember(MemoryScope.USER, "Prefer concise answers.")
            model = ScriptedModel(
                [
                    ModelResponse(content="The project uses an append-only journal."),
                    ModelResponse(content="It supports replay after restart."),
                ]
            )
            harness = Harness(model, build_default_runtime(), memory=manager)
            executor = LocalExecutor(workspace)

            harness.run("Inspect persistence", executor, journal, "chat")
            harness.send("Can it recover?", executor, journal, "chat")

            session_memory = manager.store.read(MemoryScope.SESSION)
            self.assertIn("## turn-0001", session_memory)
            self.assertIn("Inspect persistence", session_memory)
            second_context = "\n".join(
                str(message.get("content") or "") for message in model.requests[1][0]
            )
            self.assertIn("Prefer concise answers", second_context)
            self.assertIn("The project uses an append-only journal", second_context)
            self.assertGreaterEqual(len(journal.find(EventKind.MEMORY_SNAPSHOT)), 2)
            self.assertEqual(
                SessionState.from_journal(journal).memory_root,
                str((Path(directory) / "memory").resolve()),
            )

            wrong_memory = MemoryManager(MemoryStore(Path(directory) / "other-memory", "chat"))
            wrong_harness = Harness(
                ScriptedModel([ModelResponse(content="not reached")]),
                build_default_runtime(),
                memory=wrong_memory,
            )
            with self.assertRaisesRegex(ValueError, "memory directory"):
                wrong_harness.send("Continue", executor, journal, "chat")

    def test_persistent_memory_is_pinned_when_history_is_compacted(self) -> None:
        journal = Journal()
        journal.append(
            EventKind.MEMORY_SNAPSHOT,
            {
                "digest": "memory-v1",
                "user": "# User Memory\n\n## preference\nUse pytest.\n",
                "session": "# Session Memory\n\n## decision\nKeep the journal immutable.\n",
            },
        )
        journal.append(EventKind.INPUT, {"content": "initial task", "source": "user"})
        for index in range(10):
            journal.append(
                EventKind.MODEL_RESPONSE,
                {"response": ModelResponse(content=f"analysis-{index}-" + "x" * 200).to_dict()},
            )
            journal.append(
                EventKind.INPUT,
                {"content": f"observation-{index}-" + "y" * 200, "source": "test"},
            )

        view = ContextEngine(max_working_chars=1_000, snapshot_chars=400).build(
            journal, "task", "/workspace", "read only"
        )

        self.assertIsNotNone(view.memory_snapshot)
        self.assertIn("Use pytest", view.messages[1]["content"])
        self.assertIn("Keep the journal immutable", view.messages[1]["content"])

    def test_repl_memory_commands_are_explicit_and_do_not_create_turns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            workspace.mkdir()
            sessions = SessionStore(Path(directory) / "runs")
            session = sessions.create("memory-chat")
            manager = MemoryManager(MemoryStore(Path(directory) / "memory", "memory-chat"))
            harness = Harness(ScriptedModel([]), build_default_runtime(), memory=manager)
            executor = LocalExecutor(workspace)
            harness.start_session(executor, session.journal, session.session_id)
            value = "Always use pytest."
            key = f"note-{hashlib.sha256(value.encode()).hexdigest()[:12]}"
            messages = iter(
                [
                    f"/remember user {value}",
                    "/memory user",
                    f"/forget user {key}",
                    "/exit",
                ]
            )
            output: list[str] = []

            result = interactive_loop(
                harness,
                executor,
                session,
                input_fn=lambda _: next(messages),
                output_fn=output.append,
            )

            self.assertEqual(result, 0)
            self.assertEqual(len(session.state.turns), 0)
            self.assertTrue(any("Always use pytest" in line for line in output))
            self.assertEqual(manager.store.read(MemoryScope.USER), "")

    def test_session_capacity_drops_old_turns_before_explicit_notes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = MemoryManager(
                MemoryStore(Path(directory), "bounded", session_max_chars=2_000)
            )
            journal = Journal()
            manager.remember(MemoryScope.SESSION, "Pinned architectural decision.")
            for index in range(12):
                manager.record_turn(
                    f"turn-{index + 1:04d}",
                    f"request {index} " + "r" * 300,
                    f"answer {index} " + "a" * 700,
                    journal,
                )

            content = manager.store.read(MemoryScope.SESSION)
            self.assertIn("Pinned architectural decision", content)
            self.assertNotIn("## turn-0001", content)
            self.assertIn("## turn-0012", content)


if __name__ == "__main__":
    unittest.main()
