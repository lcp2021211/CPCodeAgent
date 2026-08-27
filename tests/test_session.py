from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cpcodeagent.builtin_tools import build_default_runtime
from cpcodeagent.cli import build_parser, interactive_loop, load_environment
from cpcodeagent.executor import LocalExecutor
from cpcodeagent.journal import EventKind, Journal
from cpcodeagent.kernel import Harness
from cpcodeagent.model import ScriptedModel
from cpcodeagent.session import SessionState, SessionStore
from cpcodeagent.types import ModelResponse, RunStatus, ToolCall


class SessionTests(unittest.TestCase):
    def test_dotenv_configures_cli_and_shell_values_take_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "OPENAI_API_KEY=file-key\n"
                "OPENAI_BASE_URL=https://provider.example/v1\n"
                "CPCODEAGENT_MODEL=env-model\n"
                "CPCODEAGENT_FALLBACK_MODEL=env-fallback\n"
                "CPCODEAGENT_MAX_STEPS=7\n"
                "CPCODEAGENT_MAX_SECONDS=12.5\n"
                "CPCODEAGENT_MAX_TOKENS=9000\n"
                "CPCODEAGENT_JOURNAL_DIR=/tmp/env-journals\n"
                "CPCODEAGENT_MEMORY_DIR=/tmp/env-memory\n"
            )
            with patch.dict(os.environ, {"OPENAI_API_KEY": "shell-key"}, clear=True):
                loaded = load_environment(env_file)
                args = build_parser().parse_args(["--model", "cli-model"])

                self.assertEqual(loaded, env_file.resolve())
                self.assertEqual(args.api_key, "shell-key")
                self.assertEqual(args.base_url, "https://provider.example/v1")
                self.assertEqual(args.model, "cli-model")
                self.assertEqual(args.fallback_model, "env-fallback")
                self.assertEqual(args.max_steps, 7)
                self.assertEqual(args.max_seconds, 12.5)
                self.assertEqual(args.max_tokens, 9000)
                self.assertEqual(args.journal_dir, "/tmp/env-journals")
                self.assertEqual(args.memory_dir, "/tmp/env-memory")

    def test_multiple_turns_share_context_but_reset_turn_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            journal = Journal(workspace / "session.jsonl")
            model = ScriptedModel(
                [
                    ModelResponse(
                        content="The project has one module.",
                        prompt_tokens=10,
                        completion_tokens=5,
                    ),
                    ModelResponse(
                        content="I was referring to that module.",
                        prompt_tokens=20,
                        completion_tokens=7,
                    ),
                ]
            )
            harness = Harness(model, build_default_runtime())
            executor = LocalExecutor(workspace)

            first = harness.run("Inspect the project", executor, journal, "session-1")
            second = harness.send("What were you referring to?", executor, journal)

            self.assertEqual(first.steps, 1)
            self.assertEqual(first.tokens, 15)
            self.assertEqual(second.steps, 1)
            self.assertEqual(second.tokens, 27)
            self.assertEqual(second.turn_id, "turn-0002")

            second_messages = model.requests[1][0]
            history = "\n".join(str(message.get("content")) for message in second_messages)
            self.assertIn("Inspect the project", history)
            self.assertIn("The project has one module.", history)
            self.assertIn("What were you referring to?", history)

            state = SessionState.from_journal(Journal(workspace / "session.jsonl"))
            self.assertEqual(state.session_id, "session-1")
            self.assertEqual(state.executor, {"kind": "local"})
            self.assertEqual(len(state.turns), 2)
            self.assertIsNone(state.active_turn)
            self.assertTrue(all(turn.completed for turn in state.turns))

    def test_resume_recovers_only_the_latest_incomplete_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            executor = LocalExecutor(workspace)
            journal = Journal(workspace / "session.jsonl")
            Harness(
                ScriptedModel([ModelResponse(content="First turn complete.")]),
                build_default_runtime(),
            ).run("First turn", executor, journal, "session-2")

            journal.append(
                EventKind.INPUT,
                {
                    "content": "Create resumed.txt",
                    "source": "user",
                    "session_id": "session-2",
                    "turn_id": "turn-0002",
                },
            )
            call = ToolCall(
                "write-latest",
                "write_file",
                {"path": "resumed.txt", "content": "ok\n"},
            )
            journal.append(
                EventKind.MODEL_RESPONSE,
                {"response": ModelResponse(tool_calls=(call,)).to_dict()},
            )
            harness = Harness(
                ScriptedModel([ModelResponse(content="Latest turn recovered.")]),
                build_default_runtime(),
            )

            outcome = harness.resume(executor, journal)

            self.assertEqual(outcome.status, RunStatus.SUCCEEDED)
            self.assertEqual(outcome.turn_id, "turn-0002")
            self.assertEqual(outcome.steps, 2)
            self.assertEqual((workspace / "resumed.txt").read_text(), "ok\n")
            self.assertEqual(len(journal.find(EventKind.FINAL)), 2)

    def test_send_rejects_an_incomplete_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executor = LocalExecutor(directory)
            journal = Journal()
            harness = Harness(ScriptedModel([]), build_default_runtime())
            harness.start_session(executor, journal, "session-3")
            journal.append(
                EventKind.INPUT,
                {
                    "content": "Interrupted request",
                    "source": "user",
                    "turn_id": "turn-0001",
                },
            )

            with self.assertRaisesRegex(ValueError, "resume"):
                harness.send("Do not overlap", executor, journal)

    def test_repl_persists_multiple_messages_in_one_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = SessionStore(workspace / "sessions")
            session = store.create("chat-1")
            executor = LocalExecutor(workspace)
            harness = Harness(
                ScriptedModel(
                    [ModelResponse(content="answer one"), ModelResponse(content="answer two")]
                ),
                build_default_runtime(),
            )
            harness.start_session(executor, session.journal, session.session_id)
            messages = iter(["first question", "follow-up question", "/status", "/exit"])
            output: list[str] = []

            result = interactive_loop(
                harness,
                executor,
                session,
                input_fn=lambda _: next(messages),
                output_fn=output.append,
            )

            self.assertEqual(result, 0)
            self.assertEqual(len(store.open("chat-1").state.turns), 2)
            self.assertTrue(any("turns=2" in line for line in output))
            self.assertTrue(any("Session saved: chat-1" in line for line in output))

    def test_session_store_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(directory)
            with self.assertRaises(ValueError):
                store.create("../outside")


if __name__ == "__main__":
    unittest.main()
