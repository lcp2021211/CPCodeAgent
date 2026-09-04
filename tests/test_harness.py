from __future__ import annotations

import tempfile
import unittest
from itertools import pairwise
from pathlib import Path

from cpcodeagent.builtin_tools import build_default_runtime
from cpcodeagent.context import ContextEngine
from cpcodeagent.executor import LocalExecutor
from cpcodeagent.journal import EventKind, Journal
from cpcodeagent.kernel import Harness
from cpcodeagent.model import ScriptedModel
from cpcodeagent.skills import SkillRegistry
from cpcodeagent.types import ModelResponse, RunStatus, ToolCall, Verification
from cpcodeagent.verifier import Verifier


class FileVerifier(Verifier):
    def verify(self, executor: LocalExecutor) -> Verification:
        target = executor.workspace / "answer.py"
        return Verification(target.read_text() == "answer = 42\n", "answer.py has wrong content")


class CountingContextEngine(ContextEngine):
    def __init__(self):
        super().__init__()
        self.rebuild_calls = 0

    def rebuild(self, journal):
        self.rebuild_calls += 1
        return super().rebuild(journal)


class HarnessTests(unittest.TestCase):
    def test_stall_guard_state_is_rebuilt_from_the_journal(self) -> None:
        journal = Journal()
        turn = journal.append(
            EventKind.INPUT,
            {"content": "Inspect", "source": "user", "turn_id": "turn-0001"},
        )
        journal.append(
            EventKind.STALL_STATE,
            {
                "turn_id": "turn-0001",
                "fingerprint": "durable-fingerprint",
                "streak": 1,
                "warned": True,
                "workspace_revision": "revision-1",
                "guidance": None,
            },
        )

        progress = Harness._progress(journal, turn.seq)

        self.assertEqual(progress.stall_fingerprint, "durable-fingerprint")
        self.assertEqual(progress.stall_streak, 1)
        self.assertTrue(progress.stall_warned)

    def test_normal_agent_steps_reuse_one_hot_context_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = CountingContextEngine()
            model = ScriptedModel(
                [
                    ModelResponse(
                        tool_calls=(
                            ToolCall(
                                "write-hot",
                                "write_file",
                                {"path": "hot.txt", "content": "hot\n"},
                            ),
                        )
                    ),
                    ModelResponse(content="Done."),
                ]
            )
            harness = Harness(model, build_default_runtime(), context=context)

            outcome = harness.run(
                "Create hot.txt", LocalExecutor(directory), Journal(), "hot-context"
            )

            self.assertEqual(outcome.status, RunStatus.SUCCEEDED)
            self.assertEqual(context.rebuild_calls, 1)

    def test_end_to_end_write_verify_and_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            journal = Journal(workspace / "run.jsonl")
            model = ScriptedModel(
                [
                    ModelResponse(
                        tool_calls=(
                            ToolCall(
                                "write-1",
                                "write_file",
                                {"path": "answer.py", "content": "answer = 42\n"},
                            ),
                        )
                    ),
                    ModelResponse(content="Created and verified answer.py."),
                ]
            )
            harness = Harness(model, build_default_runtime(), verifier=FileVerifier())

            outcome = harness.run("Create answer.py", LocalExecutor(workspace), journal, "run-1")

            self.assertEqual(outcome.status, RunStatus.SUCCEEDED)
            self.assertEqual((workspace / "answer.py").read_text(), "answer = 42\n")
            reloaded = Journal(workspace / "run.jsonl")
            self.assertEqual(reloaded.last(EventKind.FINAL).data["status"], "succeeded")
            self.assertTrue(reloaded.find(EventKind.CHECKPOINT))

    def test_resume_executes_an_unstarted_tool_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            journal = Journal(workspace / "resume.jsonl")
            journal.append(
                EventKind.INPUT,
                {
                    "content": "Create resumed.txt",
                    "source": "user",
                    "run_id": "resume-1",
                    "workspace": str(workspace),
                },
            )
            response = ModelResponse(
                tool_calls=(
                    ToolCall(
                        "write-resume",
                        "write_file",
                        {"path": "resumed.txt", "content": "ok\n"},
                    ),
                )
            )
            journal.append(EventKind.MODEL_RESPONSE, {"response": response.to_dict()})
            model = ScriptedModel([ModelResponse(content="Recovered and completed.")])
            harness = Harness(model, build_default_runtime())

            outcome = harness.resume(LocalExecutor(workspace), journal)

            self.assertEqual(outcome.status, RunStatus.SUCCEEDED)
            self.assertEqual((workspace / "resumed.txt").read_text(), "ok\n")

    def test_progressive_skill_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            skill_dir = workspace / ".cpcodeagent" / "skills" / "focused"
            skill_dir.mkdir(parents=True)
            skill_dir.joinpath("SKILL.md").write_text(
                "---\nname: focused\ndescription: A focused workflow.\n"
                "requires-tools: [read_file]\n---\nAlways inspect first.\n"
            )
            skills = SkillRegistry([workspace / ".cpcodeagent" / "skills"])
            model = ScriptedModel(
                [
                    ModelResponse(
                        tool_calls=(ToolCall("skill-1", "use_skill", {"name": "focused"}),)
                    ),
                    ModelResponse(content="Done."),
                ]
            )
            harness = Harness(model, build_default_runtime(skills), skills=skills)

            harness.run("Inspect the project", LocalExecutor(workspace), Journal(), "skill-run")

            first_system = model.requests[0][0][0]["content"]
            second_system = model.requests[1][0][0]["content"]
            self.assertIn("A focused workflow", first_system)
            self.assertNotIn("Always inspect first", first_system)
            self.assertIn("Always inspect first", second_system)

    def test_repeated_no_progress_stops_after_one_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            responses = []
            for index, call_count in enumerate((1, 2, 4, 8)):
                responses.append(
                    ModelResponse(
                        tool_calls=tuple(
                            ToolCall(
                                f"read-{index}-{call_index}",
                                "read_file",
                                {"path": "missing.py"},
                            )
                            for call_index in range(call_count)
                        )
                    )
                )
            model = ScriptedModel(responses)
            harness = Harness(model, build_default_runtime())
            journal = Journal()

            outcome = harness.run(
                "Find a missing file", LocalExecutor(directory), journal, "loop-run"
            )

            self.assertEqual(outcome.status, RunStatus.FAILED)
            self.assertIn("no progress", outcome.answer)
            self.assertEqual(len(model.requests), 4)
            recovery_messages = model.requests[2][0]
            recovery_roles = [message["role"] for message in recovery_messages]
            self.assertNotIn(("tool", "user"), pairwise(recovery_roles))
            self.assertIn(
                "Correct the argument types",
                recovery_messages[-1]["content"],
            )
            duplicate_results = [
                event
                for event in journal.find(EventKind.TOOL_RESULT)
                if event.data["result"].get("error") == "DUPLICATE_CALL"
            ]
            self.assertEqual(len(duplicate_results), 11)
            stall = journal.last(EventKind.STALL_STATE)
            self.assertIsNotNone(stall)
            self.assertTrue(stall.data["warned"])


if __name__ == "__main__":
    unittest.main()
