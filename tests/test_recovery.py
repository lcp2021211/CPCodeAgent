from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, ClassVar

from cpcodeagent.builtin_tools import WriteFileTool, build_default_runtime
from cpcodeagent.executor import ExecutionEnv, LocalExecutor
from cpcodeagent.journal import EventKind, Journal
from cpcodeagent.kernel import Harness
from cpcodeagent.model import ScriptedModel
from cpcodeagent.recovery import (
    ActionLedger,
    ActionState,
    EffectContract,
    EffectState,
    FileCondition,
    content_digest,
)
from cpcodeagent.tools import Tool, ToolExecution, ToolRuntime
from cpcodeagent.types import Action, Capability, ModelResponse, RunStatus, ToolCall


class InterruptingWriteFileTool(WriteFileTool):
    def __init__(self, point: str):
        self.point = point

    def execute(self, arguments: dict[str, Any], env: ExecutionEnv) -> ToolExecution:
        if self.point == "before":
            raise KeyboardInterrupt
        if self.point == "conflict":
            env.executor.write_text(
                arguments["path"],
                "concurrent value",
                operation_id=env.action_id,
            )
            raise KeyboardInterrupt
        super().execute(arguments, env)
        raise KeyboardInterrupt from None


class OpaqueSideEffectTool(Tool):
    name = "opaque_side_effect"
    description = "A test-only opaque side effect."
    input_schema: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}

    def __init__(self, runs: list[str], interrupt: bool = True):
        self.runs = runs
        self.interrupt = interrupt

    def classify(self, arguments: dict[str, Any], env: ExecutionEnv) -> Action:
        return Action(
            frozenset({Capability.WORKSPACE_WRITE}),
            (f"workspace:{env.workspace}",),
            False,
        )

    def execute(self, arguments: dict[str, Any], env: ExecutionEnv) -> ToolExecution:
        self.runs.append("started")
        if self.interrupt:
            raise KeyboardInterrupt
        return ToolExecution(True, "completed")


class RecoveryTests(unittest.TestCase):
    def test_partial_multi_file_transition_is_resumable_until_a_path_diverges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executor = LocalExecutor(directory)
            Path(directory, "a.txt").write_text("old-a", encoding="utf-8")
            Path(directory, "b.txt").write_text("old-b", encoding="utf-8")
            contract = EffectContract.file_transitions(
                (
                    FileCondition("a.txt", content_digest("old-a")),
                    FileCondition("b.txt", content_digest("old-b")),
                ),
                (
                    FileCondition("a.txt", content_digest("new-a")),
                    FileCondition("b.txt", content_digest("new-b")),
                ),
            )

            Path(directory, "a.txt").write_text("new-a", encoding="utf-8")
            self.assertEqual(contract.inspect(executor), EffectState.NOT_APPLIED)
            Path(directory, "b.txt").write_text("external", encoding="utf-8")
            self.assertEqual(contract.inspect(executor), EffectState.CONFLICT)

    def test_unknown_commit_is_not_replayed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            executor = LocalExecutor(workspace)
            journal = Journal()
            journal.append(
                EventKind.INPUT,
                {
                    "content": "Write value.txt",
                    "source": "user",
                    "run_id": "uncertain-1",
                    "workspace": str(workspace),
                },
            )
            call = ToolCall("write-1", "write_file", {"path": "value.txt", "content": "x"})
            journal.append(
                EventKind.MODEL_RESPONSE,
                {"response": ModelResponse(tool_calls=(call,)).to_dict()},
            )
            baseline = executor.snapshot().revision
            action = Action(
                frozenset({Capability.WORKSPACE_WRITE}),
                (f"file:{workspace / 'value.txt'}",),
                False,
            )
            journal.append(
                EventKind.TOOL_CALL,
                {
                    "call": call.to_dict(),
                    "action": action.to_dict(),
                    "authorized": True,
                    "workspace_revision": baseline,
                },
            )
            # Simulate a crash after the side effect but before ToolResult was committed.
            (workspace / "value.txt").write_text("x")
            harness = Harness(ScriptedModel([]), build_default_runtime())

            outcome = harness.resume(executor, journal)

            self.assertEqual(outcome.status, RunStatus.NEEDS_CONFIRMATION)
            self.assertIn("unsafe", outcome.answer)

    def test_applied_atomic_write_is_verified_without_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            journal = Journal(workspace / "session.jsonl")
            call = ToolCall("write-1", "write_file", {"path": "value.txt", "content": "x"})
            interrupted = Harness(
                ScriptedModel([ModelResponse(tool_calls=(call,))]),
                ToolRuntime([InterruptingWriteFileTool("after")]),
            )

            with self.assertRaises(KeyboardInterrupt):
                interrupted.run("Write value.txt", LocalExecutor(workspace), journal, "recover-1")

            pending = ActionLedger.from_journal(journal).pending()[0]
            self.assertEqual(pending.state, ActionState.STARTED)
            outcome = Harness(
                ScriptedModel([ModelResponse(content="Recovered.")]),
                build_default_runtime(),
            ).resume(LocalExecutor(workspace), journal)

            self.assertEqual(outcome.status, RunStatus.SUCCEEDED)
            self.assertEqual((workspace / "value.txt").read_text(), "x")
            record = ActionLedger.from_journal(journal).records[0]
            self.assertEqual(record.state, ActionState.COMMITTED)
            self.assertEqual(len(record.starts), 1)
            self.assertEqual(record.commit.data["recovered"], "effect_verified")

    def test_unapplied_atomic_write_is_retried_from_recorded_precondition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            journal = Journal(workspace / "session.jsonl")
            call = ToolCall("write-1", "write_file", {"path": "value.txt", "content": "x"})
            interrupted = Harness(
                ScriptedModel([ModelResponse(tool_calls=(call,))]),
                ToolRuntime([InterruptingWriteFileTool("before")]),
            )

            with self.assertRaises(KeyboardInterrupt):
                interrupted.run("Write value.txt", LocalExecutor(workspace), journal, "recover-2")

            pending = ActionLedger.from_journal(journal).pending()[0]
            staged = workspace / f".value.txt.cpcodeagent-{pending.action_id}.tmp"
            staged.write_text("unfinished staging bytes")

            outcome = Harness(
                ScriptedModel([ModelResponse(content="Recovered.")]),
                build_default_runtime(),
            ).resume(LocalExecutor(workspace), journal)

            self.assertEqual(outcome.status, RunStatus.SUCCEEDED)
            self.assertEqual((workspace / "value.txt").read_text(), "x")
            self.assertFalse(staged.exists())
            record = ActionLedger.from_journal(journal).records[0]
            self.assertEqual(len(record.starts), 2)
            self.assertEqual(record.commit.data["recovered"], "retried")

    def test_divergent_file_is_never_overwritten_during_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            journal = Journal(workspace / "session.jsonl")
            call = ToolCall("write-1", "write_file", {"path": "value.txt", "content": "x"})
            interrupted = Harness(
                ScriptedModel([ModelResponse(tool_calls=(call,))]),
                ToolRuntime([InterruptingWriteFileTool("conflict")]),
            )

            with self.assertRaises(KeyboardInterrupt):
                interrupted.run("Write value.txt", LocalExecutor(workspace), journal, "recover-3")

            outcome = Harness(ScriptedModel([]), build_default_runtime()).resume(
                LocalExecutor(workspace), journal
            )

            self.assertEqual(outcome.status, RunStatus.NEEDS_CONFIRMATION)
            self.assertEqual((workspace / "value.txt").read_text(), "concurrent value")
            self.assertEqual(
                ActionLedger.from_journal(journal).records[0].result.error,
                "RECOVERY_CONFLICT",
            )

    def test_started_opaque_side_effect_is_never_replayed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            journal = Journal(workspace / "session.jsonl")
            runs: list[str] = []
            call = ToolCall("opaque-1", "opaque_side_effect", {})
            interrupted = Harness(
                ScriptedModel([ModelResponse(tool_calls=(call,))]),
                ToolRuntime([OpaqueSideEffectTool(runs)]),
            )

            with self.assertRaises(KeyboardInterrupt):
                interrupted.run(
                    "Perform opaque action", LocalExecutor(workspace), journal, "recover-4"
                )

            outcome = Harness(
                ScriptedModel([]),
                ToolRuntime([OpaqueSideEffectTool(runs)]),
            ).resume(LocalExecutor(workspace), journal)

            self.assertEqual(outcome.status, RunStatus.NEEDS_CONFIRMATION)
            self.assertEqual(runs, ["started"])
            self.assertEqual(
                ActionLedger.from_journal(journal).records[0].result.error,
                "UNKNOWN_COMMIT",
            )

    def test_uncertain_action_blocks_later_unstarted_calls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            journal = Journal(workspace / "session.jsonl")
            runs: list[str] = []
            calls = (
                ToolCall("opaque-1", "opaque_side_effect", {}),
                ToolCall("write-2", "write_file", {"path": "later.txt", "content": "x"}),
            )
            interrupted = Harness(
                ScriptedModel([ModelResponse(tool_calls=calls)]),
                ToolRuntime([OpaqueSideEffectTool(runs), WriteFileTool()]),
            )

            with self.assertRaises(KeyboardInterrupt):
                interrupted.run("Perform both actions", LocalExecutor(workspace), journal, "recover-7")

            outcome = Harness(
                ScriptedModel([]),
                ToolRuntime(
                    [OpaqueSideEffectTool(runs, interrupt=False), WriteFileTool()]
                ),
            ).resume(LocalExecutor(workspace), journal)

            self.assertEqual(outcome.status, RunStatus.NEEDS_CONFIRMATION)
            self.assertEqual(runs, ["started"])
            self.assertFalse((workspace / "later.txt").exists())
            self.assertEqual(len(ActionLedger.from_journal(journal).records), 1)

    def test_durable_intent_without_started_marker_executes_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            executor = LocalExecutor(workspace)
            journal = Journal(workspace / "session.jsonl")
            call = ToolCall("opaque-1", "opaque_side_effect", {})
            harness = Harness(ScriptedModel([]), ToolRuntime())
            harness.start_session(executor, journal, "recover-5")
            turn = journal.append(
                EventKind.INPUT,
                {
                    "content": "Perform opaque action",
                    "source": "user",
                    "turn_id": "turn-0001",
                },
            )
            response = journal.append(
                EventKind.MODEL_RESPONSE,
                {"response": ModelResponse(tool_calls=(call,)).to_dict()},
            )
            action = Action(
                frozenset({Capability.WORKSPACE_WRITE}),
                (f"workspace:{workspace}",),
                False,
            )
            journal.append(
                EventKind.TOOL_CALL,
                {
                    "action_id": "intent-only",
                    "call": call.to_dict(),
                    "action": action.to_dict(),
                    "authorized": True,
                    "workspace_revision": executor.snapshot().revision,
                    "response_seq": response.seq,
                    "effect_contract": EffectContract.manual().to_dict(),
                    "prepared_result": None,
                },
            )
            self.assertGreater(response.seq, turn.seq)
            runs: list[str] = []

            outcome = Harness(
                ScriptedModel([ModelResponse(content="Completed.")]),
                ToolRuntime([OpaqueSideEffectTool(runs, interrupt=False)]),
            ).resume(executor, journal)

            self.assertEqual(outcome.status, RunStatus.SUCCEEDED)
            self.assertEqual(runs, ["started"])
            record = ActionLedger.from_journal(journal).records[0]
            self.assertEqual(len(record.starts), 1)
            self.assertTrue(record.result.ok)

    def test_committed_final_model_response_is_not_requested_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            executor = LocalExecutor(workspace)
            journal = Journal(workspace / "session.jsonl")
            harness = Harness(ScriptedModel([]), build_default_runtime())
            harness.start_session(executor, journal, "recover-6")
            journal.append(
                EventKind.INPUT,
                {
                    "content": "Answer once",
                    "source": "user",
                    "turn_id": "turn-0001",
                },
            )
            journal.append(
                EventKind.MODEL_RESPONSE,
                {"response": ModelResponse(content="Durable answer.").to_dict()},
            )

            outcome = harness.resume(executor, journal)

            self.assertEqual(outcome.status, RunStatus.SUCCEEDED)
            self.assertEqual(outcome.answer, "Durable answer.")
            self.assertEqual(len(journal.find(EventKind.MODEL_RESPONSE)), 1)


if __name__ == "__main__":
    unittest.main()
