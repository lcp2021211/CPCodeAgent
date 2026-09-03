from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, ClassVar

from cpcodeagent.executor import ExecutionEnv, LocalExecutor
from cpcodeagent.journal import EventKind, Journal
from cpcodeagent.policy import RunPolicy
from cpcodeagent.recovery import ActionLedger, ActionState, RecoveryMode
from cpcodeagent.tools import Tool, ToolExecution, ToolRuntime
from cpcodeagent.types import Action, Capability, ToolCall


class ProbeTool(Tool):
    description = "Probe scheduler behavior"
    input_schema: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}

    def __init__(self, name: str, read_only: bool, state: dict):
        self.name = name
        self.read_only = read_only
        self.state = state

    def classify(self, arguments, env: ExecutionEnv) -> Action:
        capability = Capability.READ if self.read_only else Capability.WORKSPACE_WRITE
        return Action(frozenset({capability}), (f"workspace:{env.workspace}",), self.read_only)

    def execute(self, arguments, env: ExecutionEnv) -> ToolExecution:
        if self.read_only:
            with self.state["lock"]:
                self.state["active"] += 1
                self.state["max_active"] = max(self.state["max_active"], self.state["active"])
            time.sleep(0.05)
            with self.state["lock"]:
                self.state["active"] -= 1
                self.state["reads_done"] += 1
        else:
            self.state["write_saw"] = self.state["reads_done"]
        return ToolExecution(True, self.name)


class RuntimeTests(unittest.TestCase):
    def test_duplicate_calls_in_one_response_execute_only_once(self) -> None:
        state = {
            "lock": threading.Lock(),
            "active": 0,
            "max_active": 0,
            "reads_done": 0,
            "write_saw": 0,
        }
        runtime = ToolRuntime([ProbeTool("read", True, state)])
        journal = Journal()
        with tempfile.TemporaryDirectory() as directory:
            results = runtime.execute_batch(
                [
                    ToolCall("first", "read", {"path": "same"}),
                    ToolCall("duplicate", "read", {"path": "same"}),
                ],
                RunPolicy(),
                LocalExecutor(directory),
                journal,
                response_seq=4,
            )

        self.assertTrue(results[0].ok)
        self.assertEqual(results[1].error, "DUPLICATE_CALL")
        self.assertEqual(state["reads_done"], 1)
        self.assertEqual(len(journal.find(EventKind.TOOL_CALL)), 2)
        self.assertEqual(len(journal.find(EventKind.TOOL_RESULT)), 2)

    def test_read_batches_are_parallel_and_write_is_a_barrier(self) -> None:
        state = {
            "lock": threading.Lock(),
            "active": 0,
            "max_active": 0,
            "reads_done": 0,
            "write_saw": 0,
        }
        runtime = ToolRuntime(
            [ProbeTool("read_a", True, state), ProbeTool("read_b", True, state), ProbeTool("write", False, state)]
        )
        with tempfile.TemporaryDirectory() as directory:
            results = runtime.execute_batch(
                [
                    ToolCall("1", "read_a", {}),
                    ToolCall("2", "read_b", {}),
                    ToolCall("3", "write", {}),
                ],
                RunPolicy(),
                LocalExecutor(directory),
                Journal(),
            )

        self.assertTrue(all(result.ok for result in results))
        self.assertGreaterEqual(state["max_active"], 2)
        self.assertEqual(state["write_saw"], 2)

    def test_policy_denies_path_escape_before_execution(self) -> None:
        from cpcodeagent.builtin_tools import ReadFileTool

        runtime = ToolRuntime([ReadFileTool()])
        with tempfile.TemporaryDirectory() as directory:
            result = runtime.execute_batch(
                [ToolCall("1", "read_file", {"path": "../secret.txt"})],
                RunPolicy(),
                LocalExecutor(directory),
                Journal(),
            )[0]
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "CLASSIFICATION_FAILED")

    def test_read_only_policy_denies_write(self) -> None:
        from cpcodeagent.builtin_tools import WriteFileTool

        runtime = ToolRuntime([WriteFileTool()])
        with tempfile.TemporaryDirectory() as directory:
            result = runtime.execute_batch(
                [ToolCall("1", "write_file", {"path": "x.txt", "content": "x"})],
                RunPolicy(workspace_write=False),
                LocalExecutor(directory),
                Journal(),
            )[0]
            self.assertFalse(Path(directory, "x.txt").exists())
        self.assertEqual(result.error, "POLICY_DENIED")

    def test_file_action_has_durable_lifecycle_and_observation(self) -> None:
        from cpcodeagent.builtin_tools import WriteFileTool

        runtime = ToolRuntime([WriteFileTool()])
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal()
            result = runtime.execute_batch(
                [ToolCall("write-1", "write_file", {"path": "x.txt", "content": "x"})],
                RunPolicy(),
                LocalExecutor(directory),
                journal,
                response_seq=7,
            )[0]

            lifecycle = [
                event.kind
                for event in journal.events
                if event.kind
                in {EventKind.TOOL_CALL, EventKind.TOOL_STARTED, EventKind.TOOL_RESULT}
            ]
            record = ActionLedger.from_journal(journal).records[0]

        self.assertTrue(result.ok)
        self.assertEqual(
            lifecycle,
            [EventKind.TOOL_CALL, EventKind.TOOL_STARTED, EventKind.TOOL_RESULT],
        )
        self.assertEqual(record.state, ActionState.COMMITTED)
        self.assertEqual(record.response_seq, 7)
        self.assertEqual(record.contract.mode, RecoveryMode.VERIFY_FILES)
        self.assertEqual(record.commit.data["observation"]["changed_paths"], ["x.txt"])


if __name__ == "__main__":
    unittest.main()
