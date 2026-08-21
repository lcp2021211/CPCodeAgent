from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, ClassVar

from cpcodeagent.executor import ExecutionEnv, LocalExecutor
from cpcodeagent.journal import Journal
from cpcodeagent.policy import RunPolicy
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


if __name__ == "__main__":
    unittest.main()
