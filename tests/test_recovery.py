from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cpcodeagent.builtin_tools import build_default_runtime
from cpcodeagent.executor import LocalExecutor
from cpcodeagent.journal import EventKind, Journal
from cpcodeagent.kernel import Harness
from cpcodeagent.model import ScriptedModel
from cpcodeagent.types import Action, Capability, ModelResponse, RunStatus, ToolCall


class RecoveryTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

