from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cpcodeagent.builtin_tools import build_default_runtime
from cpcodeagent.executor import ExecutionEnv, LocalExecutor
from cpcodeagent.journal import EventKind, Journal
from cpcodeagent.kernel import Harness
from cpcodeagent.model import ScriptedModel
from cpcodeagent.recovery import ActionLedger
from cpcodeagent.subagents import (
    DelegateTaskTool,
    SubagentMode,
    SubagentRunner,
    SubagentStatus,
)
from cpcodeagent.tools import ToolRuntime
from cpcodeagent.types import ModelResponse, RunStatus, ToolCall, ToolResult


def _submit(call_id: str, summary: str = "Found the cause.") -> ToolCall:
    return ToolCall(
        call_id,
        "submit_result",
        {
            "summary": summary,
            "evidence": ["module.py:10 contains the decisive branch"],
            "recommendation": "Update that branch and verify the focused behavior.",
        },
    )


def _tool_names(request: tuple) -> set[str]:
    return {
        str(item["function"]["name"])
        for item in request[1]
        if isinstance(item, dict) and isinstance(item.get("function"), dict)
    }


class SubagentTests(unittest.TestCase):
    def test_child_tool_surface_forbids_grandchildren(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            model = ScriptedModel(
                [
                    ModelResponse(
                        tool_calls=(
                            ToolCall(
                                "nested",
                                "delegate_task",
                                {"task": "create a grandchild", "mode": "inspect"},
                            ),
                        )
                    ),
                    ModelResponse(tool_calls=(_submit("submit"),)),
                    ModelResponse(content="Submitted."),
                ]
            )
            runner = SubagentRunner(model, root / "children")

            result = runner.run(
                "Inspect the relevant module.",
                SubagentMode.INSPECT,
                LocalExecutor(workspace),
                "a" * 32,
            )

            names = _tool_names(model.requests[0])
            self.assertNotIn("delegate_task", names)
            self.assertNotIn("write_file", names)
            self.assertNotIn("run_command", names)
            self.assertIn("submit_result", names)
            self.assertEqual(result.status, SubagentStatus.COMPLETED)
            child = Journal(root / "children" / ("a" * 32) / "journal.jsonl")
            failures = [
                ToolResult.from_dict(event.data["result"])
                for event in child.find(EventKind.TOOL_RESULT)
            ]
            self.assertEqual(failures[0].error, "UNKNOWN_TOOL")

    def test_parent_context_receives_only_structured_child_result(self) -> None:
        marker = "PRIVATE_CHILD_TRAJECTORY_" + ("x" * 4_000)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            workspace.joinpath("secret.txt").write_text(marker, encoding="utf-8")
            child_model = ScriptedModel(
                [
                    ModelResponse(
                        tool_calls=(
                            ToolCall("read-secret", "read_file", {"path": "secret.txt"}),
                        )
                    ),
                    ModelResponse(
                        tool_calls=(_submit("submit", "The secret file is unexpectedly large."),)
                    ),
                    ModelResponse(content="Submitted."),
                ],
                name="child",
            )
            runner = SubagentRunner(child_model, root / "children")
            tools = build_default_runtime()
            tools.register(DelegateTaskTool(runner))
            parent_model = ScriptedModel(
                [
                    ModelResponse(
                        tool_calls=(
                            ToolCall(
                                "delegate",
                                "delegate_task",
                                {"task": "Inspect secret.txt.", "mode": "inspect"},
                            ),
                        )
                    ),
                    ModelResponse(content="I reviewed the child result."),
                ],
                name="parent",
            )
            parent_journal = Journal(root / "parent.jsonl")
            executor = LocalExecutor(workspace)
            before = executor.snapshot().revision

            outcome = Harness(parent_model, tools).run(
                "Delegate inspection of secret.txt.", executor, parent_journal, "parent-run"
            )

            self.assertEqual(outcome.status, RunStatus.SUCCEEDED)
            self.assertEqual(before, executor.snapshot().revision)
            parent_context = json.dumps(parent_model.requests[1][0], ensure_ascii=False)
            self.assertIn("The secret file is unexpectedly large.", parent_context)
            self.assertNotIn("PRIVATE_CHILD_TRAJECTORY", parent_context)
            self.assertEqual(len(parent_journal.find(EventKind.MODEL_RESPONSE)), 2)
            child_journal = next((root / "children").glob("*/journal.jsonl"))
            self.assertEqual(len(Journal(child_journal).find(EventKind.MODEL_RESPONSE)), 3)
            delegate_result = next(
                ToolResult.from_dict(event.data["result"])
                for event in parent_journal.find(EventKind.TOOL_RESULT)
                if ToolResult.from_dict(event.data["result"]).call_id == "delegate"
            )
            self.assertEqual(json.loads(delegate_result.output)["status"], "completed")

    def test_patch_mode_changes_only_the_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            target = workspace / "module.py"
            target.write_text("value = 1\n", encoding="utf-8")
            model = ScriptedModel(
                [
                    ModelResponse(
                        tool_calls=(
                            ToolCall(
                                "write",
                                "write_file",
                                {"path": "module.py", "content": "value = 2\n"},
                            ),
                        )
                    ),
                    ModelResponse(tool_calls=(_submit("submit", "Prepared the isolated edit."),)),
                    ModelResponse(content="Submitted."),
                ]
            )
            runner = SubagentRunner(model, root / "children")
            executor = LocalExecutor(workspace)
            before = executor.snapshot().revision

            result = runner.run(
                "Change module.py to value 2.",
                SubagentMode.PATCH,
                executor,
                "b" * 32,
            )

            self.assertEqual(target.read_text(encoding="utf-8"), "value = 1\n")
            self.assertEqual(before, executor.snapshot().revision)
            overlay = root / "children" / ("b" * 32) / "workspace" / "module.py"
            self.assertEqual(overlay.read_text(encoding="utf-8"), "value = 2\n")
            self.assertEqual(result.artifact_id, "b" * 32)
            self.assertEqual(result.changed_paths, ("module.py",))
            patch = json.loads(
                (root / "children" / ("b" * 32) / "patch.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(patch["changes"][0]["path"], "module.py")
            self.assertNotIn("run_command", _tool_names(model.requests[0]))

    def test_recovery_reuses_completed_child_by_action_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            model = ScriptedModel(
                [
                    ModelResponse(tool_calls=(_submit("submit"),)),
                    ModelResponse(content="Submitted."),
                ]
            )
            runner = SubagentRunner(model, root / "children")
            executor = LocalExecutor(workspace)
            action_id = "c" * 32
            expected = runner.run(
                "Inspect the project.", SubagentMode.INSPECT, executor, action_id
            )
            request_count = len(model.requests)
            tool = DelegateTaskTool(runner)
            runtime = ToolRuntime((tool,))
            arguments = {"task": "Inspect the project.", "mode": "inspect"}
            call = ToolCall("delegate", "delegate_task", arguments)
            action = tool.classify(arguments, ExecutionEnv(executor))
            contract = tool.effect_contract(
                arguments, action, ExecutionEnv(executor), action_id
            )
            journal = Journal()
            journal.append(
                EventKind.TOOL_CALL,
                {
                    "action_id": action_id,
                    "call": call.to_dict(),
                    "action": action.to_dict(),
                    "authorized": True,
                    "workspace_revision": executor.snapshot().revision,
                    "response_seq": 4,
                    "effect_contract": contract.to_dict(),
                    "prepared_result": None,
                },
            )
            journal.append(
                EventKind.TOOL_STARTED,
                {
                    "action_id": action_id,
                    "call_id": call.id,
                    "workspace_revision": executor.snapshot().revision,
                    "recovery_attempt": False,
                },
            )

            result = runtime.recover_pending(
                ActionLedger.from_journal(journal).pending()[0], executor, journal
            )

            self.assertTrue(result.ok)
            self.assertEqual(SubagentStatus(json.loads(result.output)["status"]), expected.status)
            self.assertEqual(len(model.requests), request_count)
            commit = journal.last(EventKind.TOOL_RESULT)
            self.assertEqual(commit.data["recovered"], "retried")
            self.assertEqual(len(journal.find(EventKind.TOOL_RESULT)), 1)


if __name__ == "__main__":
    unittest.main()
