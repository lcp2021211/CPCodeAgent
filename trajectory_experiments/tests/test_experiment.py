from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cpcodeagent.executor import ExecutionError
from cpcodeagent.model import ResilientModel, ScriptedModel, TransientModelError
from cpcodeagent.types import ModelResponse
from trajectory_experiments.container_runtime import safe_pattern, safe_relative
from trajectory_experiments.recording import (
    RecordedModelGroup,
    RecordingModel,
    RecordingState,
    Redactor,
)
from trajectory_experiments.run_trajectories import (
    archive_previous_attempt,
    is_combined_instance,
    summary_has_runner_error,
)


class FailOnceModel:
    name = "flaky"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, tools, on_text=None):
        self.calls += 1
        if self.calls == 1:
            raise TransientModelError("temporary")
        return ModelResponse(content="recovered", prompt_tokens=3, completion_tokens=1)


class RecordingTests(unittest.TestCase):
    def test_recording_model_saves_exact_request_and_redacts_secret(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "calls.jsonl"
            model = RecordingModel(
                ScriptedModel([ModelResponse(content="done", prompt_tokens=2)]),
                output,
                {"api_key": "secret-value"},
                Redactor(["secret-value"]),
            )
            response = model.complete(
                [{"role": "user", "content": "secret-value"}],
                [{"type": "function", "function": {"name": "read_file"}}],
            )

            self.assertEqual(response.content, "done")
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved["request"]["messages"][0]["content"], "<redacted>")
            self.assertEqual(saved["request_metadata"]["api_key"], "<redacted>")
            self.assertEqual(model.totals()["prompt_tokens"], 2)

    def test_workspace_paths_reject_escape(self) -> None:
        self.assertEqual(str(safe_relative("src/main.py")), "src/main.py")
        self.assertEqual(safe_pattern("**/*.py"), "**/*.py")
        with self.assertRaises(ExecutionError):
            safe_relative("../gold.patch")
        with self.assertRaises(ExecutionError):
            safe_pattern("/tmp/*")

    def test_combined_task_detection(self) -> None:
        self.assertTrue(
            is_combined_instance("owner__repo.commit.combine_file__example")
        )
        self.assertFalse(is_combined_instance("owner__repo.commit.mutant-name"))

    def test_each_retry_is_recorded_as_a_separate_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "calls.jsonl"
            redactor = Redactor()
            state = RecordingState(output, redactor)
            recorded = RecordingModel(
                FailOnceModel(), output, {"provider_role": "primary"}, redactor, state
            )
            resilient = ResilientModel(recorded, max_attempts=2, sleeper=lambda _: None)
            model = RecordedModelGroup(resilient, state)

            response = model.complete([{"role": "user", "content": "go"}], [])
            records = [json.loads(line) for line in output.read_text().splitlines()]

            self.assertEqual(response.content, "recovered")
            self.assertEqual([item["call_index"] for item in records], [0, 1])
            self.assertEqual(records[0]["error"]["type"], "TransientModelError")
            self.assertEqual(records[1]["response"]["content"], "recovered")
            self.assertEqual(model.totals()["model_calls"], 2)

    def test_runner_error_detection_covers_agent_and_grader(self) -> None:
        self.assertTrue(summary_has_runner_error({"agent": {"status": "runner_error"}}))
        self.assertTrue(
            summary_has_runner_error(
                {
                    "agent": {"status": "succeeded"},
                    "evaluation": {"status": "runner_error"},
                }
            )
        )
        self.assertFalse(
            summary_has_runner_error(
                {
                    "agent": {"status": "succeeded"},
                    "evaluation": {"status": "completed"},
                }
            )
        )

    def test_failed_attempt_is_archived_before_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "task"
            task_dir.mkdir()
            (task_dir / "summary.json").write_text('{"agent":{"status":"runner_error"}}')
            (task_dir / "model_calls.jsonl").write_text("failed attempt\n")

            archived = archive_previous_attempt(task_dir)

            self.assertIsNotNone(archived)
            assert archived is not None
            self.assertTrue((archived / "summary.json").exists())
            self.assertTrue((archived / "model_calls.jsonl").exists())
            self.assertFalse((task_dir / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
