from __future__ import annotations

import tempfile
import unittest

from cpcodeagent.builtin_tools import build_default_runtime
from cpcodeagent.context import ContextEngine
from cpcodeagent.executor import LocalExecutor
from cpcodeagent.journal import EventKind, Journal
from cpcodeagent.kernel import Harness
from cpcodeagent.model import ScriptedModel
from cpcodeagent.planning import PlanStatus, validate_plan_items
from cpcodeagent.policy import RunPolicy
from cpcodeagent.recovery import ActionLedger, RecoveryMode
from cpcodeagent.types import Capability, ModelResponse, RunStatus, ToolCall


def _items(first: str, second: str = "pending") -> list[dict[str, str]]:
    return [
        {"id": "inspect", "text": "Inspect the relevant code", "status": first},
        {"id": "verify", "text": "Run focused verification", "status": second},
    ]


class PlanningTests(unittest.TestCase):
    def test_plan_validation_allows_only_one_active_item(self) -> None:
        with self.assertRaisesRegex(ValueError, "Only one"):
            validate_plan_items(_items("in_progress", "in_progress"))

        items = validate_plan_items(_items("completed", "in_progress"))

        self.assertEqual(items[0].status, PlanStatus.COMPLETED)
        self.assertEqual(items[1].status, PlanStatus.IN_PROGRESS)

    def test_committed_plan_is_pinned_and_replayed_from_the_journal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal()
            journal.append(
                EventKind.INPUT,
                {"content": "Complex task", "source": "user", "turn_id": "turn-0001"},
            )
            call = ToolCall("plan-1", "plan_write", {"items": _items("in_progress")})
            response = journal.append(
                EventKind.MODEL_RESPONSE,
                {"response": ModelResponse(tool_calls=(call,)).to_dict()},
            )
            result = build_default_runtime().execute_batch(
                (call,),
                RunPolicy(workspace_write=False),
                LocalExecutor(directory),
                journal,
                response_seq=response.seq,
            )[0]

            engine = ContextEngine()
            state = engine.rebuild(journal)
            view = engine.update(
                state, journal, "Complex task", directory, "read only"
            ).view
            record = ActionLedger.from_journal(journal).records[0]

            self.assertTrue(result.ok)
            self.assertEqual(record.action.capabilities, frozenset({Capability.RUNTIME_WRITE}))
            self.assertEqual(record.contract.mode, RecoveryMode.RETRY_SAFE)
            self.assertIsNotNone(state.plan)
            self.assertIn("Current execution plan", str(view.messages))
            self.assertIn("Inspect the relevant code", str(view.messages))

            replayed = engine.rebuild(journal)
            self.assertEqual(replayed.plan, state.plan)

    def test_new_user_turn_resets_the_previous_turn_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal()
            journal.append(
                EventKind.INPUT,
                {"content": "First", "source": "user", "turn_id": "turn-0001"},
            )
            call = ToolCall("plan-1", "plan_write", {"items": _items("in_progress")})
            response = journal.append(
                EventKind.MODEL_RESPONSE,
                {"response": ModelResponse(tool_calls=(call,)).to_dict()},
            )
            build_default_runtime().execute_batch(
                (call,), RunPolicy(), LocalExecutor(directory), journal, response_seq=response.seq
            )
            engine = ContextEngine()
            state = engine.rebuild(journal)
            self.assertIsNotNone(state.plan)

            journal.append(
                EventKind.INPUT,
                {"content": "Second", "source": "user", "turn_id": "turn-0002"},
            )
            engine.update(state, journal, "Second", directory, "write allowed")

            self.assertIsNone(state.plan)
            self.assertEqual(state.turn_id, "turn-0002")

    def test_stale_plan_reminder_is_ephemeral_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal()
            journal.append(
                EventKind.INPUT,
                {"content": "Complex", "source": "user", "turn_id": "turn-0001"},
            )
            call = ToolCall("plan-1", "plan_write", {"items": _items("in_progress")})
            response = journal.append(
                EventKind.MODEL_RESPONSE,
                {"response": ModelResponse(tool_calls=(call,)).to_dict()},
            )
            build_default_runtime().execute_batch(
                (call,), RunPolicy(), LocalExecutor(directory), journal, response_seq=response.seq
            )
            for index in range(3):
                journal.append(
                    EventKind.MODEL_RESPONSE,
                    {"response": ModelResponse(content=f"step {index}").to_dict()},
                )
            before = journal.events
            view = ContextEngine().build(
                journal, "Complex", directory, "write allowed"
            )

            self.assertIn("Plan reminder", str(view.messages))
            self.assertEqual(journal.events, before)

    def test_incomplete_plan_blocks_final_answer_until_plan_is_completed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            initial = ToolCall(
                "plan-initial", "plan_write", {"items": _items("in_progress")}
            )
            completed = ToolCall(
                "plan-complete", "plan_write", {"items": _items("completed", "completed")}
            )
            journal = Journal()
            model = ScriptedModel(
                [
                    ModelResponse(tool_calls=(initial,)),
                    ModelResponse(content="Premature completion."),
                    ModelResponse(tool_calls=(completed,)),
                    ModelResponse(content="Plan completed."),
                ]
            )
            harness = Harness(model, build_default_runtime())

            outcome = harness.run(
                "Perform a complex task", LocalExecutor(directory), journal, "plan-run"
            )

            guards = [
                event
                for event in journal.find(EventKind.INPUT)
                if event.data.get("source") == "plan_guard"
            ]
            self.assertEqual(outcome.status, RunStatus.SUCCEEDED)
            self.assertEqual(outcome.answer, "Plan completed.")
            self.assertEqual(outcome.steps, 4)
            self.assertEqual(len(guards), 1)
            self.assertIn("unfinished items", guards[0].data["content"])


if __name__ == "__main__":
    unittest.main()
