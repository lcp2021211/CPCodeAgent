"""The small THINK -> ACT -> CHECK -> FINISH kernel."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path

from .context import ContextEngine
from .executor import Executor
from .journal import EventKind, Journal
from .model import ContextOverflowError, Model, ModelError
from .policy import Approver, RejectingApprover, RunPolicy
from .recovery import ActionLedger, ActionRecord
from .session import SessionState
from .skills import SkillRegistry
from .tools import ToolRuntime
from .types import (
    ModelResponse,
    RunEvent,
    RunEventKind,
    RunEventSink,
    RunLimits,
    RunOutcome,
    RunProgress,
    RunStatus,
    ToolCall,
    ToolResult,
)
from .verifier import Verifier


class Harness:
    def __init__(
        self,
        model: Model,
        tools: ToolRuntime,
        context: ContextEngine | None = None,
        skills: SkillRegistry | None = None,
        policy: RunPolicy | None = None,
        approver: Approver | None = None,
        verifier: Verifier | None = None,
        limits: RunLimits | None = None,
        event_sink: RunEventSink | None = None,
    ):
        self.model = model
        self.tools = tools
        self.context = context or ContextEngine()
        self.skills = skills or SkillRegistry()
        self.policy = policy or RunPolicy()
        self.approver = approver or RejectingApprover()
        self.verifier = verifier
        self.limits = limits or RunLimits()
        self.event_sink = event_sink

    def run(
        self,
        task: str,
        executor: Executor,
        journal: Journal | None = None,
        run_id: str | None = None,
    ) -> RunOutcome:
        journal = journal or Journal()
        run_id = run_id or uuid.uuid4().hex[:12]
        if journal.events:
            raise ValueError("run() requires an empty journal; use resume() for an existing run")
        self.start_session(executor, journal, run_id)
        return self.send(task, executor, journal, run_id)

    def start_session(
        self,
        executor: Executor,
        journal: Journal,
        session_id: str | None = None,
    ) -> str:
        """Initialize one durable session and freeze its execution boundary."""

        if journal.events:
            state = SessionState.from_journal(journal, session_id)
            if session_id is not None and state.session_id != session_id:
                raise ValueError(
                    f"Journal belongs to session {state.session_id}, not {session_id}"
                )
            return state.session_id
        session_id = session_id or uuid.uuid4().hex[:12]
        journal.append(
            EventKind.SESSION_START,
            {
                "session_id": session_id,
                "workspace": str(executor.workspace),
                "policy": self.policy.to_dict(),
                "executor": self._executor_boundary(executor),
            },
        )
        return session_id

    def send(
        self,
        message: str,
        executor: Executor,
        journal: Journal,
        session_id: str | None = None,
    ) -> RunOutcome:
        """Append one user turn and drive it to a durable terminal result."""

        if not message.strip():
            raise ValueError("Message must not be empty")
        session_id = self.start_session(executor, journal, session_id)
        state = SessionState.from_journal(journal, session_id)
        self._validate_boundary(state, executor)
        if state.active_turn is not None:
            raise ValueError(
                f"Turn {state.active_turn.turn_id} is incomplete; call resume() before send()"
            )
        turn_id = state.next_turn_id
        turn_start = journal.append(
            EventKind.INPUT,
            {
                "content": message,
                "source": "user",
                "session_id": session_id,
                "turn_id": turn_id,
            },
        )
        return self._drive(
            message,
            executor,
            journal,
            session_id,
            turn_id,
            turn_start.seq,
        )

    def _validate_boundary(self, state: SessionState, executor: Executor) -> None:
        if state.workspace and Path(state.workspace).resolve() != executor.workspace:
            raise ValueError(
                f"Session workspace is {state.workspace}; received {executor.workspace}"
            )
        if state.policy is not None and state.policy != self.policy.to_dict():
            raise ValueError("Session policy cannot change between turns")
        if state.executor is not None and state.executor != self._executor_boundary(executor):
            raise ValueError("Session executor cannot change between turns")

    @staticmethod
    def _executor_boundary(executor: Executor) -> dict[str, str]:
        if executor.hard_sandbox:
            return {"kind": "docker", "image": str(getattr(executor, "image", ""))}
        return {"kind": "local"}

    def resume(self, executor: Executor, journal: Journal) -> RunOutcome:
        state = SessionState.from_journal(journal)
        self._validate_boundary(state, executor)
        turn = state.active_turn
        if turn is None:
            last_turn = state.last_turn
            if last_turn is None or last_turn.final_seq is None:
                raise ValueError("Session has no turn to resume")
            final = journal.events[last_turn.final_seq]
            return self._outcome_from_final(final.data, journal)

        task = turn.content
        run_id = state.session_id
        turn_start_seq = turn.input_seq

        def recover(record: ActionRecord) -> RunOutcome | None:
            self._emit(RunEventKind.TOOLS_START, calls=(record.call,))
            result = self.tools.recover_pending(record, executor, journal)
            self._emit(RunEventKind.TOOLS_END, results=(result,))
            if result.error not in {"UNKNOWN_COMMIT", "RECOVERY_CONFLICT"}:
                return None
            return self._finish(
                journal,
                run_id,
                RunStatus.NEEDS_CONFIRMATION,
                result.output,
                self._progress(journal, turn_start_seq),
                turn.turn_id,
            )

        ledger = ActionLedger.from_journal(journal, turn_start_seq)
        last_response = journal.last(EventKind.MODEL_RESPONSE, turn_start_seq)
        older_pending = tuple(
            record
            for record in ledger.pending()
            if last_response is None or record.intent.seq < last_response.seq
        )
        for pending in older_pending:
            outcome = recover(pending)
            if outcome is not None:
                return outcome

        if last_response:
            response = ModelResponse.from_dict(last_response.data["response"])
            # Preserve the provider's original call order during recovery.  A
            # later unstarted action must not cross an earlier uncertain write.
            for call in response.tool_calls:
                ledger = ActionLedger.from_journal(journal, turn_start_seq)
                record = ledger.find_intent(last_response.seq, call.id)
                if record is None:
                    self._execute_tools(
                        (call,),
                        executor,
                        journal,
                        response_seq=last_response.seq,
                    )
                    continue
                if record.commit is not None:
                    continue
                outcome = recover(record)
                if outcome is not None:
                    return outcome

        ledger = ActionLedger.from_journal(journal, turn_start_seq)
        for pending in ledger.pending():
            outcome = recover(pending)
            if outcome is not None:
                return outcome

        if last_response:
            response = ModelResponse.from_dict(last_response.data["response"])
            continued_after_response = bool(
                journal.find(EventKind.INPUT, last_response.seq)
            )
            if not response.tool_calls and not continued_after_response:
                outcome = self._settle_completion(
                    response,
                    executor,
                    journal,
                    run_id,
                    turn.turn_id,
                    self._progress(journal, turn_start_seq),
                )
                if outcome is not None:
                    return outcome
        self._checkpoint(journal, executor, ())
        return self._drive(
            task,
            executor,
            journal,
            run_id,
            turn.turn_id,
            turn_start_seq,
        )

    def _drive(
        self,
        task: str,
        executor: Executor,
        journal: Journal,
        run_id: str,
        turn_id: str,
        turn_start_seq: int,
    ) -> RunOutcome:
        progress = self._progress(journal, turn_start_seq)
        if not progress.started_at:
            progress.started_at = time.monotonic()

        while True:
            budget_error = self._budget_error(progress)
            if budget_error:
                return self._finish(
                    journal,
                    run_id,
                    RunStatus.BUDGET_EXHAUSTED,
                    budget_error,
                    progress,
                    turn_id,
                )

            view = self.context.build(
                journal,
                task,
                str(executor.workspace),
                self.policy.describe(),
                self.skills.catalog(),
            )
            try:
                response = self._complete_model(view.messages)
            except ContextOverflowError:
                compacted = self.context.build(
                    journal,
                    task,
                    str(executor.workspace),
                    self.policy.describe(),
                    self.skills.catalog(),
                    force_compact=True,
                )
                try:
                    response = self._complete_model(compacted.messages)
                except ModelError as exc:
                    return self._finish(
                        journal,
                        run_id,
                        RunStatus.FAILED,
                        f"Model failed: {exc}",
                        progress,
                        turn_id,
                    )
            except ModelError as exc:
                return self._finish(
                    journal,
                    run_id,
                    RunStatus.FAILED,
                    f"Model failed: {exc}",
                    progress,
                    turn_id,
                )

            response_event = journal.append(
                EventKind.MODEL_RESPONSE,
                {"response": response.to_dict()},
            )
            progress.steps += 1
            progress.tokens += response.prompt_tokens + response.completion_tokens
            if response.tool_calls:
                results = self._execute_tools(
                    response.tool_calls,
                    executor,
                    journal,
                    response_seq=response_event.seq,
                )
                current_view = self.context.build(
                    journal,
                    task,
                    str(executor.workspace),
                    self.policy.describe(),
                    self.skills.catalog(),
                )
                self._checkpoint(
                    journal,
                    executor,
                    tuple(result.call_id for result in results),
                    current_view.memory_snapshot,
                )
                fingerprint = _fingerprint(response.tool_calls, results)
                progress.recent_fingerprints.append(fingerprint)
                progress.recent_fingerprints = progress.recent_fingerprints[-3:]
                if len(progress.recent_fingerprints) == 3 and len(set(progress.recent_fingerprints)) == 1:
                    if progress.recovery_used:
                        return self._finish(
                            journal,
                            run_id,
                            RunStatus.FAILED,
                            "Stopped after the same action batch made no progress repeatedly.",
                            progress,
                            turn_id,
                        )
                    journal.append(
                        EventKind.INPUT,
                        {
                            "content": (
                                "The last action batch repeated without progress. Reassess the evidence "
                                "and choose a different approach; do not repeat the same calls."
                            ),
                            "source": "kernel_recovery",
                            "turn_id": turn_id,
                        },
                    )
                    progress.recovery_used = True
                    progress.recent_fingerprints.clear()
                continue

            outcome = self._settle_completion(
                response,
                executor,
                journal,
                run_id,
                turn_id,
                progress,
            )
            if outcome is not None:
                return outcome

    def _complete_model(self, messages: tuple[dict, ...]) -> ModelResponse:
        self._emit(RunEventKind.MODEL_START, model=self.model.name)
        try:
            return self.model.complete(
                messages,
                self.tools.schemas(),
                lambda text: self._emit(RunEventKind.TEXT_DELTA, text=text),
            )
        finally:
            self._emit(RunEventKind.MODEL_END)

    def _execute_tools(
        self,
        calls: tuple[ToolCall, ...] | list[ToolCall],
        executor: Executor,
        journal: Journal,
        response_seq: int | None = None,
    ) -> tuple[ToolResult, ...]:
        calls = tuple(calls)
        self._emit(RunEventKind.TOOLS_START, calls=calls)
        results = self.tools.execute_batch(
            calls,
            self.policy,
            executor,
            journal,
            self.approver,
            response_seq=response_seq,
        )
        self._emit(RunEventKind.TOOLS_END, results=results)
        return results

    def _emit(self, kind: RunEventKind, **data: object) -> None:
        if self.event_sink is not None:
            self.event_sink(RunEvent(kind, dict(data)))

    def _settle_completion(
        self,
        response: ModelResponse,
        executor: Executor,
        journal: Journal,
        run_id: str,
        turn_id: str,
        progress: RunProgress,
    ) -> RunOutcome | None:
        """Commit a completed model answer without requesting it a second time."""

        if not response.content.strip():
            return self._finish(
                journal,
                run_id,
                RunStatus.FAILED,
                "Model returned an empty response.",
                progress,
                turn_id,
            )
        if self.verifier is not None:
            self._emit(RunEventKind.VERIFY_START)
            verification = self.verifier.verify(executor)
            self._emit(
                RunEventKind.VERIFY_END,
                passed=verification.passed,
                output=verification.output,
            )
            if not verification.passed:
                journal.append(
                    EventKind.INPUT,
                    {
                        "content": f"Completion verification failed:\n{verification.output}",
                        "source": "verifier",
                        "turn_id": turn_id,
                    },
                )
                self._checkpoint(journal, executor, ())
                return None
        return self._finish(
            journal,
            run_id,
            RunStatus.SUCCEEDED,
            response.content,
            progress,
            turn_id,
        )

    def _budget_error(self, progress: RunProgress) -> str | None:
        if progress.steps >= self.limits.max_steps:
            return f"Step budget exhausted ({self.limits.max_steps})."
        if progress.tokens >= self.limits.max_tokens:
            return f"Token budget exhausted ({self.limits.max_tokens})."
        if time.monotonic() - progress.started_at >= self.limits.max_seconds:
            return f"Time budget exhausted ({self.limits.max_seconds}s)."
        return None

    @staticmethod
    def _checkpoint(
        journal: Journal,
        executor: Executor,
        completed_call_ids: tuple[str, ...],
        memory_snapshot: str | None = None,
    ) -> None:
        journal.append(
            EventKind.CHECKPOINT,
            {
                "workspace_revision": executor.snapshot().revision,
                "memory_snapshot": memory_snapshot,
                "completed_call_ids": list(completed_call_ids),
            },
        )

    @staticmethod
    def _progress(journal: Journal, after_seq: int = -1) -> RunProgress:
        responses = [
            ModelResponse.from_dict(event.data["response"])
            for event in journal.find(EventKind.MODEL_RESPONSE, after_seq)
        ]
        return RunProgress(
            steps=len(responses),
            tokens=sum(item.prompt_tokens + item.completion_tokens for item in responses),
            started_at=time.monotonic(),
            recovery_used=any(
                event.data.get("source") == "kernel_recovery"
                for event in journal.find(EventKind.INPUT, after_seq)
            ),
        )

    def _finish(
        self,
        journal: Journal,
        run_id: str,
        status: RunStatus,
        answer: str,
        progress: RunProgress,
        turn_id: str,
    ) -> RunOutcome:
        data = {
            "run_id": run_id,
            "status": status.value,
            "answer": answer,
            "steps": progress.steps,
            "tokens": progress.tokens,
            "turn_id": turn_id,
        }
        journal.append(EventKind.FINAL, data)
        return self._outcome_from_final(data, journal)

    @staticmethod
    def _outcome_from_final(data: dict, journal: Journal) -> RunOutcome:
        return RunOutcome(
            run_id=data["run_id"],
            status=RunStatus(data["status"]),
            answer=data["answer"],
            steps=int(data["steps"]),
            tokens=int(data["tokens"]),
            journal_path=str(journal.path) if journal.path else None,
            turn_id=data.get("turn_id"),
        )


def _fingerprint(calls: tuple[ToolCall, ...], results: tuple[ToolResult, ...]) -> str:
    value = {
        "calls": [{"name": call.name, "arguments": call.arguments} for call in calls],
        "results": [
            {
                "ok": result.ok,
                "error": result.error,
                "output": result.output[:500],
            }
            for result in results
        ],
    }
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()
