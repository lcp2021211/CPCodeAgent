"""Semantic tools and a deterministic read-parallel/write-barrier runtime."""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

from .executor import ExecutionEnv, ExecutionError, Executor
from .journal import EventKind, Journal
from .policy import Approver, RejectingApprover, RunPolicy
from .recovery import (
    ActionRecord,
    EffectContract,
    EffectState,
    RecoveryMode,
)
from .types import Action, Decision, ToolCall, ToolResult


@dataclass(frozen=True)
class ToolExecution:
    ok: bool
    output: str = ""
    error: str | None = None
    artifacts: tuple[str, ...] = ()


class Tool(ABC):
    name: str
    description: str
    input_schema: dict[str, Any]

    def describe(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    @abstractmethod
    def classify(self, arguments: dict[str, Any], env: ExecutionEnv) -> Action:
        """Describe effects before execution."""

    @abstractmethod
    def execute(self, arguments: dict[str, Any], env: ExecutionEnv) -> ToolExecution:
        """Execute inside the provided environment."""

    def effect_contract(
        self,
        arguments: dict[str, Any],
        action: Action,
        env: ExecutionEnv,
        action_id: str,
    ) -> EffectContract:
        """Describe how an interrupted action can be reconciled.

        Extension tools get conservative defaults: reads/idempotent operations
        are retryable, while an opaque side effect requires human resolution
        once it has started.
        """

        if action.read_only or action.idempotent:
            return EffectContract.retry_safe()
        return EffectContract.manual()


@dataclass(frozen=True)
class _Prepared:
    call: ToolCall
    tool: Tool | None
    action: Action | None
    immediate: ToolResult | None
    action_id: str | None = None
    contract: EffectContract | None = None


class ToolRuntime:
    def __init__(self, tools: Iterable[Tool] = (), max_parallel_reads: int = 8):
        self._tools: dict[str, Tool] = {}
        self.max_parallel_reads = max(1, max_parallel_reads)
        for tool in tools:
            self.register(tool)

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._tools)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def schemas(self) -> tuple[dict[str, Any], ...]:
        return tuple(tool.describe() for tool in self._tools.values())

    def execute_batch(
        self,
        calls: Sequence[ToolCall],
        policy: RunPolicy,
        executor: Executor,
        journal: Journal,
        approver: Approver | None = None,
        response_seq: int | None = None,
    ) -> tuple[ToolResult, ...]:
        env = ExecutionEnv(executor)
        approver = approver or RejectingApprover()
        prepared: list[_Prepared] = []
        seen: set[str] = set()
        for call in calls:
            key = semantic_tool_call_key(call)
            if key in seen:
                prepared.append(
                    _Prepared(
                        call,
                        self._tools.get(call.name),
                        None,
                        ToolResult(
                            call.id,
                            False,
                            output=(
                                f"Duplicate {call.name} call suppressed: identical arguments "
                                "already appeared in this model response."
                            ),
                            error="DUPLICATE_CALL",
                        ),
                    )
                )
                continue
            seen.add(key)
            prepared.append(self._prepare(call, policy, env, approver))

        results: list[ToolResult] = []
        index = 0
        while index < len(prepared):
            item = prepared[index]
            if item.immediate is not None:
                item = self._materialize(item, env)
                self._record_intents((item,), executor, journal, response_seq)
                assert item.immediate is not None
                self._commit(journal, item, item.immediate)
                segment_results = [item.immediate]
                index += 1
            elif item.action and item.action.read_only:
                end = index
                while (
                    end < len(prepared)
                    and prepared[end].immediate is None
                    and prepared[end].action is not None
                    and prepared[end].action.read_only
                ):
                    end += 1
                segment = tuple(self._materialize(value, env) for value in prepared[index:end])
                self._record_intents(segment, executor, journal, response_seq)
                with ThreadPoolExecutor(
                    max_workers=min(self.max_parallel_reads, len(segment))
                ) as pool:
                    segment_results = list(
                        pool.map(
                            lambda value: self._settle(value, executor, journal),
                            segment,
                        )
                    )
                index = end
            else:
                item = self._materialize(item, env)
                self._record_intents((item,), executor, journal, response_seq)
                if item.immediate is not None:
                    self._commit(journal, item, item.immediate)
                    segment_results = [item.immediate]
                else:
                    segment_results = [self._execute_and_commit(item, executor, journal)]
                index += 1

            results.extend(segment_results)
        return tuple(results)

    def _materialize(self, item: _Prepared, env: ExecutionEnv) -> _Prepared:
        action_id = uuid.uuid4().hex
        if item.immediate is not None or item.tool is None or item.action is None:
            return replace(item, action_id=action_id)
        try:
            contract = item.tool.effect_contract(
                item.call.arguments,
                item.action,
                env,
                action_id,
            )
        except ExecutionError as exc:
            return replace(
                item,
                action_id=action_id,
                immediate=ToolResult(
                    item.call.id,
                    False,
                    output=str(exc),
                    error=exc.code,
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            return replace(
                item,
                action_id=action_id,
                immediate=ToolResult(
                    item.call.id,
                    False,
                    output=str(exc),
                    error="EFFECT_PREPARATION_FAILED",
                ),
            )
        return replace(item, action_id=action_id, contract=contract)

    @staticmethod
    def _record_intents(
        items: Sequence[_Prepared],
        executor: Executor,
        journal: Journal,
        response_seq: int | None,
    ) -> None:
        baseline = executor.snapshot().revision
        for item in items:
            assert item.action_id is not None
            journal.append(
                EventKind.TOOL_CALL,
                {
                    "action_id": item.action_id,
                    "call": item.call.to_dict(),
                    "action": item.action.to_dict() if item.action else None,
                    "authorized": item.immediate is None,
                    "workspace_revision": baseline,
                    "response_seq": response_seq,
                    "effect_contract": (
                        item.contract.to_dict() if item.contract is not None else None
                    ),
                    "prepared_result": (
                        item.immediate.to_dict() if item.immediate is not None else None
                    ),
                },
            )

    def recover_pending(
        self,
        pending: ActionRecord,
        executor: Executor,
        journal: Journal,
    ) -> ToolResult:
        tool = self._tools.get(pending.call.name)
        contract = pending.contract
        if contract is None and pending.action is not None:
            contract = (
                EffectContract.retry_safe()
                if pending.action.idempotent or pending.action.read_only
                else EffectContract.manual()
            )
        item = _Prepared(
            pending.call,
            tool,
            pending.action,
            pending.prepared_result,
            pending.action_id,
            contract,
        )

        if pending.prepared_result is not None:
            self._commit(journal, item, pending.prepared_result, recovered="prepared")
            return pending.prepared_result
        if not pending.authorized or pending.action is None:
            result = ToolResult(
                pending.call.id,
                False,
                error="INTERRUPTED_BEFORE_AUTHORIZATION",
            )
            self._commit(journal, item, result, recovered="not_authorized")
            return result
        if tool is None:
            result = ToolResult(pending.call.id, False, error="UNKNOWN_TOOL")
            self._commit(journal, item, result, recovered="tool_missing")
            return result

        # Old journals did not have a durable started marker.  A non-idempotent
        # call in that format is inherently ambiguous; pretending it was merely
        # an unstarted intent would reintroduce blind replay.
        if pending.legacy:
            if pending.action.idempotent or pending.action.read_only:
                return self._execute_and_commit(item, executor, journal, recovered=True)
            result = self._unknown_commit(pending)
            self._commit(journal, item, result, recovered="legacy_unknown")
            return result

        # A durable intent without TOOL_STARTED proves the executor never gained
        # control, so even an opaque command may be started exactly once now.
        if not pending.starts:
            return self._execute_and_commit(item, executor, journal, recovered=True)

        contract = item.contract or EffectContract.manual()
        if contract.mode is RecoveryMode.RETRY_SAFE:
            return self._execute_and_commit(item, executor, journal, recovered=True)
        if contract.mode is RecoveryMode.MANUAL:
            result = self._unknown_commit(pending)
            self._commit(journal, item, result, recovered="manual_required")
            return result

        executor.discard_staged_writes(
            pending.action_id,
            tuple(condition.path for condition in contract.after),
        )
        state = contract.inspect(executor)
        if state is EffectState.APPLIED:
            result = ToolResult(
                pending.call.id,
                True,
                output=(
                    f"Recovered {pending.call.name}: its recorded file postcondition "
                    "is already present."
                ),
                artifacts=(f"recovered:{pending.action_id}",),
            )
            self._commit(journal, item, result, recovered="effect_verified")
            return result
        if state is EffectState.NOT_APPLIED:
            return self._execute_and_commit(item, executor, journal, recovered=True)

        result = ToolResult(
            pending.call.id,
            False,
            output=(
                "The interrupted action's target matches neither its recorded precondition "
                "nor postcondition. The harness will not overwrite the divergent state."
            ),
            error="RECOVERY_CONFLICT",
        )
        self._commit(journal, item, result, recovered="conflict")
        return result

    def _prepare(
        self,
        call: ToolCall,
        policy: RunPolicy,
        env: ExecutionEnv,
        approver: Approver,
    ) -> _Prepared:
        tool = self._tools.get(call.name)
        if tool is None:
            return _Prepared(call, None, None, ToolResult(call.id, False, error="UNKNOWN_TOOL"))
        validation_error = _validate(tool.input_schema, call.arguments)
        if validation_error:
            return _Prepared(
                call,
                tool,
                None,
                ToolResult(call.id, False, output=validation_error, error="INVALID_ARGUMENTS"),
            )
        try:
            action = tool.classify(call.arguments, env)
        except (ExecutionError, KeyError, TypeError, ValueError) as exc:
            return _Prepared(
                call,
                tool,
                None,
                ToolResult(call.id, False, output=str(exc), error="CLASSIFICATION_FAILED"),
            )
        decision, reason = policy.decide(action, env.workspace)
        if decision is Decision.DENY:
            return _Prepared(
                call,
                tool,
                action,
                ToolResult(call.id, False, output=reason, error="POLICY_DENIED"),
            )
        if decision is Decision.ASK and not approver.approve(call, action, reason):
            return _Prepared(
                call,
                tool,
                action,
                ToolResult(call.id, False, output=reason, error="APPROVAL_REJECTED"),
            )
        return _Prepared(call, tool, action, None)

    def _settle(
        self,
        item: _Prepared,
        executor: Executor,
        journal: Journal,
    ) -> ToolResult:
        if item.immediate is not None:
            self._commit(journal, item, item.immediate)
            return item.immediate
        return self._execute_and_commit(item, executor, journal)

    def _execute_and_commit(
        self,
        item: _Prepared,
        executor: Executor,
        journal: Journal,
        recovered: bool = False,
    ) -> ToolResult:
        assert item.tool is not None and item.action is not None
        assert item.action_id is not None
        assert item.contract is not None

        contract = item.contract
        if contract.mode is RecoveryMode.VERIFY_FILES:
            state = contract.inspect(executor)
            if state is EffectState.APPLIED:
                result = ToolResult(
                    item.call.id,
                    True,
                    output=f"{item.tool.name}: requested file state is already present.",
                )
                self._commit(journal, item, result, recovered="already_satisfied")
                return result
            if state is EffectState.CONFLICT:
                result = ToolResult(
                    item.call.id,
                    False,
                    output="The target changed after the action was prepared; no write was attempted.",
                    error="PRECONDITION_CHANGED",
                )
                self._commit(journal, item, result)
                return result

        before = executor.snapshot() if not item.action.read_only else None
        journal.append(
            EventKind.TOOL_STARTED,
            {
                "action_id": item.action_id,
                "call_id": item.call.id,
                "workspace_revision": before.revision if before is not None else None,
                "recovery_attempt": recovered,
            },
        )
        result = self._invoke(item, ExecutionEnv(executor, item.action_id))

        if not result.ok and contract.mode is RecoveryMode.VERIFY_FILES:
            state = contract.inspect(executor)
            if state is EffectState.APPLIED:
                result = ToolResult(
                    item.call.id,
                    True,
                    output=(
                        f"{item.tool.name} reported an error, but its durable "
                        "postcondition was verified."
                    ),
                    artifacts=result.artifacts,
                )
            elif state is EffectState.CONFLICT:
                result = ToolResult(
                    item.call.id,
                    False,
                    output=(
                        "Execution failed and the target now matches neither the "
                        "recorded precondition nor postcondition."
                    ),
                    error="RECOVERY_CONFLICT",
                    artifacts=result.artifacts,
                )

        observation: dict[str, Any] | None = None
        if before is not None:
            after = executor.snapshot()
            changed = after.changed_since(before)
            artifacts = tuple(
                dict.fromkeys((*result.artifacts, *(f"changed:{path}" for path in changed)))
            )
            result = ToolResult(
                result.call_id,
                result.ok,
                result.output,
                result.error,
                artifacts,
            )
            observation = {
                "before_revision": before.revision,
                "after_revision": after.revision,
                "changed_paths": list(changed),
            }
        self._commit(
            journal,
            item,
            result,
            observation=observation,
            recovered="retried" if recovered else None,
        )
        return result

    def _invoke(self, item: _Prepared, env: ExecutionEnv) -> ToolResult:
        assert item.tool is not None and item.action is not None
        attempts = 2 if item.action.idempotent else 1
        for attempt in range(attempts):
            try:
                execution = item.tool.execute(item.call.arguments, env)
                return ToolResult(
                    call_id=item.call.id,
                    ok=execution.ok,
                    output=execution.output,
                    error=execution.error,
                    artifacts=execution.artifacts,
                )
            except ExecutionError as exc:
                if not (exc.retryable and attempt + 1 < attempts):
                    return ToolResult(item.call.id, False, output=str(exc), error=exc.code)
            # Tools are extension points. A tool bug must become evidence for the model
            # instead of tearing down the durable session.
            except Exception as exc:  # noqa: BLE001
                return ToolResult(item.call.id, False, output=str(exc), error="TOOL_CRASH")
        raise AssertionError("unreachable")

    @staticmethod
    def _commit(
        journal: Journal,
        item: _Prepared,
        result: ToolResult,
        observation: dict[str, Any] | None = None,
        recovered: str | None = None,
    ) -> None:
        assert item.action_id is not None
        journal.append(
            EventKind.TOOL_RESULT,
            {
                "action_id": item.action_id,
                "result": result.to_dict(),
                "observation": observation,
                "recovered": recovered,
            },
        )

    @staticmethod
    def _unknown_commit(pending: ActionRecord) -> ToolResult:
        return ToolResult(
            pending.call.id,
            False,
            output=(
                "The action started before interruption, but its effect cannot be "
                "verified safely. Automatic replay is unsafe, so it was not replayed."
            ),
            error="UNKNOWN_COMMIT",
        )


def semantic_tool_call_key(call: ToolCall) -> str:
    """Canonical action identity independent of provider-generated call IDs."""

    return json.dumps(
        {"name": call.name, "arguments": call.arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate(schema: dict[str, Any], arguments: Any, path: str = "arguments") -> str | None:
    expected = schema.get("type")
    python_types = {
        "object": dict,
        "array": list,
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
    }
    if expected in python_types and not isinstance(arguments, python_types[expected]):
        return f"{path} must be {expected}"
    if expected == "object":
        required = schema.get("required", ())
        missing = [key for key in required if key not in arguments]
        if missing:
            return f"Missing required arguments: {', '.join(missing)}"
        properties = schema.get("properties", {})
        for key, value in arguments.items():
            if key in properties:
                error = _validate(properties[key], value, f"{path}.{key}")
                if error:
                    return error
    if expected == "array" and "items" in schema:
        for index, value in enumerate(arguments):
            error = _validate(schema["items"], value, f"{path}[{index}]")
            if error:
                return error
    return None
