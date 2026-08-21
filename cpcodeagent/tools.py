"""Semantic tools and a deterministic read-parallel/write-barrier runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .executor import ExecutionEnv, ExecutionError, Executor
from .journal import Event, EventKind, Journal
from .policy import Approver, RejectingApprover, RunPolicy
from .types import Action, Capability, Decision, ToolCall, ToolResult


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


@dataclass(frozen=True)
class _Prepared:
    call: ToolCall
    tool: Tool | None
    action: Action | None
    immediate: ToolResult | None


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
    ) -> tuple[ToolResult, ...]:
        env = ExecutionEnv(executor)
        approver = approver or RejectingApprover()
        prepared = [self._prepare(call, policy, env, approver) for call in calls]

        results: list[ToolResult] = []
        index = 0
        while index < len(prepared):
            item = prepared[index]
            if item.immediate is not None:
                self._record_calls((item,), executor, journal)
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
                segment = prepared[index:end]
                self._record_calls(segment, executor, journal)
                with ThreadPoolExecutor(
                    max_workers=min(self.max_parallel_reads, len(segment))
                ) as pool:
                    segment_results = list(pool.map(lambda value: self._execute(value, env), segment))
                index = end
            else:
                self._record_calls((item,), executor, journal)
                segment_results = [self._execute(item, env)]
                index += 1

            for result in segment_results:
                journal.append(EventKind.TOOL_RESULT, {"result": result.to_dict()})
                results.append(result)
        return tuple(results)

    @staticmethod
    def _record_calls(
        items: Sequence[_Prepared], executor: Executor, journal: Journal
    ) -> None:
        baseline = executor.snapshot().revision
        for item in items:
            journal.append(
                EventKind.TOOL_CALL,
                {
                    "call": item.call.to_dict(),
                    "action": item.action.to_dict() if item.action else None,
                    "authorized": item.immediate is None,
                    "workspace_revision": baseline,
                },
            )

    def recover_pending(
        self,
        pending: Event,
        executor: Executor,
        journal: Journal,
    ) -> ToolResult:
        call = ToolCall.from_dict(pending.data["call"])
        action_data = pending.data.get("action")
        if not pending.data.get("authorized") or action_data is None:
            result = ToolResult(call.id, False, error="INTERRUPTED_BEFORE_AUTHORIZATION")
        else:
            action = Action.from_dict(action_data)
            current_revision = executor.snapshot().revision
            baseline = pending.data.get("workspace_revision")
            external = Capability.EXTERNAL_WRITE in action.capabilities
            if action.idempotent or (not external and current_revision == baseline):
                tool = self._tools.get(call.name)
                if tool is None:
                    result = ToolResult(call.id, False, error="UNKNOWN_TOOL")
                else:
                    result = self._execute(_Prepared(call, tool, action, None), ExecutionEnv(executor))
            else:
                result = ToolResult(
                    call.id,
                    False,
                    output="Execution may already have produced a side effect; automatic replay is unsafe.",
                    error="UNKNOWN_COMMIT",
                )
        journal.append(EventKind.TOOL_RESULT, {"result": result.to_dict(), "recovered": True})
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

    def _execute(self, item: _Prepared, env: ExecutionEnv) -> ToolResult:
        assert item.tool is not None and item.action is not None
        before = env.executor.snapshot() if not item.action.read_only else None
        attempts = 2 if item.action.idempotent else 1
        for attempt in range(attempts):
            try:
                execution = item.tool.execute(item.call.arguments, env)
                artifacts = list(execution.artifacts)
                if before is not None:
                    after = env.executor.snapshot()
                    artifacts.extend(f"changed:{path}" for path in after.changed_since(before))
                return ToolResult(
                    call_id=item.call.id,
                    ok=execution.ok,
                    output=execution.output,
                    error=execution.error,
                    artifacts=tuple(dict.fromkeys(artifacts)),
                )
            except ExecutionError as exc:
                if not (exc.retryable and attempt + 1 < attempts):
                    return ToolResult(item.call.id, False, output=str(exc), error=exc.code)
            # Tools are extension points. A tool bug must become evidence for the model
            # instead of tearing down the durable session.
            except Exception as exc:  # noqa: BLE001
                return ToolResult(item.call.id, False, output=str(exc), error="TOOL_CRASH")
        raise AssertionError("unreachable")


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
