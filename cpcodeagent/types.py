"""Small, provider-neutral value objects shared by the harness layers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Capability(str, Enum):
    READ = "read"
    RUNTIME_WRITE = "runtime_write"
    WORKSPACE_WRITE = "workspace_write"
    NETWORK = "network"
    EXTERNAL_WRITE = "external_write"


class Decision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class RunStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NEEDS_CONFIRMATION = "needs_confirmation"


class RunEventKind(str, Enum):
    MODEL_START = "model_start"
    MODEL_END = "model_end"
    TEXT_DELTA = "text_delta"
    TOOLS_START = "tools_start"
    TOOLS_END = "tools_end"
    VERIFY_START = "verify_start"
    VERIFY_END = "verify_end"


@dataclass(frozen=True)
class RunEvent:
    """One ephemeral progress event; durable state still lives in the Journal."""

    kind: RunEventKind
    data: dict[str, Any] = field(default_factory=dict)


RunEventSink = Callable[[RunEvent], None]


@dataclass(frozen=True)
class Action:
    """The semantic meaning of one concrete tool call."""

    capabilities: frozenset[Capability]
    targets: tuple[str, ...] = ()
    idempotent: bool = True

    @property
    def read_only(self) -> bool:
        return self.capabilities <= {Capability.READ}

    def to_dict(self) -> dict[str, Any]:
        return {
            "capabilities": sorted(cap.value for cap in self.capabilities),
            "targets": list(self.targets),
            "idempotent": self.idempotent,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Action:
        return cls(
            capabilities=frozenset(Capability(value) for value in data["capabilities"]),
            targets=tuple(data.get("targets", ())),
            idempotent=bool(data.get("idempotent", False)),
        )


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        return cls(id=data["id"], name=data["name"], arguments=dict(data["arguments"]))


@dataclass(frozen=True)
class ModelResponse:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelResponse:
        return cls(
            content=data.get("content", ""),
            tool_calls=tuple(ToolCall.from_dict(item) for item in data.get("tool_calls", ())),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            model=data.get("model", ""),
        )


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    ok: bool
    output: str = ""
    error: str | None = None
    artifacts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "ok": self.ok,
            "output": self.output,
            "error": self.error,
            "artifacts": list(self.artifacts),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolResult:
        return cls(
            call_id=data["call_id"],
            ok=bool(data["ok"]),
            output=data.get("output", ""),
            error=data.get("error"),
            artifacts=tuple(data.get("artifacts", ())),
        )


@dataclass(frozen=True)
class ContextView:
    messages: tuple[dict[str, Any], ...]
    memory_snapshot: str | None = None


@dataclass(frozen=True)
class RunLimits:
    max_steps: int = 40
    max_seconds: float = 1_800
    max_tokens: int = 200_000


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    status: RunStatus
    answer: str
    steps: int
    tokens: int
    journal_path: str | None = None
    turn_id: str | None = None

    @property
    def session_id(self) -> str:
        return self.run_id


@dataclass(frozen=True)
class Verification:
    passed: bool
    output: str


@dataclass
class RunProgress:
    steps: int = 0
    tokens: int = 0
    started_at: float = 0.0
    stall_fingerprint: str | None = None
    stall_streak: int = 0
    stall_warned: bool = False
