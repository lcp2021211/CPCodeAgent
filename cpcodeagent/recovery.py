"""Crash-consistency contracts and the durable action-ledger projection.

The journal is the write-ahead log.  This module contains only pure value
objects and a projection over that log; it never performs a tool side effect.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .executor import Executor
from .journal import Event, EventKind, Journal
from .types import Action, ToolCall, ToolResult


class RecoveryMode(str, Enum):
    """How a started action may be recovered when its result is missing."""

    RETRY_SAFE = "retry_safe"
    VERIFY_FILES = "verify_files"
    MANUAL = "manual"


class EffectState(str, Enum):
    """Current relationship between the world and a recorded effect contract."""

    APPLIED = "applied"
    NOT_APPLIED = "not_applied"
    CONFLICT = "conflict"


class ActionState(str, Enum):
    INTENT = "intent"
    STARTED = "started"
    COMMITTED = "committed"


@dataclass(frozen=True)
class FileCondition:
    """Expected content identity for one workspace-relative file.

    ``digest=None`` means that the path must not be a regular file.
    """

    path: str
    digest: str | None

    def matches(self, executor: Executor) -> bool:
        return executor.file_digest(self.path) == self.digest

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "digest": self.digest}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileCondition:
        digest = data.get("digest")
        return cls(str(data["path"]), str(digest) if digest is not None else None)


@dataclass(frozen=True)
class EffectContract:
    """A durable recovery contract captured before an action starts."""

    mode: RecoveryMode
    before: tuple[FileCondition, ...] = ()
    after: tuple[FileCondition, ...] = ()

    def __post_init__(self) -> None:
        if self.mode is RecoveryMode.VERIFY_FILES:
            if not self.before or len(self.before) != len(self.after):
                raise ValueError("verify_files contracts require paired before/after conditions")
            if tuple(item.path for item in self.before) != tuple(item.path for item in self.after):
                raise ValueError("before/after conditions must describe the same paths")
        elif self.before or self.after:
            raise ValueError(f"{self.mode.value} contracts cannot contain file conditions")

    @classmethod
    def retry_safe(cls) -> EffectContract:
        return cls(RecoveryMode.RETRY_SAFE)

    @classmethod
    def manual(cls) -> EffectContract:
        return cls(RecoveryMode.MANUAL)

    @classmethod
    def file_transition(
        cls,
        before: FileCondition,
        after: FileCondition,
    ) -> EffectContract:
        return cls(RecoveryMode.VERIFY_FILES, (before,), (after,))

    @classmethod
    def file_transitions(
        cls,
        before: tuple[FileCondition, ...],
        after: tuple[FileCondition, ...],
    ) -> EffectContract:
        """Describe one retryable action spanning several deterministic files."""

        return cls(RecoveryMode.VERIFY_FILES, before, after)

    def inspect(self, executor: Executor) -> EffectState:
        if self.mode is not RecoveryMode.VERIFY_FILES:
            raise ValueError(f"Cannot inspect a {self.mode.value} recovery contract")
        # Check the postcondition first: writing the desired bytes over identical
        # bytes is both a valid precondition and an already-satisfied effect.
        if self.after and all(condition.matches(executor) for condition in self.after):
            return EffectState.APPLIED
        # A multi-file action may stop after applying only some files. It remains
        # safely resumable while every path still matches either its recorded
        # precondition or postcondition. Any third state is a real conflict.
        if self.before and all(
            before.matches(executor) or after.matches(executor)
            for before, after in zip(self.before, self.after, strict=True)
        ):
            return EffectState.NOT_APPLIED
        return EffectState.CONFLICT

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "before": [item.to_dict() for item in self.before],
            "after": [item.to_dict() for item in self.after],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EffectContract:
        return cls(
            RecoveryMode(data["mode"]),
            tuple(FileCondition.from_dict(item) for item in data.get("before", ())),
            tuple(FileCondition.from_dict(item) for item in data.get("after", ())),
        )


@dataclass(frozen=True)
class ActionRecord:
    """Current state of one action, projected from immutable journal events."""

    action_id: str
    call: ToolCall
    action: Action | None
    authorized: bool
    contract: EffectContract | None
    prepared_result: ToolResult | None
    intent: Event
    response_seq: int | None = None
    starts: tuple[Event, ...] = ()
    commit: Event | None = None
    legacy: bool = False

    @property
    def state(self) -> ActionState:
        if self.commit is not None:
            return ActionState.COMMITTED
        if self.starts:
            return ActionState.STARTED
        return ActionState.INTENT

    @property
    def result(self) -> ToolResult | None:
        if self.commit is None:
            return None
        return ToolResult.from_dict(self.commit.data["result"])


class ActionLedger:
    """Deterministically projects action state from a session Journal."""

    def __init__(self, records: tuple[ActionRecord, ...]):
        self.records = records
        self._by_id = {record.action_id: record for record in records}

    @classmethod
    def from_journal(cls, journal: Journal, after_seq: int = -1) -> ActionLedger:
        builders: dict[str, dict[str, Any]] = {}
        order: list[str] = []

        for event in journal.events:
            if event.seq <= after_seq:
                continue
            if event.kind is EventKind.TOOL_CALL:
                call = ToolCall.from_dict(event.data["call"])
                action_id = str(event.data.get("action_id") or f"legacy-{event.seq}")
                if action_id in builders:
                    raise ValueError(f"Action ID {action_id} appears more than once")
                action_data = event.data.get("action")
                contract_data = event.data.get("effect_contract")
                prepared_data = event.data.get("prepared_result")
                builders[action_id] = {
                    "action_id": action_id,
                    "call": call,
                    "action": Action.from_dict(action_data) if action_data else None,
                    "authorized": bool(event.data.get("authorized")),
                    "contract": (
                        EffectContract.from_dict(contract_data) if contract_data else None
                    ),
                    "prepared_result": (
                        ToolResult.from_dict(prepared_data) if prepared_data else None
                    ),
                    "intent": event,
                    "response_seq": event.data.get("response_seq"),
                    "starts": [],
                    "commit": None,
                    "legacy": "action_id" not in event.data,
                }
                order.append(action_id)
                continue

            if event.kind is EventKind.TOOL_STARTED:
                action_id = _event_action_id(event, builders)
                if action_id is None:
                    raise ValueError(f"Tool start at seq {event.seq} has no matching intent")
                if builders[action_id]["commit"] is not None:
                    raise ValueError(f"Action {action_id} started after it was committed")
                builders[action_id]["starts"].append(event)
                continue

            if event.kind is EventKind.TOOL_RESULT:
                action_id = _event_action_id(event, builders)
                if action_id is None:
                    raise ValueError(f"Tool result at seq {event.seq} has no matching intent")
                if builders[action_id]["commit"] is not None:
                    raise ValueError(f"Action {action_id} has more than one committed result")
                builders[action_id]["commit"] = event

        records = tuple(
            ActionRecord(
                **{
                    **builders[action_id],
                    "starts": tuple(builders[action_id]["starts"]),
                }
            )
            for action_id in order
        )
        return cls(records)

    def get(self, action_id: str) -> ActionRecord:
        return self._by_id[action_id]

    def pending(self) -> tuple[ActionRecord, ...]:
        return tuple(record for record in self.records if record.commit is None)

    def find_intent(self, response_seq: int, call_id: str) -> ActionRecord | None:
        for record in reversed(self.records):
            if record.call.id == call_id and record.response_seq == response_seq:
                return record
        for record in reversed(self.records):
            if record.call.id == call_id and record.response_seq is None:
                return record
        return None


def content_digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _event_action_id(
    event: Event,
    builders: dict[str, dict[str, Any]],
) -> str | None:
    explicit = event.data.get("action_id")
    if explicit is not None:
        value = str(explicit)
        return value if value in builders else None

    # Compatibility with journals written before action IDs existed.  A call ID
    # may repeat across turns, so bind to the newest still-uncommitted intent.
    result_data = event.data.get("result")
    call_id = event.data.get("call_id")
    if isinstance(result_data, dict):
        call_id = result_data.get("call_id")
    for action_id in reversed(tuple(builders)):
        builder = builders[action_id]
        if builder["call"].id == call_id and builder["commit"] is None:
            return action_id
    return None
