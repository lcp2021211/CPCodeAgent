"""Durable session identity and turn-state projection."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .journal import EventKind, Journal
from .types import RunStatus

_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class TurnState:
    """One user request and its optional terminal result."""

    turn_id: str
    input_seq: int
    content: str
    final_seq: int | None = None
    status: RunStatus | None = None

    @property
    def completed(self) -> bool:
        return self.final_seq is not None


@dataclass(frozen=True)
class SessionState:
    """A read-only projection derived from a session journal."""

    session_id: str
    workspace: str | None
    policy: dict[str, Any] | None
    executor: dict[str, Any] | None
    turns: tuple[TurnState, ...]

    @property
    def active_turn(self) -> TurnState | None:
        if self.turns and not self.turns[-1].completed:
            return self.turns[-1]
        return None

    @property
    def last_turn(self) -> TurnState | None:
        return self.turns[-1] if self.turns else None

    @property
    def next_turn_id(self) -> str:
        return f"turn-{len(self.turns) + 1:04d}"

    @classmethod
    def from_journal(
        cls,
        journal: Journal,
        fallback_session_id: str | None = None,
    ) -> SessionState:
        start = journal.last(EventKind.SESSION_START)
        session_id = fallback_session_id
        workspace: str | None = None
        policy: dict[str, Any] | None = None
        executor: dict[str, Any] | None = None
        if start is not None:
            session_id = str(start.data["session_id"])
            workspace = start.data.get("workspace")
            raw_policy = start.data.get("policy")
            if isinstance(raw_policy, dict):
                policy = dict(raw_policy)
            raw_executor = start.data.get("executor")
            if isinstance(raw_executor, dict):
                executor = dict(raw_executor)

        mutable_turns: list[dict[str, Any]] = []
        for event in journal.events:
            if event.kind is EventKind.INPUT and _is_user_input(event.data, mutable_turns):
                turn_id = str(
                    event.data.get("turn_id")
                    or event.data.get("run_id")
                    or f"turn-{len(mutable_turns) + 1:04d}"
                )
                session_id = session_id or event.data.get("run_id")
                workspace = workspace or event.data.get("workspace")
                mutable_turns.append(
                    {
                        "turn_id": turn_id,
                        "input_seq": event.seq,
                        "content": str(event.data.get("content", "")),
                        "final_seq": None,
                        "status": None,
                    }
                )
            elif event.kind is EventKind.FINAL:
                turn = _matching_open_turn(mutable_turns, event.data.get("turn_id"))
                if turn is not None:
                    turn["final_seq"] = event.seq
                    try:
                        turn["status"] = RunStatus(event.data["status"])
                    except (KeyError, ValueError):
                        turn["status"] = None

        if not session_id:
            raise ValueError("Journal has no session identity")
        turns = tuple(TurnState(**turn) for turn in mutable_turns)
        return cls(str(session_id), workspace, policy, executor, turns)


@dataclass(frozen=True)
class Session:
    session_id: str
    journal: Journal

    @property
    def state(self) -> SessionState:
        return SessionState.from_journal(self.journal, self.session_id)


class SessionStore:
    """Maps validated session IDs to durable JSONL journals."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    def create(self, session_id: str | None = None) -> Session:
        session_id = session_id or uuid.uuid4().hex[:12]
        path = self._path(session_id)
        if path.exists():
            raise FileExistsError(f"Session already exists: {session_id}")
        return Session(session_id, Journal(path))

    def open(self, session_id: str) -> Session:
        path = self._path(session_id)
        if not path.is_file():
            raise FileNotFoundError(f"Unknown session: {session_id}")
        return Session(session_id, Journal(path))

    def _path(self, session_id: str) -> Path:
        if not _SESSION_ID.fullmatch(session_id):
            raise ValueError("Session ID may contain only letters, numbers, '-' and '_'")
        return self.root / f"{session_id}.jsonl"


def _is_user_input(data: dict[str, Any], turns: list[dict[str, Any]]) -> bool:
    source = data.get("source")
    return source == "user" or (source is None and not turns)


def _matching_open_turn(
    turns: list[dict[str, Any]], turn_id: object
) -> dict[str, Any] | None:
    for turn in reversed(turns):
        if turn["final_seq"] is not None:
            continue
        if turn_id is None or turn["turn_id"] == turn_id:
            return turn
    return None
