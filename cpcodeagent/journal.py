"""Append-only run journal used for context, recovery, and replay."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class EventKind(str, Enum):
    SESSION_START = "session_start"
    INPUT = "input"
    MODEL_RESPONSE = "model_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    CHECKPOINT = "checkpoint"
    FINAL = "final"


@dataclass(frozen=True)
class Event:
    seq: int
    time: float
    kind: EventKind
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "time": self.time,
            "kind": self.kind.value,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Event:
        return cls(
            seq=int(value["seq"]),
            time=float(value["time"]),
            kind=EventKind(value["kind"]),
            data=dict(value["data"]),
        )


class Journal:
    """A thread-safe in-memory journal with optional durable JSONL backing."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path).expanduser().resolve() if path else None
        self._lock = threading.RLock()
        self._events: list[Event] = []
        if self.path and self.path.exists():
            self._load()

    @property
    def events(self) -> tuple[Event, ...]:
        with self._lock:
            return tuple(self._events)

    def append(self, kind: EventKind, data: dict[str, Any]) -> Event:
        with self._lock:
            event = Event(seq=len(self._events), time=time.time(), kind=kind, data=data)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                encoded = json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(encoded + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            self._events.append(event)
            return event

    def find(self, kind: EventKind, after_seq: int = -1) -> tuple[Event, ...]:
        return tuple(
            event for event in self.events if event.kind is kind and event.seq > after_seq
        )

    def last(self, kind: EventKind, after_seq: int = -1) -> Event | None:
        for event in reversed(self.events):
            if event.seq <= after_seq:
                break
            if event.kind is kind:
                return event
        return None

    def pending_tool_calls(self, after_seq: int = -1) -> tuple[Event, ...]:
        completed = {
            event.data["result"]["call_id"]
            for event in self.events
            if event.kind is EventKind.TOOL_RESULT and event.seq > after_seq
        }
        return tuple(
            event
            for event in self.events
            if event.kind is EventKind.TOOL_CALL
            and event.seq > after_seq
            and event.data["call"]["id"] not in completed
        )

    def __iter__(self) -> Iterator[Event]:
        return iter(self.events)

    def _load(self) -> None:
        assert self.path is not None
        loaded: list[Event] = []
        raw_lines = self.path.read_bytes().splitlines(keepends=True)
        valid_bytes = 0
        truncated_tail = False
        for index, raw_line in enumerate(raw_lines):
            line_number = index + 1
            if not raw_line.strip():
                continue
            try:
                event = Event.from_dict(json.loads(raw_line.decode("utf-8")))
            except (UnicodeDecodeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                incomplete_tail = index == len(raw_lines) - 1 and not raw_line.endswith(b"\n")
                if incomplete_tail:
                    truncated_tail = True
                    break
                raise ValueError(
                    f"Corrupt journal {self.path} at line {line_number}: {exc}"
                ) from exc
            if event.seq != len(loaded):
                raise ValueError(
                    f"Non-contiguous journal sequence at line {line_number}: "
                    f"expected {len(loaded)}, got {event.seq}"
                )
            loaded.append(event)
            valid_bytes += len(raw_line)
        if truncated_tail:
            with self.path.open("r+b") as handle:
                handle.truncate(valid_bytes)
        self._events = loaded
