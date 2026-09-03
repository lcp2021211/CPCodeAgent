"""Lossless, redacted recording around the public CPCodeAgent interfaces."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

from cpcodeagent.model import Model
from cpcodeagent.types import ModelResponse, RunEvent, RunEventKind


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict"):
        return to_jsonable(value.to_dict())
    if dataclasses.is_dataclass(value):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(item) for item in value]
    return repr(value)


class Redactor:
    """Recursively redact known secrets without altering ordinary task content."""

    _SENSITIVE_KEYS: ClassVar[set[str]] = {
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "openai_api_key",
        "token",
    }

    def __init__(self, secret_values: Sequence[str] = ()):
        self._secrets = tuple(
            sorted({value for value in secret_values if value}, key=len, reverse=True)
        )

    @classmethod
    def from_environment(cls) -> Redactor:
        values = []
        for key, value in os.environ.items():
            lowered = key.lower()
            if any(part in lowered for part in ("api_key", "token", "secret", "password")):
                values.append(value)
        return cls(values)

    def apply(self, value: Any) -> Any:
        value = to_jsonable(value)
        if isinstance(value, str):
            for secret in self._secrets:
                value = value.replace(secret, "<redacted>")
            return value
        if isinstance(value, list):
            return [self.apply(item) for item in value]
        if isinstance(value, dict):
            redacted: dict[str, Any] = {}
            for key, item in value.items():
                if key.lower() in self._SENSITIVE_KEYS:
                    redacted[key] = "<redacted>"
                else:
                    redacted[key] = self.apply(item)
            return redacted
        return value


class JsonlWriter:
    def __init__(self, path: Path, redactor: Redactor):
        self.path = path
        self.redactor = redactor
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, value: dict[str, Any]) -> None:
        encoded = json.dumps(self.redactor.apply(value), ensure_ascii=False, sort_keys=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class RecordingState:
    """Shared writer and counters across primary retries and model fallback."""

    def __init__(self, output_path: Path, redactor: Redactor):
        self.writer = JsonlWriter(output_path, redactor)
        self.lock = threading.Lock()
        self.call_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.model_seconds = 0.0

    def next_call(self) -> int:
        with self.lock:
            call_index = self.call_count
            self.call_count += 1
            return call_index

    def add_result(self, duration: float, response: ModelResponse | None = None) -> None:
        with self.lock:
            self.model_seconds += duration
            if response is not None:
                self.prompt_tokens += response.prompt_tokens
                self.completion_tokens += response.completion_tokens

    def totals(self) -> dict[str, Any]:
        with self.lock:
            return {
                "model_calls": self.call_count,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "model_seconds": self.model_seconds,
            }


def write_json(path: Path, value: Any, redactor: Redactor | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (redactor or Redactor()).apply(value)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class RecordingModel(Model):
    """Record every exact model input and completed output as one SFT-ready step."""

    def __init__(
        self,
        inner: Model,
        output_path: Path,
        request_metadata: dict[str, Any],
        redactor: Redactor,
        state: RecordingState | None = None,
    ):
        self.inner = inner
        self.name = inner.name
        self.request_metadata = request_metadata
        self.state = state or RecordingState(output_path, redactor)
        self.writer = self.state.writer

    @property
    def call_count(self) -> int:
        return self.state.totals()["model_calls"]

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        on_text: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        call_index = self.state.next_call()

        request = {
            "messages": list(messages),
            "tools": list(tools),
        }
        started_at = utc_now()
        started = time.monotonic()
        streamed_text: list[str] = []

        def capture_text(text: str) -> None:
            streamed_text.append(text)
            if on_text is not None:
                on_text(text)

        try:
            response = self.inner.complete(messages, tools, capture_text)
        except Exception as exc:
            duration = time.monotonic() - started
            record = {
                "kind": "model_call",
                "call_index": call_index,
                "started_at": started_at,
                "finished_at": utc_now(),
                "duration_seconds": duration,
                "request": request,
                "request_hash": canonical_hash(request),
                "system_prompt_hash": canonical_hash(messages[0] if messages else {}),
                "tools_hash": canonical_hash(list(tools)),
                "request_metadata": self.request_metadata,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
            if streamed_text:
                record["partial_response"] = {"content": "".join(streamed_text)}
            self.writer.append(record)
            self.state.add_result(duration)
            raise

        duration = time.monotonic() - started
        response_dict = response.to_dict()
        self.writer.append(
            {
                "kind": "model_call",
                "call_index": call_index,
                "started_at": started_at,
                "finished_at": utc_now(),
                "duration_seconds": duration,
                "request": request,
                "request_hash": canonical_hash(request),
                "system_prompt_hash": canonical_hash(messages[0] if messages else {}),
                "tools_hash": canonical_hash(list(tools)),
                "request_metadata": self.request_metadata,
                "response": response_dict,
            }
        )
        self.state.add_result(duration, response)
        return response

    def totals(self) -> dict[str, Any]:
        return self.state.totals()


class RecordedModelGroup:
    """Expose one resilient model while aggregating its recorded attempts."""

    def __init__(self, inner: Model, state: RecordingState):
        self.inner = inner
        self.state = state
        self.name = inner.name

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        on_text: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        return self.inner.complete(messages, tools, on_text)

    def totals(self) -> dict[str, Any]:
        return self.state.totals()


class EventRecorder:
    """Persist structured progress events and stream text separately."""

    def __init__(
        self,
        directory: Path,
        redactor: Redactor,
        verbose: bool = True,
        prefix: str = "",
    ):
        self.writer = JsonlWriter(directory / "events.jsonl", redactor)
        self.stream_path = directory / "stream.txt"
        self.stream_path.parent.mkdir(parents=True, exist_ok=True)
        self._stream_lock = threading.Lock()
        self.verbose = verbose
        self.prefix = f"{prefix} " if prefix else ""

    def __call__(self, event: RunEvent) -> None:
        if event.kind is RunEventKind.TEXT_DELTA:
            text = str(event.data.get("text", ""))
            with self._stream_lock, self.stream_path.open("a", encoding="utf-8") as handle:
                handle.write(text)
            return

        payload = {
            "time": utc_now(),
            "kind": event.kind.value,
            "data": event.data,
        }
        self.writer.append(payload)
        if not self.verbose:
            return
        if event.kind is RunEventKind.MODEL_START:
            print(f"{self.prefix}model: {event.data.get('model', '?')}", flush=True)
        elif event.kind is RunEventKind.MODEL_END:
            print(f"{self.prefix}model response received", flush=True)
        elif event.kind is RunEventKind.TOOLS_START:
            calls = event.data.get("calls", ())
            names = [getattr(call, "name", "?") for call in calls]
            print(f"{self.prefix}tools: {', '.join(names)}", flush=True)
        elif event.kind is RunEventKind.TOOLS_END:
            print(f"{self.prefix}tools completed", flush=True)
        elif event.kind is RunEventKind.VERIFY_START:
            print(f"{self.prefix}verifying patch", flush=True)
        elif event.kind is RunEventKind.VERIFY_END:
            print(
                f"{self.prefix}patch verifier: {event.data.get('passed')}",
                flush=True,
            )
