"""Provider-neutral model interface and a small OpenAI-compatible adapter."""

from __future__ import annotations

import json
import random
import time
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from .types import ModelResponse, ToolCall


class ModelError(RuntimeError):
    pass


class TransientModelError(ModelError):
    pass


class ContextOverflowError(ModelError):
    pass


class Model(Protocol):
    name: str

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        on_text: Callable[[str], None] | None = None,
    ) -> ModelResponse: ...


class ScriptedModel:
    """Deterministic model for tests, examples, and offline development."""

    def __init__(self, responses: Sequence[ModelResponse], name: str = "scripted"):
        self.name = name
        self._responses = deque(responses)
        self.requests: list[tuple[Sequence[dict[str, Any]], Sequence[dict[str, Any]]]] = []

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        on_text: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        self.requests.append((messages, tools))
        if not self._responses:
            raise ModelError("ScriptedModel ran out of responses")
        response = self._responses.popleft()
        if on_text is not None and response.content:
            on_text(response.content)
        if response.model:
            return response
        return ModelResponse(
            content=response.content,
            tool_calls=response.tool_calls,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            model=self.name,
        )


class OpenAICompatibleModel:
    """Adapter for OpenAI and OpenAI-compatible chat-completions APIs."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        client: Any | None = None,
        **request_options: Any,
    ):
        self.name = model
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - optional runtime
                raise RuntimeError("Install the 'openai' package to use this model adapter") from exc
            client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        self._client = client
        self._options = request_options

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        on_text: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        params: dict[str, Any] = {
            "model": self.name,
            "messages": list(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
            **self._options,
        }
        if tools:
            params["tools"] = list(tools)

        stream = self._open_stream(params)
        content: list[str] = []
        visible = _VisibleContentFilter()
        tool_fragments: dict[int, dict[str, str]] = {}
        prompt_tokens = 0
        completion_tokens = 0
        try:
            for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    completion_tokens = getattr(usage, "completion_tokens", 0) or 0

                choices = getattr(chunk, "choices", None) or ()
                if not choices:
                    continue
                delta = choices[0].delta
                text = getattr(delta, "content", None)
                if text:
                    exposed = visible.feed(text)
                    if exposed:
                        content.append(exposed)
                        if on_text is not None:
                            on_text(exposed)

                # Compatible providers may expose private reasoning separately as
                # ``reasoning_content``. It is deliberately neither streamed nor
                # copied into the durable assistant response.

                for raw_delta in getattr(delta, "tool_calls", None) or ():
                    index = getattr(raw_delta, "index", None)
                    index = len(tool_fragments) if index is None else int(index)
                    fragment = tool_fragments.setdefault(
                        index, {"id": "", "name": "", "arguments": ""}
                    )
                    if getattr(raw_delta, "id", None):
                        fragment["id"] = raw_delta.id
                    function = getattr(raw_delta, "function", None)
                    if function is not None:
                        if getattr(function, "name", None):
                            fragment["name"] = function.name
                        if getattr(function, "arguments", None):
                            fragment["arguments"] += function.arguments
        # Provider SDK exception classes vary across compatible endpoints; this adapter
        # normalizes every provider failure into the harness error taxonomy.
        except Exception as exc:  # noqa: BLE001
            self._raise_normalized(exc)
            raise AssertionError("unreachable")

        tail = visible.finish()
        if tail:
            content.append(tail)
            if on_text is not None:
                on_text(tail)

        calls = tuple(
            ToolCall(
                id=value["id"] or f"stream-call-{index}",
                name=value["name"],
                arguments=_parse_arguments(value["arguments"]),
            )
            for index, value in sorted(tool_fragments.items())
        )
        return ModelResponse(
            content="".join(content),
            tool_calls=calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=self.name,
        )

    def _open_stream(self, params: dict[str, Any]) -> Any:
        try:
            return self._client.chat.completions.create(**params)
        except Exception as exc:  # noqa: BLE001
            if getattr(exc, "status_code", None) == 400 and "stream_options" in params:
                compatible = dict(params)
                compatible.pop("stream_options")
                try:
                    return self._client.chat.completions.create(**compatible)
                except Exception as fallback_exc:  # noqa: BLE001
                    self._raise_normalized(fallback_exc)
            self._raise_normalized(exc)
        raise AssertionError("unreachable")

    @staticmethod
    def _raise_normalized(exc: Exception) -> None:
        status = getattr(exc, "status_code", None)
        message = str(exc)
        lowered = message.lower()
        if "context" in lowered and any(word in lowered for word in ("length", "window", "token")):
            raise ContextOverflowError(message) from exc
        if status == 429 or (isinstance(status, int) and status >= 500):
            raise TransientModelError(message) from exc
        if any(token in exc.__class__.__name__.lower() for token in ("timeout", "connection")):
            raise TransientModelError(message) from exc
        raise ModelError(message) from exc


class _VisibleContentFilter:
    """Remove provider-emitted thinking blocks across arbitrary stream chunks."""

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._pending = ""
        self._hidden = False
        self._visible_started = False

    def feed(self, text: str) -> str:
        data = self._pending + text
        self._pending = ""
        output: list[str] = []

        while data:
            markers = (self._CLOSE,) if self._hidden else (self._OPEN, self._CLOSE)
            match = self._first_marker(data, markers)
            if match is not None:
                index, marker = match
                if not self._hidden:
                    self._append_visible(output, data[:index])
                data = data[index + len(marker) :]
                if self._hidden:
                    self._hidden = False
                elif marker == self._OPEN:
                    self._hidden = True
                # An unmatched closing tag is provider framing too; omit it.
                continue

            retained = self._partial_marker_length(data, markers)
            ready = data[:-retained] if retained else data
            if not self._hidden:
                self._append_visible(output, ready)
            self._pending = data[-retained:] if retained else ""
            break

        return "".join(output)

    def finish(self) -> str:
        pending = self._pending
        self._pending = ""
        if self._hidden:
            return ""
        output: list[str] = []
        self._append_visible(output, pending)
        return "".join(output)

    def _append_visible(self, output: list[str], value: str) -> None:
        if not self._visible_started:
            value = value.lstrip()
            if not value:
                return
            self._visible_started = True
        output.append(value)

    @staticmethod
    def _first_marker(data: str, markers: tuple[str, ...]) -> tuple[int, str] | None:
        matches = ((data.find(marker), marker) for marker in markers)
        present = tuple(item for item in matches if item[0] >= 0)
        return min(present, default=None, key=lambda item: item[0])

    @staticmethod
    def _partial_marker_length(data: str, markers: tuple[str, ...]) -> int:
        maximum = min(len(data), max(len(marker) for marker in markers) - 1)
        for length in range(maximum, 0, -1):
            if any(marker.startswith(data[-length:]) for marker in markers):
                return length
        return 0


class ResilientModel:
    """Owns API retry and one explicit provider fallback."""

    def __init__(
        self,
        primary: Model,
        fallback: Model | None = None,
        max_attempts: int = 3,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.primary = primary
        self.fallback = fallback
        self.max_attempts = max_attempts
        self._sleep = sleeper
        self.name = primary.name

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        on_text: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        last_error: TransientModelError | None = None
        emitted = False

        def track(text: str) -> None:
            nonlocal emitted
            emitted = True
            if on_text is not None:
                on_text(text)

        for attempt in range(self.max_attempts):
            try:
                return self.primary.complete(messages, tools, track)
            except ContextOverflowError:
                raise
            except TransientModelError as exc:
                if emitted:
                    raise ModelError(
                        "Model stream was interrupted after output began; automatic retry "
                        "was suppressed to avoid duplicate text."
                    ) from exc
                last_error = exc
                if attempt + 1 < self.max_attempts:
                    delay = (2**attempt) + random.uniform(0, 0.25)
                    self._sleep(delay)
        if self.fallback is not None:
            try:
                return self.fallback.complete(messages, tools, track)
            except ModelError as exc:
                raise ModelError(
                    f"Primary model failed ({last_error}); fallback failed ({exc})"
                ) from exc
        assert last_error is not None
        raise last_error


def _parse_arguments(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"__raw__": raw}
    return value if isinstance(value, dict) else {"__raw__": raw}
