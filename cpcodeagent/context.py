"""Build replayable model views from the immutable run journal."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from .journal import EventKind, Journal
from .types import ContextView, ModelResponse, ToolCall, ToolResult


class ContextEngine:
    def __init__(
        self,
        max_working_chars: int = 80_000,
        max_tool_output_chars: int = 12_000,
        snapshot_chars: int = 8_000,
    ):
        self.max_working_chars = max_working_chars
        self.max_tool_output_chars = max_tool_output_chars
        self.snapshot_chars = snapshot_chars

    def build(
        self,
        journal: Journal,
        task: str,
        workspace: str,
        policy: str,
        skill_catalog: Sequence[dict[str, str]] = (),
        force_compact: bool = False,
    ) -> ContextView:
        active_skill = self._active_skill(journal)
        system = self._system_prompt(task, workspace, policy, skill_catalog, active_skill)
        blocks = self._history_blocks(journal)
        limit = self.max_working_chars // 2 if force_compact else self.max_working_chars
        total = sum(_message_size(message) for block in blocks for message in block)

        memory: str | None = None
        selected = blocks
        if total > limit:
            selected, dropped = self._select_recent(blocks, max(4_000, int(limit * 0.65)))
            memory = self._snapshot(dropped)

        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        if memory:
            messages.append({"role": "user", "content": memory})
        for block in selected:
            messages.extend(block)
        return ContextView(tuple(messages), memory)

    def _system_prompt(
        self,
        task: str,
        workspace: str,
        policy: str,
        skill_catalog: Sequence[dict[str, str]],
        active_skill: str | None,
    ) -> str:
        skills = "\n".join(
            f"- {item['name']} ({item['version']}): {item['description']}"
            for item in skill_catalog
        ) or "- No skills discovered"
        active = active_skill or "No skill is active. Use use_skill only when a catalog entry fits."
        return f"""You are a coding agent operating through a constrained harness.

Current request:
{task}

Workspace: {workspace}
Execution policy: {policy}

Rules:
- Inspect relevant code before editing it.
- Use tools for facts and actions; never claim an action you did not observe.
- Treat a policy denial as a hard boundary and choose another approach.
- When the task is complete, respond without tool calls; the harness will verify it.

Available skills (descriptions only):
{skills}

Active skill instructions:
{active}
"""

    def _history_blocks(self, journal: Journal) -> list[list[dict[str, Any]]]:
        blocks: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] | None = None
        call_names: dict[str, str] = {}

        for event in journal.events:
            if event.kind is EventKind.INPUT:
                current = [{"role": "user", "content": event.data["content"]}]
                blocks.append(current)
            elif event.kind is EventKind.MODEL_RESPONSE:
                response = ModelResponse.from_dict(event.data["response"])
                for call in response.tool_calls:
                    call_names[call.id] = call.name
                message: dict[str, Any] = {
                    "role": "assistant",
                    "content": response.content or None,
                }
                if response.tool_calls:
                    message["tool_calls"] = [_wire_tool_call(call) for call in response.tool_calls]
                current = [message]
                blocks.append(current)
            elif event.kind is EventKind.TOOL_CALL:
                call = ToolCall.from_dict(event.data["call"])
                call_names[call.id] = call.name
            elif event.kind is EventKind.TOOL_RESULT:
                result = ToolResult.from_dict(event.data["result"])
                name = call_names.get(result.call_id, "")
                if name == "use_skill" and result.ok:
                    content = "Skill activated; its snapshotted instructions are in system context."
                else:
                    content = self._tool_content(result)
                message = {"role": "tool", "tool_call_id": result.call_id, "content": content}
                if current is None:
                    current = [message]
                    blocks.append(current)
                else:
                    current.append(message)
        return blocks

    def _active_skill(self, journal: Journal) -> str | None:
        call_names: dict[str, str] = {}
        active: str | None = None
        for event in journal.events:
            if event.kind is EventKind.MODEL_RESPONSE:
                response = ModelResponse.from_dict(event.data["response"])
                call_names.update({call.id: call.name for call in response.tool_calls})
            elif event.kind is EventKind.TOOL_CALL:
                call = ToolCall.from_dict(event.data["call"])
                call_names[call.id] = call.name
            elif event.kind is EventKind.TOOL_RESULT:
                result = ToolResult.from_dict(event.data["result"])
                if result.ok and call_names.get(result.call_id) == "use_skill":
                    active = result.output
        return active

    def _tool_content(self, result: ToolResult) -> str:
        status = "ok" if result.ok else f"error:{result.error or 'unknown'}"
        content = result.output
        if len(content) > self.max_tool_output_chars:
            head = self.max_tool_output_chars * 2 // 3
            tail = self.max_tool_output_chars - head
            content = f"{content[:head]}\n... tool output truncated ...\n{content[-tail:]}"
        artifacts = f"\nArtifacts: {', '.join(result.artifacts)}" if result.artifacts else ""
        return f"[{status}]\n{content}{artifacts}".strip()

    @staticmethod
    def _select_recent(
        blocks: list[list[dict[str, Any]]], target_chars: int
    ) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
        selected: list[list[dict[str, Any]]] = []
        size = 0
        for block in reversed(blocks):
            block_size = sum(_message_size(message) for message in block)
            if selected and size + block_size > target_chars:
                break
            selected.append(block)
            size += block_size
        selected.reverse()
        dropped_count = len(blocks) - len(selected)
        return selected, blocks[:dropped_count]

    def _snapshot(self, blocks: list[list[dict[str, Any]]]) -> str:
        lines: list[str] = []
        canonical = json.dumps(blocks, ensure_ascii=False, sort_keys=True)
        for block in blocks:
            for message in block:
                role = message["role"]
                content = str(message.get("content") or "").replace("\n", " ").strip()
                calls = message.get("tool_calls", ())
                call_names = ", ".join(call["function"]["name"] for call in calls)
                suffix = f" tools=[{call_names}]" if call_names else ""
                lines.append(f"- {role}: {content[:500]}{suffix}")
        digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        body = "\n".join(lines)
        if len(body) > self.snapshot_chars:
            body = body[: self.snapshot_chars] + "\n... snapshot truncated ..."
        return (
            f"[Memory snapshot of {len(blocks)} earlier blocks; source hash {digest}]\n"
            f"{body}\n[End memory snapshot]"
        )


def _wire_tool_call(call: ToolCall) -> dict[str, Any]:
    return {
        "id": call.id,
        "type": "function",
        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
    }


def _message_size(message: dict[str, Any]) -> int:
    return len(json.dumps(message, ensure_ascii=False))
