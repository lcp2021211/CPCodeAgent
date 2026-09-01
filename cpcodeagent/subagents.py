"""Bounded child agents with isolated context, execution, and result transfer."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

from .builtin_tools import (
    EditFileTool,
    ListFilesTool,
    PlanWriteTool,
    ReadFileTool,
    ReadSkillResourceTool,
    RunCommandTool,
    SearchTextTool,
    UseSkillTool,
    WriteFileTool,
)
from .context import ContextEngine
from .executor import DockerExecutor, ExecutionEnv, Executor, LocalExecutor
from .journal import EventKind, Journal
from .kernel import Harness
from .model import Model
from .policy import RunPolicy
from .recovery import EffectContract
from .skills import SkillRegistry
from .tools import Tool, ToolExecution, ToolRuntime
from .types import (
    Action,
    Capability,
    Decision,
    RunLimits,
    RunOutcome,
    RunStatus,
    ToolCall,
    ToolResult,
)

_ACTION_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_IGNORED_PARTS = frozenset(
    {
        ".git",
        ".cpcodeagent",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "node_modules",
        "venv",
    }
)


class SubagentMode(str, Enum):
    INSPECT = "inspect"
    PATCH = "patch"


class SubagentStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass(frozen=True)
class SubagentResult:
    """The only child-agent data allowed to cross into the parent context."""

    status: SubagentStatus
    summary: str
    evidence: tuple[str, ...] = ()
    recommendation: str = ""
    artifact_id: str | None = None
    changed_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary", _bounded_text(self.summary, 2_000, "summary"))
        object.__setattr__(
            self,
            "evidence",
            tuple(_bounded_text(item, 500, "evidence item") for item in self.evidence[:8]),
        )
        object.__setattr__(
            self,
            "recommendation",
            _bounded_text(self.recommendation, 1_000, "recommendation", allow_empty=True),
        )
        object.__setattr__(self, "changed_paths", tuple(self.changed_paths[:100]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "summary": self.summary,
            "evidence": list(self.evidence),
            "recommendation": self.recommendation,
            "artifact_id": self.artifact_id,
            "changed_paths": list(self.changed_paths),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubagentResult:
        return cls(
            SubagentStatus(str(data["status"])),
            str(data["summary"]),
            tuple(str(item) for item in data.get("evidence", ())),
            str(data.get("recommendation", "")),
            str(data["artifact_id"]) if data.get("artifact_id") else None,
            tuple(str(item) for item in data.get("changed_paths", ())),
        )


class SubmitResultTool(Tool):
    """Child-only terminal handoff; status remains owned by the runtime."""

    name = "submit_result"
    description = (
        "Submit the concise result of this delegated task. Include only decisive evidence "
        "and a recommendation; never include the full trajectory."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "recommendation": {"type": "string"},
        },
        "required": ["summary", "evidence", "recommendation"],
    }

    def classify(self, arguments: dict[str, Any], env: ExecutionEnv) -> Action:
        _submission(arguments)
        return Action(
            frozenset({Capability.RUNTIME_WRITE}),
            ("runtime:subagent-result",),
            True,
        )

    def execute(self, arguments: dict[str, Any], env: ExecutionEnv) -> ToolExecution:
        summary, evidence, recommendation = _submission(arguments)
        return ToolExecution(
            True,
            json.dumps(
                {
                    "summary": summary,
                    "evidence": list(evidence),
                    "recommendation": recommendation,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        )


class _CompatibleSkillRegistry(SkillRegistry):
    """A child view that hides skills requiring tools outside its boundary."""

    def __init__(self, source: SkillRegistry, available_tools: frozenset[str]):
        self.source = source
        self.available_tools = available_tools

    def catalog(self) -> tuple[dict[str, str], ...]:
        return tuple(
            item
            for item in self.source.catalog()
            if set(self.source.get(item["name"]).required_tools) <= self.available_tools
        )

    def get(self, name: str):
        skill = self.source.get(name)
        missing = set(skill.required_tools) - self.available_tools
        if missing:
            raise KeyError(f"Skill is incompatible with this child: {name}")
        return skill

    def read_resource(self, name: str, relative_path: str) -> str:
        self.get(name)
        return self.source.read_resource(name, relative_path)


class SubagentRunner:
    """Runs one retryable child task under an action-ID-scoped durable directory."""

    def __init__(
        self,
        model: Model,
        root: str | Path,
        skills: SkillRegistry | None = None,
        limits: RunLimits | None = None,
        max_context_tokens: int = 64_000,
    ):
        self.model = model
        self.root = Path(root).expanduser().resolve()
        self.skills = skills or SkillRegistry()
        self.limits = limits or RunLimits(max_steps=12, max_seconds=600, max_tokens=60_000)
        self.max_context_tokens = max_context_tokens

    def run(
        self,
        task: str,
        mode: SubagentMode,
        parent_executor: Executor,
        action_id: str,
    ) -> SubagentResult:
        try:
            self.root.relative_to(parent_executor.workspace)
        except ValueError:
            pass
        else:
            raise ValueError("Subagent state must live outside the parent workspace")
        run_dir = self._run_dir(action_id)
        cached = run_dir / "result.json"
        if cached.is_file():
            return SubagentResult.from_dict(json.loads(cached.read_text(encoding="utf-8")))

        run_dir.mkdir(parents=True, exist_ok=True)
        child_executor = self._executor(mode, parent_executor, run_dir)
        journal = Journal(run_dir / "journal.jsonl")
        tools, child_skills = self._tools(
            mode, allow_commands=parent_executor.hard_sandbox
        )
        policy = RunPolicy(
            workspace_write=mode is SubagentMode.PATCH,
            allowed_hosts=(),
            external_writes=Decision.DENY,
        )
        harness = Harness(
            model=self.model,
            tools=tools,
            context=ContextEngine(max_context_tokens=self.max_context_tokens),
            skills=child_skills,
            policy=policy,
            limits=self.limits,
            memory=None,
        )

        if journal.events:
            outcome = harness.resume(child_executor, journal)
        else:
            outcome = harness.run(
                self._task_prompt(task, mode),
                child_executor,
                journal,
                f"child-{action_id[:24]}",
            )

        result = self._result_from_journal(journal, outcome)
        if mode is SubagentMode.PATCH:
            try:
                artifact_id, changed = self._write_patch_artifact(
                    action_id,
                    parent_executor.workspace,
                    child_executor.workspace,
                    run_dir,
                )
                result = replace(result, artifact_id=artifact_id, changed_paths=changed)
            except ValueError as exc:
                evidence = (*result.evidence[:7], f"Patch artifact omitted: {exc}")
                result = replace(
                    result,
                    status=SubagentStatus.PARTIAL,
                    evidence=evidence,
                    recommendation=(
                        "Review the isolated workspace before reproducing any child changes."
                    ),
                )
        _atomic_json(cached, result.to_dict())
        return result

    def _run_dir(self, action_id: str) -> Path:
        if not _ACTION_ID.fullmatch(action_id):
            raise ValueError("Invalid subagent action ID")
        return self.root / action_id

    def _executor(
        self,
        mode: SubagentMode,
        parent: Executor,
        run_dir: Path,
    ) -> Executor:
        if mode is SubagentMode.INSPECT:
            return parent
        overlay = run_dir / "workspace"
        baseline = run_dir / "baseline.json"
        if not overlay.is_dir():
            temporary = Path(tempfile.mkdtemp(prefix="workspace-", dir=run_dir))
            try:
                _copy_workspace(parent.workspace, temporary)
                _atomic_json(baseline, _digest_map(temporary))
                os.replace(temporary, overlay)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        elif not baseline.is_file():
            raise RuntimeError("Subagent overlay exists without its baseline")
        if parent.hard_sandbox:
            return DockerExecutor(overlay, image=str(getattr(parent, "image", "python:3.12-slim")))
        return LocalExecutor(overlay)

    def _tools(
        self, mode: SubagentMode, *, allow_commands: bool
    ) -> tuple[ToolRuntime, SkillRegistry]:
        values: list[Tool] = [
            ReadFileTool(),
            ListFilesTool(),
            SearchTextTool(),
            PlanWriteTool(),
        ]
        if mode is SubagentMode.PATCH:
            values.extend((WriteFileTool(), EditFileTool()))
            if allow_commands:
                values.append(RunCommandTool())
        runtime = ToolRuntime(values)
        skill_tools = frozenset({"use_skill", "read_skill_resource", "submit_result"})
        child_skills = _CompatibleSkillRegistry(self.skills, runtime.names | skill_tools)
        if child_skills.catalog():
            runtime.register(UseSkillTool(child_skills, lambda: runtime.names))
            runtime.register(ReadSkillResourceTool(child_skills))
        runtime.register(SubmitResultTool())
        return runtime, child_skills

    @staticmethod
    def _task_prompt(task: str, mode: SubagentMode) -> str:
        boundary = (
            "Inspect only. You cannot modify files or run commands."
            if mode is SubagentMode.INSPECT
            else (
                "Work only in the isolated overlay. Your edits do not enter the parent "
                "workspace; report exactly which files you changed."
            )
        )
        return (
            "You are a bounded child agent. You receive no parent transcript, plan, or memory. "
            "Complete only the delegated task below. You cannot create another agent. "
            f"{boundary} Use submit_result once with a concise summary, decisive evidence, and "
            "recommendation, then end with one short confirmation.\n\nDelegated task:\n"
            f"{task.strip()}"
        )

    @staticmethod
    def _result_from_journal(journal: Journal, outcome: RunOutcome) -> SubagentResult:
        submitted = _last_submission(journal)
        if submitted is not None:
            summary, evidence, recommendation = submitted
            status = (
                SubagentStatus.COMPLETED
                if outcome.status is RunStatus.SUCCEEDED
                else SubagentStatus.PARTIAL
            )
            return SubagentResult(status, summary, evidence, recommendation)
        status = (
            SubagentStatus.PARTIAL
            if outcome.status is RunStatus.SUCCEEDED
            else SubagentStatus.FAILED
        )
        return SubagentResult(
            status,
            _bounded_text(outcome.answer or "Child task produced no result.", 2_000, "summary"),
            (),
            "Review the child journal or delegate a narrower task.",
        )

    @staticmethod
    def _write_patch_artifact(
        action_id: str,
        parent_workspace: Path,
        overlay: Path,
        run_dir: Path,
    ) -> tuple[str | None, tuple[str, ...]]:
        artifact_path = run_dir / "patch.json"
        if artifact_path.is_file():
            data = json.loads(artifact_path.read_text(encoding="utf-8"))
            return action_id, tuple(str(item["path"]) for item in data.get("changes", ()))

        baseline_path = run_dir / "baseline.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        after = _digest_map(overlay)
        changed = tuple(
            sorted(path for path in set(baseline) | set(after) if baseline.get(path) != after.get(path))
        )
        if not changed:
            return None, ()
        if len(changed) > 100:
            raise ValueError("more than 100 files changed")

        changes: list[dict[str, Any]] = []
        artifact_bytes = 0
        for relative in changed:
            target = overlay / relative
            if target.is_symlink():
                raise ValueError("symbolic-link changes are unsupported")
            raw = target.read_bytes() if target.is_file() else None
            artifact_bytes += len(raw or b"")
            if artifact_bytes > 10_000_000:
                raise ValueError("changed file content exceeds 10 MB")
            content = base64.b64encode(raw).decode("ascii") if raw is not None else None
            changes.append(
                {
                    "path": relative,
                    "before_digest": baseline.get(relative),
                    "after_digest": after.get(relative),
                    "content_base64": content,
                }
            )
        _atomic_json(
            artifact_path,
            {
                "artifact_id": action_id,
                "workspace": str(parent_workspace),
                "changes": changes,
            },
        )
        return action_id, changed


class DelegateTaskTool(Tool):
    name = "delegate_task"
    description = (
        "Delegate one bounded independent task to a child agent with fresh context. "
        "inspect is read-only; patch writes only to an isolated overlay. The child returns "
        "a concise structured result, never its trajectory."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "task": {"type": "string"},
            "mode": {"type": "string", "enum": ["inspect", "patch"]},
        },
        "required": ["task", "mode"],
    }

    def __init__(self, runner: SubagentRunner):
        self.runner = runner

    def classify(self, arguments: dict[str, Any], env: ExecutionEnv) -> Action:
        _delegation(arguments)
        return Action(
            frozenset({Capability.RUNTIME_WRITE}),
            (f"subagent:{arguments['mode']}",),
            True,
        )

    def effect_contract(
        self,
        arguments: dict[str, Any],
        action: Action,
        env: ExecutionEnv,
        action_id: str,
    ) -> EffectContract:
        return EffectContract.retry_safe()

    def execute(self, arguments: dict[str, Any], env: ExecutionEnv) -> ToolExecution:
        task, mode = _delegation(arguments)
        if env.action_id is None:
            raise RuntimeError("delegate_task requires a durable action ID")
        result = self.runner.run(task, mode, env.executor, env.action_id)
        artifacts = (f"subagent:{env.action_id}",)
        if result.artifact_id:
            artifacts += (f"patch:{result.artifact_id}",)
        return ToolExecution(
            result.status is not SubagentStatus.FAILED,
            result.to_json(),
            "SUBAGENT_FAILED" if result.status is SubagentStatus.FAILED else None,
            artifacts,
        )


def _delegation(arguments: dict[str, Any]) -> tuple[str, SubagentMode]:
    task = _bounded_text(str(arguments["task"]), 4_000, "task")
    return task, SubagentMode(str(arguments["mode"]))


def _submission(arguments: dict[str, Any]) -> tuple[str, tuple[str, ...], str]:
    evidence = arguments["evidence"]
    if not isinstance(evidence, list):
        raise TypeError("evidence must be a list")
    summary = _bounded_text(str(arguments["summary"]), 2_000, "summary")
    items = tuple(_bounded_text(str(item), 500, "evidence item") for item in evidence[:8])
    recommendation = _bounded_text(
        str(arguments["recommendation"]), 1_000, "recommendation", allow_empty=True
    )
    return summary, items, recommendation


def _last_submission(journal: Journal) -> tuple[str, tuple[str, ...], str] | None:
    calls: dict[str, ToolCall] = {}
    submitted: tuple[str, tuple[str, ...], str] | None = None
    for event in journal.events:
        if event.kind is EventKind.TOOL_CALL:
            call = ToolCall.from_dict(event.data["call"])
            calls[call.id] = call
        elif event.kind is EventKind.TOOL_RESULT:
            result = ToolResult.from_dict(event.data["result"])
            call = calls.get(result.call_id)
            if result.ok and call is not None and call.name == "submit_result":
                submitted = _submission(call.arguments)
    return submitted


def _bounded_text(value: str, limit: int, name: str, *, allow_empty: bool = False) -> str:
    value = " ".join(value.strip().split())
    if not value and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    if len(value) > limit:
        value = value[: limit - 1] + "…"
    return value


def _copy_workspace(source: Path, destination: Path) -> None:
    for root, directories, files in os.walk(source, followlinks=False):
        root_path = Path(root)
        directories[:] = [
            name
            for name in directories
            if name not in _IGNORED_PARTS and not (root_path / name).is_symlink()
        ]
        relative_root = root_path.relative_to(source)
        target_root = destination / relative_root
        target_root.mkdir(parents=True, exist_ok=True)
        for name in files:
            path = root_path / name
            relative = path.relative_to(source)
            if path.is_symlink() or any(part in _IGNORED_PARTS for part in relative.parts):
                continue
            shutil.copy2(path, target_root / name)


def _digest_map(root: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for directory, directories, files in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        directories[:] = [
            name
            for name in directories
            if name not in _IGNORED_PARTS and not (directory_path / name).is_symlink()
        ]
        for name in files:
            path = directory_path / name
            relative_path = path.relative_to(root)
            if path.is_symlink() or any(
                part in _IGNORED_PARTS for part in relative_path.parts
            ):
                continue
            relative = relative_path.as_posix()
            values[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return values


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
