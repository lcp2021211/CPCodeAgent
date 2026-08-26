"""A deliberately small coding-tool surface built on the Action contract."""

from __future__ import annotations

import difflib
from collections.abc import Callable
from typing import Any, ClassVar

from .executor import ExecutionEnv, ExecutionError
from .recovery import EffectContract, FileCondition, content_digest
from .skills import SkillRegistry
from .tools import Tool, ToolExecution, ToolRuntime
from .types import Action, Capability


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a UTF-8 text file inside the workspace with line numbers."
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer"},
            "limit": {"type": "integer"},
        },
        "required": ["path"],
    }

    def classify(self, arguments: dict[str, Any], env: ExecutionEnv) -> Action:
        target = env.executor.resolve(arguments["path"])
        return Action(frozenset({Capability.READ}), (f"file:{target}",), True)

    def execute(self, arguments: dict[str, Any], env: ExecutionEnv) -> ToolExecution:
        return ToolExecution(
            True,
            env.executor.read_text(
                arguments["path"],
                int(arguments.get("offset", 1)),
                int(arguments.get("limit", 2_000)),
            ),
        )


class ListFilesTool(Tool):
    name = "list_files"
    description = "List files inside the workspace using a glob pattern."
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "limit": {"type": "integer"},
        },
    }

    def classify(self, arguments: dict[str, Any], env: ExecutionEnv) -> Action:
        return Action(frozenset({Capability.READ}), (f"workspace:{env.workspace}",), True)

    def execute(self, arguments: dict[str, Any], env: ExecutionEnv) -> ToolExecution:
        files = env.executor.list_files(
            arguments.get("pattern", "**/*"), int(arguments.get("limit", 500))
        )
        return ToolExecution(True, "\n".join(files) if files else "(no files)")


class SearchTextTool(Tool):
    name = "search_text"
    description = "Search literal text in workspace files."
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "pattern": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["query"],
    }

    def classify(self, arguments: dict[str, Any], env: ExecutionEnv) -> Action:
        return Action(frozenset({Capability.READ}), (f"workspace:{env.workspace}",), True)

    def execute(self, arguments: dict[str, Any], env: ExecutionEnv) -> ToolExecution:
        hits = env.executor.search_text(
            arguments["query"],
            arguments.get("pattern", "**/*"),
            int(arguments.get("limit", 200)),
        )
        return ToolExecution(True, "\n".join(hits) if hits else "(no matches)")


class WriteFileTool(Tool):
    name = "write_file"
    description = "Create or replace a UTF-8 text file inside the workspace."
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    }

    def classify(self, arguments: dict[str, Any], env: ExecutionEnv) -> Action:
        target = env.executor.resolve(arguments["path"])
        return Action(frozenset({Capability.WORKSPACE_WRITE}), (f"file:{target}",), False)

    def effect_contract(
        self,
        arguments: dict[str, Any],
        action: Action,
        env: ExecutionEnv,
        action_id: str,
    ) -> EffectContract:
        target = env.executor.resolve(arguments["path"])
        relative = target.relative_to(env.workspace).as_posix()
        return EffectContract.file_transition(
            FileCondition(relative, env.executor.file_digest(relative)),
            FileCondition(relative, content_digest(arguments["content"])),
        )

    def execute(self, arguments: dict[str, Any], env: ExecutionEnv) -> ToolExecution:
        env.executor.write_text(
            arguments["path"],
            arguments["content"],
            operation_id=env.action_id,
        )
        return ToolExecution(True, f"Wrote {arguments['path']}")


class EditFileTool(Tool):
    name = "edit_file"
    description = "Replace one exact, unique string in a UTF-8 workspace file."
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old": {"type": "string"},
            "new": {"type": "string"},
        },
        "required": ["path", "old", "new"],
    }

    def classify(self, arguments: dict[str, Any], env: ExecutionEnv) -> Action:
        target = env.executor.resolve(arguments["path"])
        return Action(frozenset({Capability.WORKSPACE_WRITE}), (f"file:{target}",), False)

    def effect_contract(
        self,
        arguments: dict[str, Any],
        action: Action,
        env: ExecutionEnv,
        action_id: str,
    ) -> EffectContract:
        path = env.executor.resolve(arguments["path"])
        if not path.is_file():
            raise ExecutionError("NOT_FOUND", f"File does not exist: {arguments['path']}")
        original = path.read_text(encoding="utf-8")
        count = original.count(arguments["old"])
        if count != 1:
            raise ExecutionError(
                "AMBIGUOUS_EDIT",
                f"Expected old text exactly once, found {count} occurrences.",
            )
        relative = path.relative_to(env.workspace).as_posix()
        updated = original.replace(arguments["old"], arguments["new"], 1)
        return EffectContract.file_transition(
            FileCondition(relative, content_digest(original)),
            FileCondition(relative, content_digest(updated)),
        )

    def execute(self, arguments: dict[str, Any], env: ExecutionEnv) -> ToolExecution:
        path = env.executor.resolve(arguments["path"])
        if not path.is_file():
            raise ExecutionError("NOT_FOUND", f"File does not exist: {arguments['path']}")
        original = path.read_text(encoding="utf-8")
        count = original.count(arguments["old"])
        if count != 1:
            return ToolExecution(
                False,
                f"Expected old text exactly once, found {count} occurrences.",
                "AMBIGUOUS_EDIT",
            )
        updated = original.replace(arguments["old"], arguments["new"], 1)
        env.executor.write_text(
            arguments["path"],
            updated,
            operation_id=env.action_id,
        )
        diff = "".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile=f"a/{arguments['path']}",
                tofile=f"b/{arguments['path']}",
                n=3,
            )
        )
        return ToolExecution(True, diff or f"Edited {arguments['path']}")


class RunCommandTool(Tool):
    name = "run_command"
    description = "Run an argv command in the workspace. Shell syntax is intentionally unsupported."
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "argv": {"type": "array", "items": {"type": "string"}},
            "timeout": {"type": "number"},
        },
        "required": ["argv"],
    }

    def classify(self, arguments: dict[str, Any], env: ExecutionEnv) -> Action:
        return Action(
            frozenset({Capability.WORKSPACE_WRITE}),
            (f"workspace:{env.workspace}",),
            False,
        )

    def execute(self, arguments: dict[str, Any], env: ExecutionEnv) -> ToolExecution:
        result = env.executor.run(
            arguments["argv"], timeout=float(arguments.get("timeout", 120)), network=False
        )
        output = result.output or "(no output)"
        if result.returncode != 0:
            return ToolExecution(
                False,
                f"{output}\n[exit code: {result.returncode}]",
                "COMMAND_FAILED",
            )
        return ToolExecution(True, output)


class UseSkillTool(Tool):
    name = "use_skill"
    description = "Load one skill's complete instructions into the current run context."
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    def __init__(self, registry: SkillRegistry, available_tools: Callable[[], frozenset[str]]):
        self.registry = registry
        self.available_tools = available_tools

    def classify(self, arguments: dict[str, Any], env: ExecutionEnv) -> Action:
        self.registry.get(arguments["name"])
        return Action(frozenset({Capability.READ}), (f"skill:{arguments['name']}",), True)

    def execute(self, arguments: dict[str, Any], env: ExecutionEnv) -> ToolExecution:
        skill = self.registry.get(arguments["name"])
        missing = sorted(set(skill.required_tools) - self.available_tools())
        if missing:
            return ToolExecution(
                False,
                f"Skill requires unavailable tools: {', '.join(missing)}",
                "SKILL_INCOMPATIBLE",
            )
        content = (
            f"# Active skill: {skill.name}\n"
            f"Version: {skill.digest}\n\n{skill.instructions}"
        )
        return ToolExecution(True, content, artifacts=(f"skill:{skill.name}@{skill.digest}",))


class ReadSkillResourceTool(Tool):
    name = "read_skill_resource"
    description = "Read a supporting file from a discovered skill directory."
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "path": {"type": "string"}},
        "required": ["name", "path"],
    }

    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    def classify(self, arguments: dict[str, Any], env: ExecutionEnv) -> Action:
        self.registry.get(arguments["name"])
        return Action(
            frozenset({Capability.READ}),
            (f"skill-resource:{arguments['name']}/{arguments['path']}",),
            True,
        )

    def execute(self, arguments: dict[str, Any], env: ExecutionEnv) -> ToolExecution:
        return ToolExecution(
            True, self.registry.read_resource(arguments["name"], arguments["path"])
        )


def build_default_runtime(skills: SkillRegistry | None = None) -> ToolRuntime:
    runtime = ToolRuntime(
        [
            ReadFileTool(),
            ListFilesTool(),
            SearchTextTool(),
            WriteFileTool(),
            EditFileTool(),
            RunCommandTool(),
        ]
    )
    if skills is not None:
        runtime.register(UseSkillTool(skills, lambda: runtime.names))
        runtime.register(ReadSkillResourceTool(skills))
    return runtime
