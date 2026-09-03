"""Container-backed tools for SWE-smith tasks.

The classes in this file adapt the public CPCodeAgent extension interfaces.  No
code under ``cpcodeagent/`` is patched or monkey-patched.
"""

from __future__ import annotations

import difflib
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

import docker
from docker.models.containers import Container

from cpcodeagent.builtin_tools import PlanWriteTool
from cpcodeagent.executor import (
    CommandResult,
    ExecutionEnv,
    ExecutionError,
    Executor,
    WorkspaceSnapshot,
)
from cpcodeagent.tools import Tool, ToolExecution, ToolRuntime
from cpcodeagent.types import Action, Capability, Verification

CONTAINER_WORKDIR = PurePosixPath("/testbed")


def safe_relative(path: str) -> PurePosixPath:
    value = PurePosixPath(path)
    if not path or value.is_absolute() or ".." in value.parts:
        raise ExecutionError("OUTSIDE_WORKSPACE", f"Path escapes workspace: {path}")
    normalized = PurePosixPath(*[part for part in value.parts if part not in ("", ".")])
    if not normalized.parts:
        raise ExecutionError("INVALID_PATH", "Path must name a workspace entry")
    return normalized


def safe_pattern(pattern: str) -> str:
    value = PurePosixPath(pattern)
    if value.is_absolute() or ".." in value.parts:
        raise ExecutionError("OUTSIDE_WORKSPACE", f"Pattern escapes workspace: {pattern}")
    return pattern or "**/*"


class PersistentContainerExecutor(Executor):
    """Run commands in one initialized SWE-smith task container."""

    hard_sandbox = True

    def __init__(
        self,
        container: Container,
        image: str,
        host_cwd: Path,
        output_limit: int = 20_000,
    ):
        # Executor.__init__ requires a host directory.  This executor deliberately
        # exposes the actual model-visible container path instead.
        self.workspace = Path(str(CONTAINER_WORKDIR))
        self.output_limit = output_limit
        self.container = container
        self.image = image
        self.host_cwd = host_cwd.resolve()
        self.docker_binary = shutil.which("docker")
        if self.docker_binary is None:
            raise ValueError("Docker executable is not available")

    def resolve(self, path: str | Path) -> Path:
        # The experiment tools never dereference this host-side Path.  It exists
        # for Action target descriptions and policy checks only.
        return Path(str(CONTAINER_WORKDIR / safe_relative(str(path))))

    def run(
        self,
        argv: list[str] | tuple[str, ...],
        timeout: float = 120,
        network: bool = False,
    ) -> CommandResult:
        if network:
            raise ExecutionError("NETWORK_DENIED", "Task containers run without network")
        if not argv or not all(isinstance(part, str) and part for part in argv):
            raise ExecutionError("INVALID_COMMAND", "argv must contain non-empty strings")
        seconds = max(1, int(timeout))
        command = [
            self.docker_binary,
            "exec",
            "--user",
            "root",
            "--workdir",
            str(CONTAINER_WORKDIR),
            self.container.id,
            "timeout",
            "--signal=KILL",
            "--kill-after=5s",
            f"{seconds}s",
            *argv,
        ]
        try:
            process = subprocess.run(
                command,
                cwd=self.host_cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout + 10,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ExecutionError("TIMEOUT", f"Command timed out after {timeout}s") from exc
        return CommandResult(process.returncode, self._truncate(process.stdout or ""))

    def snapshot(self) -> WorkspaceSnapshot:
        status = self._exec(["git", "status", "--porcelain=v1", "--untracked-files=all"])
        diff = self._exec(["git", "diff", "--binary", "--no-ext-diff", "HEAD"])
        lines = tuple(line for line in status.splitlines() if line)
        files: list[tuple[str, str]] = []
        for line in lines:
            path = line[3:].split(" -> ")[-1]
            result = self.container.exec_run(
                ["git", "hash-object", "--", path],
                workdir=str(CONTAINER_WORKDIR),
                user="root",
            )
            digest = result.output.decode("utf-8", errors="replace").strip()
            files.append((path, digest or "deleted"))
        revision = hashlib.sha256((status + "\n" + diff).encode("utf-8")).hexdigest()
        return WorkspaceSnapshot(revision, tuple(sorted(files)))

    def final_diff(self) -> str:
        # Intent-to-add makes untracked files appear in a normal portable git diff.
        self._exec(["git", "add", "-N", "--", "."], check=False)
        return self._exec(["git", "diff", "--binary", "--no-ext-diff", "HEAD"])

    def _exec(self, argv: list[str], check: bool = True) -> str:
        result = self.container.exec_run(
            argv,
            workdir=str(CONTAINER_WORKDIR),
            user="root",
        )
        output = result.output.decode("utf-8", errors="replace")
        if check and result.exit_code != 0:
            raise ExecutionError(
                "CONTAINER_COMMAND_FAILED",
                f"{' '.join(argv)} failed ({result.exit_code}): {output}",
            )
        return output

    def _truncate(self, value: str) -> str:
        if len(value) <= self.output_limit:
            return value
        head = self.output_limit * 2 // 3
        tail = self.output_limit - head
        return f"{value[:head]}\n... output truncated ...\n{value[-tail:]}"


class ContainerWorkspace:
    def __init__(self, executor: PersistentContainerExecutor):
        self.executor = executor
        self.container = executor.container

    def resolve(self, path: str, for_write: bool = False) -> PurePosixPath:
        relative = safe_relative(path)
        candidate = CONTAINER_WORKDIR / relative
        check_target = candidate.parent if for_write else candidate
        result = self.container.exec_run(["realpath", "-m", "--", str(check_target)], user="root")
        resolved = result.output.decode("utf-8", errors="replace").strip()
        if result.exit_code != 0 or not (
            resolved == str(CONTAINER_WORKDIR) or resolved.startswith(str(CONTAINER_WORKDIR) + "/")
        ):
            raise ExecutionError("OUTSIDE_WORKSPACE", f"Path escapes workspace: {path}")
        return candidate

    def read_bytes(self, path: str) -> bytes:
        target = self.resolve(path)
        try:
            chunks, _ = self.container.get_archive(str(target))
            payload = b"".join(chunks)
        except docker.errors.NotFound as exc:
            raise ExecutionError("NOT_FOUND", f"File does not exist: {path}") from exc
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
            member = next((item for item in archive.getmembers() if item.isfile()), None)
            if member is None:
                raise ExecutionError("NOT_FOUND", f"File does not exist: {path}")
            handle = archive.extractfile(member)
            if handle is None:
                raise ExecutionError("NOT_FOUND", f"File does not exist: {path}")
            return handle.read()

    def write_bytes(self, path: str, content: bytes) -> None:
        target = self.resolve(path, for_write=True)
        parent = target.parent
        mkdir = self.container.exec_run(["mkdir", "-p", "--", str(parent)], user="root")
        if mkdir.exit_code != 0:
            raise ExecutionError("WRITE_FAILED", mkdir.output.decode(errors="replace"))
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            info = tarfile.TarInfo(target.name)
            info.size = len(content)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(content))
        buffer.seek(0)
        if not self.container.put_archive(str(parent), buffer.getvalue()):
            raise ExecutionError("WRITE_FAILED", f"Could not write {path}")

    def python_json(self, script: str, args: list[str]) -> Any:
        result = self.executor.run(["python", "-c", script, *args], timeout=120)
        if result.returncode != 0:
            raise ExecutionError("CONTAINER_COMMAND_FAILED", result.output)
        try:
            return json.loads(result.output)
        except json.JSONDecodeError as exc:
            raise ExecutionError("INVALID_TOOL_OUTPUT", result.output) from exc


class _ContainerTool(Tool):
    def __init__(self, workspace: ContainerWorkspace):
        self.workspace = workspace


class ReadFileTool(_ContainerTool):
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
        target = self.workspace.resolve(arguments["path"])
        return Action(frozenset({Capability.READ}), (f"file:{target}",), True)

    def execute(self, arguments: dict[str, Any], env: ExecutionEnv) -> ToolExecution:
        raw = self.workspace.read_bytes(arguments["path"])
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        start = max(0, int(arguments.get("offset", 1)) - 1)
        limit = max(1, int(arguments.get("limit", 2_000)))
        selected = lines[start : start + limit]
        rendered = "\n".join(f"{start + index + 1}\t{line}" for index, line in enumerate(selected))
        if start + len(selected) < len(lines):
            rendered += f"\n... ({len(lines)} lines total)"
        return ToolExecution(True, rendered or "(empty file)")


class ListFilesTool(_ContainerTool):
    name = "list_files"
    description = "List files inside the workspace using a glob pattern."
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "limit": {"type": "integer"},
        },
    }
    _SCRIPT = """
import json, pathlib, sys
root = pathlib.Path('/testbed')
pattern, limit = sys.argv[1], int(sys.argv[2])
items = []
for path in sorted(root.glob(pattern)):
    if path.is_file() and '.git' not in path.parts:
        items.append(path.relative_to(root).as_posix())
        if len(items) >= limit:
            break
print(json.dumps(items))
"""

    def classify(self, arguments: dict[str, Any], env: ExecutionEnv) -> Action:
        safe_pattern(arguments.get("pattern", "**/*"))
        return Action(frozenset({Capability.READ}), ("workspace:/testbed",), True)

    def execute(self, arguments: dict[str, Any], env: ExecutionEnv) -> ToolExecution:
        pattern = safe_pattern(arguments.get("pattern", "**/*"))
        items = self.workspace.python_json(
            self._SCRIPT, [pattern, str(max(1, int(arguments.get("limit", 500))))]
        )
        return ToolExecution(True, "\n".join(items) if items else "(no files)")


class SearchTextTool(_ContainerTool):
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
    _SCRIPT = """
import json, pathlib, sys
root = pathlib.Path('/testbed')
query, pattern, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
hits = []
for path in sorted(root.glob(pattern)):
    if not path.is_file() or '.git' in path.parts:
        continue
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except (UnicodeDecodeError, OSError):
        continue
    relative = path.relative_to(root).as_posix()
    for number, line in enumerate(lines, 1):
        if query in line:
            hits.append(f'{relative}:{number}:{line.strip()}')
            if len(hits) >= limit:
                print(json.dumps(hits))
                raise SystemExit
print(json.dumps(hits))
"""

    def classify(self, arguments: dict[str, Any], env: ExecutionEnv) -> Action:
        safe_pattern(arguments.get("pattern", "**/*"))
        return Action(frozenset({Capability.READ}), ("workspace:/testbed",), True)

    def execute(self, arguments: dict[str, Any], env: ExecutionEnv) -> ToolExecution:
        pattern = safe_pattern(arguments.get("pattern", "**/*"))
        hits = self.workspace.python_json(
            self._SCRIPT,
            [
                str(arguments["query"]),
                pattern,
                str(max(1, int(arguments.get("limit", 200)))),
            ],
        )
        return ToolExecution(True, "\n".join(hits) if hits else "(no matches)")


class WriteFileTool(_ContainerTool):
    name = "write_file"
    description = "Create or replace a UTF-8 text file inside the workspace."
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    }

    def classify(self, arguments: dict[str, Any], env: ExecutionEnv) -> Action:
        target = self.workspace.resolve(arguments["path"], for_write=True)
        return Action(frozenset({Capability.WORKSPACE_WRITE}), (f"file:{target}",), False)

    def execute(self, arguments: dict[str, Any], env: ExecutionEnv) -> ToolExecution:
        self.workspace.write_bytes(arguments["path"], arguments["content"].encode("utf-8"))
        return ToolExecution(True, f"Wrote {arguments['path']}")


class EditFileTool(_ContainerTool):
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
        target = self.workspace.resolve(arguments["path"])
        return Action(frozenset({Capability.WORKSPACE_WRITE}), (f"file:{target}",), False)

    def execute(self, arguments: dict[str, Any], env: ExecutionEnv) -> ToolExecution:
        original = self.workspace.read_bytes(arguments["path"]).decode("utf-8", errors="replace")
        count = original.count(arguments["old"])
        if count != 1:
            return ToolExecution(
                False,
                f"Expected old text exactly once, found {count} occurrences.",
                "AMBIGUOUS_EDIT",
            )
        updated = original.replace(arguments["old"], arguments["new"], 1)
        self.workspace.write_bytes(arguments["path"], updated.encode("utf-8"))
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


class RunCommandTool(_ContainerTool):
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
            ("workspace:/testbed",),
            False,
        )

    def execute(self, arguments: dict[str, Any], env: ExecutionEnv) -> ToolExecution:
        result = self.workspace.executor.run(
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


def build_container_runtime(executor: PersistentContainerExecutor) -> ToolRuntime:
    workspace = ContainerWorkspace(executor)
    return ToolRuntime(
        [
            PlanWriteTool(),
            ReadFileTool(workspace),
            ListFilesTool(workspace),
            SearchTextTool(workspace),
            WriteFileTool(workspace),
            EditFileTool(workspace),
            RunCommandTool(workspace),
        ]
    )


class RequirePatchVerifier:
    """A non-leaking completion gate: require a non-empty patch, not hidden tests."""

    def __init__(self, executor: PersistentContainerExecutor):
        self.executor = executor

    def verify(self, executor: Executor) -> Verification:
        patch = self.executor.final_diff()
        return Verification(bool(patch.strip()), "No workspace patch was produced.")


def disconnect_network(container: Container) -> None:
    container.reload()
    networks = list(container.attrs.get("NetworkSettings", {}).get("Networks", {}))
    client = docker.from_env()
    for network_name in networks:
        client.networks.get(network_name).disconnect(container, force=True)
    container.reload()
    remaining = container.attrs.get("NetworkSettings", {}).get("Networks", {})
    if remaining:
        raise RuntimeError(f"Could not isolate task container network: {sorted(remaining)}")


def replace_git_history_with_baseline(container: Container) -> None:
    """Remove benchmark history so the agent cannot inspect hidden parent commits."""

    top = container.exec_run(
        ["git", "rev-parse", "--show-toplevel"],
        workdir=str(CONTAINER_WORKDIR),
        user="root",
    )
    root = top.output.decode("utf-8", errors="replace").strip()
    if top.exit_code != 0 or root != str(CONTAINER_WORKDIR):
        raise RuntimeError(f"Unexpected task repository root: {root!r}")

    commands = [
        ["rm", "-rf", "--", str(CONTAINER_WORKDIR / ".git")],
        ["git", "init"],
        ["git", "config", "user.email", "trajectory@localhost"],
        ["git", "config", "user.name", "Trajectory Runner"],
        ["git", "add", "-A"],
        ["git", "commit", "-m", "isolated task baseline"],
    ]
    for command in commands:
        result = container.exec_run(
            command,
            workdir=str(CONTAINER_WORKDIR),
            user="root",
        )
        if result.exit_code != 0:
            output = result.output.decode("utf-8", errors="replace")
            raise RuntimeError(f"Failed to prepare isolated git baseline: {command}: {output}")
