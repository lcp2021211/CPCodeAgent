"""Workspace-confined file operations and replaceable command executors."""

from __future__ import annotations

import hashlib
import os
import shutil
import signal
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


class ExecutionError(RuntimeError):
    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    output: str


@dataclass(frozen=True)
class WorkspaceSnapshot:
    revision: str
    files: tuple[tuple[str, str], ...]

    def changed_since(self, previous: WorkspaceSnapshot) -> tuple[str, ...]:
        before = dict(previous.files)
        after = dict(self.files)
        return tuple(
            sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
        )


class Executor:
    """Base execution world. File operations are confined to one workspace."""

    hard_sandbox = False

    def __init__(self, workspace: str | Path, output_limit: int = 20_000):
        self.workspace = Path(workspace).expanduser().resolve()
        if not self.workspace.is_dir():
            raise ValueError(f"Workspace does not exist or is not a directory: {self.workspace}")
        self.output_limit = output_limit

    def resolve(self, path: str | Path) -> Path:
        raw = Path(path).expanduser()
        candidate = raw.resolve() if raw.is_absolute() else (self.workspace / raw).resolve()
        try:
            candidate.relative_to(self.workspace)
        except ValueError as exc:
            raise ExecutionError("OUTSIDE_WORKSPACE", f"Path escapes workspace: {path}") from exc
        return candidate

    def read_text(self, path: str, offset: int = 1, limit: int = 2_000) -> str:
        target = self.resolve(path)
        if not target.is_file():
            raise ExecutionError("NOT_FOUND", f"File does not exist: {path}")
        text = target.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        start = max(0, offset - 1)
        selected = lines[start : start + max(1, limit)]
        rendered = "\n".join(f"{start + index + 1}\t{line}" for index, line in enumerate(selected))
        if start + len(selected) < len(lines):
            rendered += f"\n... ({len(lines)} lines total)"
        return rendered or "(empty file)"

    def write_text(self, path: str, content: str) -> None:
        target = self.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def list_files(self, pattern: str = "**/*", limit: int = 500) -> tuple[str, ...]:
        matches: list[str] = []
        for path in sorted(self.workspace.glob(pattern)):
            if path.is_file() and ".git" not in path.parts:
                matches.append(path.relative_to(self.workspace).as_posix())
                if len(matches) >= limit:
                    break
        return tuple(matches)

    def search_text(self, query: str, pattern: str = "**/*", limit: int = 200) -> tuple[str, ...]:
        hits: list[str] = []
        for relative in self.list_files(pattern, limit=10_000):
            path = self.resolve(relative)
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for line_number, line in enumerate(lines, start=1):
                if query in line:
                    hits.append(f"{relative}:{line_number}:{line.strip()}")
                    if len(hits) >= limit:
                        return tuple(hits)
        return tuple(hits)

    def run(
        self,
        argv: Sequence[str],
        timeout: float = 120,
        network: bool = False,
    ) -> CommandResult:
        if network:
            raise ExecutionError(
                "NETWORK_UNENFORCED",
                "The local executor cannot enforce a network allowlist; use DockerExecutor",
            )
        return self._run_process(argv, cwd=self.workspace, timeout=timeout)

    def snapshot(self) -> WorkspaceSnapshot:
        files: list[tuple[str, str]] = []
        for path in sorted(self.workspace.rglob("*")):
            if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(self.workspace).as_posix()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            files.append((relative, digest))
        encoded = "\n".join(f"{path}:{digest}" for path, digest in files).encode()
        return WorkspaceSnapshot(hashlib.sha256(encoded).hexdigest(), tuple(files))

    def _run_process(
        self,
        argv: Sequence[str],
        cwd: Path,
        timeout: float,
    ) -> CommandResult:
        if not argv or not all(isinstance(part, str) and part for part in argv):
            raise ExecutionError("INVALID_COMMAND", "argv must contain non-empty strings")
        try:
            process = subprocess.Popen(
                list(argv),
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise ExecutionError("COMMAND_NOT_FOUND", str(exc)) from exc
        except OSError as exc:
            raise ExecutionError("SPAWN_FAILED", str(exc), retryable=True) from exc
        try:
            output, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
            raise ExecutionError(
                "TIMEOUT",
                f"Command timed out after {timeout}s\n{self._truncate(output)}",
                retryable=False,
            ) from exc
        return CommandResult(process.returncode, self._truncate(output or ""))

    def _truncate(self, value: str) -> str:
        if len(value) <= self.output_limit:
            return value
        head = self.output_limit * 2 // 3
        tail = self.output_limit - head
        return f"{value[:head]}\n... output truncated ...\n{value[-tail:]}"


class LocalExecutor(Executor):
    """Trusted local process execution plus strictly confined Python file tools."""


class DockerExecutor(Executor):
    """Runs commands in an ephemeral, capability-reduced Docker container."""

    hard_sandbox = True

    def __init__(
        self,
        workspace: str | Path,
        image: str = "python:3.12-slim",
        output_limit: int = 20_000,
    ):
        super().__init__(workspace, output_limit)
        if shutil.which("docker") is None:
            raise ValueError("Docker executable is not available")
        self.image = image

    def run(
        self,
        argv: Sequence[str],
        timeout: float = 120,
        network: bool = False,
    ) -> CommandResult:
        docker_argv = [
            "docker",
            "run",
            "--rm",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=256",
            "--memory=1g",
            "--cpus=2",
            "--network=bridge" if network else "--network=none",
            "-v",
            f"{self.workspace}:/workspace:rw",
            "-w",
            "/workspace",
            self.image,
            *argv,
        ]
        return self._run_process(docker_argv, cwd=self.workspace, timeout=timeout)


@dataclass(frozen=True)
class ExecutionEnv:
    executor: Executor

    @property
    def workspace(self) -> Path:
        return self.executor.workspace
