"""Command-line host for the harness."""

from __future__ import annotations

import argparse
import os
import shlex
from collections.abc import Callable
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from .builtin_tools import build_default_runtime
from .context import ContextEngine
from .executor import DockerExecutor, LocalExecutor
from .kernel import Harness
from .memory import MemoryManager, MemoryScope, MemoryStore
from .model import OpenAICompatibleModel, ResilientModel
from .policy import RunPolicy
from .session import Session, SessionState, SessionStore
from .skills import SkillRegistry
from .types import Decision, RunLimits, RunOutcome
from .ui import TerminalUI
from .verifier import CommandVerifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compact, replayable coding-agent harness")
    parser.add_argument(
        "task",
        nargs="?",
        help="One-off first message; omit it to start an interactive session",
    )
    parser.add_argument(
        "--resume",
        metavar="SESSION_ID",
        help="Resume a durable session and enter its interactive prompt",
    )
    parser.add_argument(
        "--workspace",
        help="Workspace for a new session; resumed sessions restore their workspace",
    )
    parser.add_argument("--model", default=os.getenv("CPCODEAGENT_MODEL", "gpt-4.1"))
    parser.add_argument(
        "--fallback-model", default=os.getenv("CPCODEAGENT_FALLBACK_MODEL") or None
    )
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL") or None)
    parser.add_argument(
        "--executor",
        choices=("local", "docker"),
        help="Executor for a new session; resumed sessions restore it",
    )
    parser.add_argument("--docker-image", help="Docker image for a new Docker session")
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--allow-host", action="append")
    parser.add_argument(
        "--external-writes", choices=("allow", "ask", "deny")
    )
    parser.add_argument("--skill-dir", action="append", default=[])
    parser.add_argument(
        "--verify",
        default=os.getenv("CPCODEAGENT_VERIFY") or None,
        help="Verifier argv, parsed without a shell",
    )
    parser.add_argument(
        "--max-steps", type=int, default=os.getenv("CPCODEAGENT_MAX_STEPS", "40")
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=os.getenv("CPCODEAGENT_MAX_SECONDS", "1800"),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=os.getenv("CPCODEAGENT_MAX_TOKENS", "200000"),
    )
    parser.add_argument(
        "--context-window-tokens",
        type=int,
        default=os.getenv("CPCODEAGENT_CONTEXT_WINDOW_TOKENS", "128000"),
        help="Provider context window used by the 50/70/90%% compression policy",
    )
    parser.add_argument(
        "--journal-dir",
        default=os.getenv(
            "CPCODEAGENT_JOURNAL_DIR", str(Path.home() / ".cpcodeagent" / "runs")
        ),
    )
    parser.add_argument(
        "--memory-dir",
        default=os.getenv("CPCODEAGENT_MEMORY_DIR") or None,
        help="User/session Markdown memory directory",
    )
    return parser


def load_environment(path: str | Path | None = None) -> Path | None:
    """Load one nearest .env without overriding real process environment variables."""

    if path is None:
        discovered = find_dotenv(".env", usecwd=True)
        if not discovered:
            return None
        target = Path(discovered)
    else:
        target = Path(path).expanduser().resolve()
        if not target.is_file():
            return None
    load_dotenv(target, override=False)
    return target.resolve()


def main(argv: list[str] | None = None) -> int:
    load_environment()
    args = build_parser().parse_args(argv)
    ui = TerminalUI()
    if args.resume and args.task:
        raise SystemExit("A task cannot be combined with --resume")
    if not args.api_key:
        raise SystemExit("Set OPENAI_API_KEY or pass --api-key")

    journal_dir = Path(args.journal_dir).expanduser().resolve()
    configured_memory_dir = (
        Path(args.memory_dir).expanduser().resolve()
        if args.memory_dir
        else journal_dir.parent / "memory"
    )
    store = SessionStore(journal_dir)
    if args.resume:
        try:
            session = store.open(args.resume)
            state = session.state
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        workspace = _resume_workspace(args.workspace, state)
        policy = _resume_policy(args, state)
        if state.memory_root is not None:
            stored_memory_dir = Path(state.memory_root).expanduser().resolve()
            if args.memory_dir is not None and configured_memory_dir != stored_memory_dir:
                raise SystemExit("A session restores its original memory directory")
            memory_dir = stored_memory_dir
        else:
            memory_dir = configured_memory_dir
    else:
        session = store.create()
        workspace = Path(
            args.workspace or os.getenv("CPCODEAGENT_WORKSPACE", ".")
        ).expanduser().resolve()
        policy = _new_policy(args)
        memory_dir = configured_memory_dir

    executor = _build_executor(args, workspace, state if args.resume else None)
    if not executor.hard_sandbox:
        ui.warning(
            "LocalExecutor confines built-in file tools but is not an OS command sandbox. "
            "Use --executor docker for hard command isolation."
        )

    roots = [workspace / ".cpcodeagent" / "skills"]
    roots.extend(Path(path) for path in args.skill_dir)
    roots.append(Path.home() / ".cpcodeagent" / "skills")
    skills = SkillRegistry(roots)
    tools = build_default_runtime(skills)
    memory = MemoryManager(MemoryStore(memory_dir, session.session_id))

    primary = OpenAICompatibleModel(args.model, args.api_key, args.base_url)
    fallback = (
        OpenAICompatibleModel(args.fallback_model, args.api_key, args.base_url)
        if args.fallback_model
        else None
    )
    model = ResilientModel(primary, fallback)
    verifier = CommandVerifier(shlex.split(args.verify)) if args.verify else None
    harness = Harness(
        model=model,
        tools=tools,
        context=ContextEngine(max_context_tokens=args.context_window_tokens),
        skills=skills,
        policy=policy,
        approver=ui,
        verifier=verifier,
        limits=RunLimits(args.max_steps, args.max_seconds, args.max_tokens),
        event_sink=ui.handle,
        memory=memory,
    )

    if args.resume:
        if session.state.active_turn is not None:
            ui.info(f"Recovering interrupted turn {session.state.active_turn.turn_id}…")
            try:
                outcome = harness.resume(executor, session.journal)
            except KeyboardInterrupt:
                ui.warning(f"Recovery interrupted. Resume session {session.session_id} later.")
                return 130
            _print_outcome(outcome, ui=ui)
        return interactive_loop(harness, executor, session, ui=ui)

    harness.start_session(executor, session.journal, session.session_id)
    if args.task is None:
        return interactive_loop(harness, executor, session, ui=ui)

    outcome = harness.send(args.task, executor, session.journal, session.session_id)
    _print_outcome(outcome, show_journal=True, ui=ui)
    return 0 if outcome.status.value == "succeeded" else 1


def interactive_loop(
    harness: Harness,
    executor: LocalExecutor | DockerExecutor,
    session: Session,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    ui: TerminalUI | None = None,
) -> int:
    """Run a small terminal REPL over one durable session."""

    state = session.state
    if ui is not None:
        ui.banner(state, harness.model.name)
    else:
        output_fn(f"CPCodeAgent session: {session.session_id}")
        output_fn(f"Workspace: {state.workspace or executor.workspace}")
        output_fn("Commands: /status, /memory, /remember, /forget, /help, /exit")

    while True:
        try:
            raw = ui.prompt() if ui is not None else input_fn("you> ")
        except (EOFError, KeyboardInterrupt):
            _message(f"Session saved: {session.session_id}", output_fn, ui)
            return 0

        message = raw.strip()
        if not message:
            continue
        if message in {"/exit", "/quit"}:
            _message(f"Session saved: {session.session_id}", output_fn, ui)
            return 0
        if message == "/help":
            _message(
                "/status shows session state; /memory shows persistent memory; "
                "/remember <user|session> <text> saves a note; "
                "/forget <user|session> <key|all> removes notes; /exit saves and leaves.",
                output_fn,
                ui,
            )
            continue
        if message == "/status":
            current = session.state
            status = (
                current.last_turn.status.value
                if current.last_turn and current.last_turn.status
                else "idle"
            )
            _message(
                f"session={current.session_id} turns={len(current.turns)} "
                f"last_status={status}",
                output_fn,
                ui,
            )
            continue
        if message == "/memory" or message.startswith("/memory "):
            _handle_memory_command(message, harness, session, output_fn, ui)
            continue
        if message.startswith(("/remember ", "/forget ")):
            _handle_memory_command(message, harness, session, output_fn, ui)
            continue
        if message.startswith("/"):
            _message(f"Unknown command: {message}. Use /help.", output_fn, ui)
            continue

        try:
            outcome = harness.send(message, executor, session.journal, session.session_id)
        except KeyboardInterrupt:
            _message(
                f"Turn interrupted and journaled. Resume session {session.session_id} later.",
                output_fn,
                ui,
            )
            return 130
        except ValueError as exc:
            if ui is not None:
                ui.error(f"Session error: {exc}")
            else:
                output_fn(f"Session error: {exc}")
            continue
        _print_outcome(outcome, output_fn=output_fn, ui=ui)


def _handle_memory_command(
    message: str,
    harness: Harness,
    session: Session,
    output_fn: Callable[[str], None],
    ui: TerminalUI | None,
) -> None:
    manager = harness.memory
    if manager is None:
        _message("Persistent memory is not configured.", output_fn, ui)
        return
    parts = message.split(maxsplit=2)
    command = parts[0]
    try:
        if command == "/memory":
            scope = parts[1] if len(parts) > 1 else "all"
            if scope not in {"all", MemoryScope.USER.value, MemoryScope.SESSION.value}:
                raise ValueError("Usage: /memory [user|session]")
            documents: list[str] = []
            if scope in {"all", MemoryScope.USER.value}:
                documents.append(manager.store.read(MemoryScope.USER) or "# User Memory\n\n(empty)")
            if scope in {"all", MemoryScope.SESSION.value}:
                documents.append(
                    manager.store.read(MemoryScope.SESSION) or "# Session Memory\n\n(empty)"
                )
            _message("\n\n".join(documents), output_fn, ui)
            return

        if len(parts) < 3:
            raise ValueError(f"Usage: {command} <user|session> <text|key|all>")
        scope = MemoryScope(parts[1])
        value = parts[2].strip()
        if command == "/remember":
            key = manager.remember(scope, value, session.journal)
            manager.snapshot(session.journal)
            _message(f"Remembered {scope.value} memory as {key}.", output_fn, ui)
            return
        if command == "/forget":
            changed = manager.forget(
                scope,
                None if value == "all" else value,
                session.journal,
            )
            manager.snapshot(session.journal)
            suffix = "removed" if changed else "not found"
            _message(f"Memory {suffix}.", output_fn, ui)
            return
        raise ValueError("Unknown memory command")
    except (OSError, ValueError) as exc:
        _message(f"Memory error: {exc}", output_fn, ui)


def _new_policy(args: argparse.Namespace) -> RunPolicy:
    return RunPolicy(
        workspace_write=not args.read_only,
        allowed_hosts=tuple(args.allow_host or ()),
        external_writes=Decision(args.external_writes or Decision.ASK.value),
    )


def _build_executor(
    args: argparse.Namespace,
    workspace: Path,
    state: SessionState | None,
) -> LocalExecutor | DockerExecutor:
    stored = state.executor if state is not None else None
    if stored is not None:
        kind = str(stored.get("kind", "local"))
        image = str(stored.get("image", "python:3.12-slim"))
        if args.executor is not None and args.executor != kind:
            raise SystemExit("A session restores its original executor")
        if args.docker_image is not None and args.docker_image != image:
            raise SystemExit("A session restores its original Docker image")
    else:
        kind = args.executor or os.getenv("CPCODEAGENT_EXECUTOR", "local")
        image = args.docker_image or os.getenv(
            "CPCODEAGENT_DOCKER_IMAGE", "python:3.12-slim"
        )
        if kind not in {"local", "docker"}:
            raise SystemExit("CPCODEAGENT_EXECUTOR must be 'local' or 'docker'")
    if kind == "docker":
        return DockerExecutor(workspace, image=image)
    return LocalExecutor(workspace)


def _resume_policy(args: argparse.Namespace, state: SessionState) -> RunPolicy:
    if state.policy is None:
        return _new_policy(args)
    if args.read_only or args.allow_host is not None or args.external_writes is not None:
        raise SystemExit(
            "A session restores its original policy; start a new session to change permissions"
        )
    return RunPolicy.from_dict(state.policy)


def _resume_workspace(value: str | None, state: SessionState) -> Path:
    stored = Path(state.workspace).expanduser().resolve() if state.workspace else None
    requested = Path(value).expanduser().resolve() if value else stored
    requested = requested or Path.cwd()
    if stored is not None and requested != stored:
        raise SystemExit(
            f"Session {state.session_id} belongs to {stored}; omit --workspace or use that path"
        )
    return requested


def _print_outcome(
    outcome: RunOutcome,
    show_journal: bool = False,
    output_fn: Callable[[str], None] = print,
    ui: TerminalUI | None = None,
) -> None:
    if ui is not None:
        ui.outcome(outcome, show_journal)
        return
    output_fn(
        f"\nagent> {outcome.answer}\n"
        f"[{outcome.status.value}] session={outcome.session_id} "
        f"turn={outcome.turn_id or '?'} steps={outcome.steps}"
    )
    if show_journal and outcome.journal_path:
        output_fn(f"Journal: {outcome.journal_path}")


def _message(
    value: str,
    output_fn: Callable[[str], None],
    ui: TerminalUI | None,
) -> None:
    if ui is not None:
        ui.info(value)
    else:
        output_fn(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
