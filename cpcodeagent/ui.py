"""Rich terminal presentation for streaming harness events."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from rich.text import Text

from .session import SessionState
from .types import Action, RunEvent, RunEventKind, RunOutcome, RunStatus, ToolCall


class TerminalUI:
    """One stateful adapter from neutral RunEvents to a readable terminal UI."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self._status: Status | None = None
        self._stream_started = False
        self._stream_text = ""
        self._tool_names: dict[str, str] = {}

    def banner(self, state: SessionState, model: str) -> None:
        executor = (state.executor or {}).get("kind", "local")
        body = (
            f"[bold cyan]CPCodeAgent[/bold cyan]  [dim]session {state.session_id}[/dim]\n"
            f"Model: [magenta]{model}[/magenta]   Executor: [cyan]{executor}[/cyan]\n"
            f"Workspace: [dim]{state.workspace or '.'}[/dim]\n"
            "[dim]/status  /memory  /remember  /forget  /help  /exit[/dim]"
        )
        self.console.print(Panel(body, border_style="blue", expand=False))

    def prompt(self) -> str:
        return self.console.input("[bold cyan]you ›[/bold cyan] ")

    def handle(self, event: RunEvent) -> None:
        if event.kind is RunEventKind.MODEL_START:
            self._stop_status()
            self._stream_started = False
            self._stream_text = ""
            self._status = self.console.status("[magenta]Thinking…[/magenta]", spinner="dots")
            self._status.start()
        elif event.kind is RunEventKind.TEXT_DELTA:
            self._text_delta(str(event.data.get("text", "")))
        elif event.kind is RunEventKind.MODEL_END:
            self._stop_status()
            if self._stream_started and not self._stream_text.endswith("\n"):
                self.console.print()
        elif event.kind is RunEventKind.TOOLS_START:
            self._tools_start(event.data.get("calls", ()))
        elif event.kind is RunEventKind.TOOLS_END:
            self._tools_end(event.data.get("results", ()))
        elif event.kind is RunEventKind.VERIFY_START:
            self._stop_status()
            self._status = self.console.status("[yellow]Verifying…[/yellow]", spinner="dots")
            self._status.start()
        elif event.kind is RunEventKind.VERIFY_END:
            self._stop_status()
            passed = bool(event.data.get("passed"))
            self.console.print(
                "[green]✓ verification passed[/green]"
                if passed
                else "[red]✗ verification failed; continuing[/red]"
            )

    def outcome(self, outcome: RunOutcome, show_journal: bool = False) -> None:
        self._stop_status()
        streamed = self._stream_text == outcome.answer and bool(self._stream_text)
        if not streamed:
            style = "red" if outcome.status is not RunStatus.SUCCEEDED else "green"
            self.console.print(Panel(outcome.answer, title="agent", border_style=style))
        status_style = "green" if outcome.status is RunStatus.SUCCEEDED else "red"
        self.console.print(
            f"[{status_style}]{outcome.status.value}[/{status_style}] "
            f"[dim]turn={outcome.turn_id or '?'}  steps={outcome.steps}  "
            f"tokens={outcome.tokens}[/dim]"
        )
        if show_journal and outcome.journal_path:
            self.console.print(f"[dim]Journal: {outcome.journal_path}[/dim]")

    def info(self, message: str) -> None:
        self._stop_status()
        self.console.print(f"[cyan]•[/cyan] {message}")

    def warning(self, message: str) -> None:
        self._stop_status()
        self.console.print(f"[yellow]warning:[/yellow] {message}")

    def error(self, message: str) -> None:
        self._stop_status()
        self.console.print(f"[red]error:[/red] {message}")

    def approve(self, call: ToolCall, action: Action, reason: str) -> bool:
        self._stop_status()
        targets = "\n".join(f"  • {target}" for target in action.targets) or "  (none)"
        self.console.print(
            Panel(
                f"[bold]{call.name}[/bold]\n{reason}\n[dim]{targets}[/dim]",
                title="Approval required",
                border_style="yellow",
            )
        )
        return self.console.input("[yellow]Allow once? [y/N][/yellow] ").strip().lower() in {
            "y",
            "yes",
        }

    def _text_delta(self, text: str) -> None:
        if not text:
            return
        self._stop_status()
        if not self._stream_started:
            self.console.print("[bold green]agent ›[/bold green] ", end="")
            self._stream_started = True
        self._stream_text += text
        self.console.print(text, end="", markup=False, highlight=False, soft_wrap=True)

    def _tools_start(self, calls: Any) -> None:
        self._stop_status()
        count = 0
        for call in calls:
            if not isinstance(call, ToolCall):
                continue
            count += 1
            self._tool_names[call.id] = call.name
            if call.name == "plan_write":
                detail = "updating execution plan"
            elif call.name == "write_shared_contract":
                detail = f"contract {call.arguments.get('name', '?')}"
            elif call.name == "delegate_task":
                mode = str(call.arguments.get("mode", "inspect"))
                task = _one_line(str(call.arguments.get("task", "")), 90)
                detail = f"[sub:{mode}] {task}"
            elif call.name in {"read_subagent_patch", "apply_subagent_patch"}:
                detail = f"artifact {call.arguments.get('artifact_id', '?')}"
            else:
                detail = _brief(call.arguments)
            self.console.print(
                f"[cyan]▸ {call.name}[/cyan] [dim]{detail}[/dim]"
            )
        if count:
            label = "tool" if count == 1 else "tools"
            self._status = self.console.status(
                f"[cyan]Running {count} {label}…[/cyan]", spinner="dots"
            )
            self._status.start()

    def _tools_end(self, results: Any) -> None:
        self._stop_status()
        for result in results:
            name = self._tool_names.get(result.call_id, result.call_id)
            if result.ok:
                if name == "plan_write":
                    self.console.print(
                        Panel(
                            Text(result.output),
                            title="✓ execution plan",
                            border_style="cyan",
                            expand=False,
                        )
                    )
                    continue
                if name == "write_shared_contract":
                    try:
                        payload = json.loads(result.output)
                    except (TypeError, ValueError):
                        payload = {}
                    contract_id = str(payload.get("contract_id", "created"))
                    self.console.print(
                        f"[green]✓ shared contract[/green] [dim]{contract_id}[/dim]"
                    )
                    continue
                if name == "delegate_task":
                    try:
                        payload = json.loads(result.output)
                    except (TypeError, ValueError):
                        payload = {}
                    status = str(payload.get("status", "completed"))
                    summary = str(payload.get("summary", result.output))
                    self.console.print(
                        f"[green]✓ subagent {status}[/green] [dim]{_one_line(summary)}[/dim]"
                    )
                    continue
                if name == "read_subagent_patch":
                    self.console.print("[green]✓ patch preview ready[/green]")
                    continue
                if name == "apply_subagent_patch":
                    self.console.print(
                        f"[green]✓ patch applied[/green] [dim]{_one_line(result.output)}[/dim]"
                    )
                    continue
                detail = _one_line(result.output)
                suffix = f" [dim]{detail}[/dim]" if detail else ""
                self.console.print(f"[green]✓ {name}[/green]{suffix}")
            else:
                detail = _one_line(result.output or result.error or "failed")
                self.console.print(f"[red]✗ {name}[/red] [dim]{detail}[/dim]")

    def _stop_status(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None


def _brief(arguments: dict[str, Any], limit: int = 110) -> str:
    value = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _one_line(value: str, limit: int = 100) -> str:
    line = " ".join(value.strip().splitlines())
    return line if len(line) <= limit else line[: limit - 1] + "…"
