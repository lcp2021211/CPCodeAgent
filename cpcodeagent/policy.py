"""One explicit policy decision between model intent and execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .types import Action, Capability, Decision, ToolCall


class Approver(Protocol):
    def approve(self, call: ToolCall, action: Action, reason: str) -> bool: ...


class RejectingApprover:
    def approve(self, call: ToolCall, action: Action, reason: str) -> bool:
        return False


class ConsoleApprover:
    def approve(self, call: ToolCall, action: Action, reason: str) -> bool:
        print(f"\nApproval required for {call.name}: {reason}")
        print(f"Targets: {', '.join(action.targets) or '(none)'}")
        answer = input("Allow once? [y/N] ").strip().lower()
        return answer in {"y", "yes"}


@dataclass(frozen=True)
class RunPolicy:
    """The complete authority granted to one run."""

    workspace_write: bool = True
    allowed_hosts: tuple[str, ...] = ()
    external_writes: Decision = Decision.ASK

    def decide(self, action: Action, workspace: Path) -> tuple[Decision, str]:
        workspace = workspace.resolve()

        for target in action.targets:
            if target.startswith("file:"):
                candidate = Path(target.removeprefix("file:")).resolve()
                if not _inside(candidate, workspace):
                    return Decision.DENY, f"path is outside workspace: {candidate}"

        if Capability.WORKSPACE_WRITE in action.capabilities and not self.workspace_write:
            return Decision.DENY, "this run is read-only"

        if Capability.NETWORK in action.capabilities:
            hosts = {
                target.removeprefix("host:")
                for target in action.targets
                if target.startswith("host:")
            }
            if not hosts or not all(_host_allowed(host, self.allowed_hosts) for host in hosts):
                return Decision.DENY, "network target is not allowlisted"

        if Capability.EXTERNAL_WRITE in action.capabilities:
            if self.external_writes is Decision.DENY:
                return Decision.DENY, "external side effects are disabled"
            if self.external_writes is Decision.ASK:
                return Decision.ASK, "the call has an external side effect"

        return Decision.ALLOW, "inside the run policy"

    def describe(self) -> str:
        write = "allowed inside the workspace" if self.workspace_write else "disabled"
        network = ", ".join(self.allowed_hosts) if self.allowed_hosts else "disabled"
        return (
            f"Workspace writes: {write}. Network allowlist: {network}. "
            f"External writes: {self.external_writes.value}."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_write": self.workspace_write,
            "allowed_hosts": list(self.allowed_hosts),
            "external_writes": self.external_writes.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunPolicy:
        return cls(
            workspace_write=bool(data.get("workspace_write", True)),
            allowed_hosts=tuple(str(host) for host in data.get("allowed_hosts", ())),
            external_writes=Decision(data.get("external_writes", Decision.ASK.value)),
        )


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _host_allowed(host: str, allowed: tuple[str, ...]) -> bool:
    return "*" in allowed or host in allowed or any(
        entry.startswith("*.") and host.endswith(entry[1:]) for entry in allowed
    )
