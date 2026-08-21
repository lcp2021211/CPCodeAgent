"""One explicit completion gate instead of a lifecycle-hook matrix."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .executor import ExecutionError, Executor
from .types import Verification


class Verifier(Protocol):
    def verify(self, executor: Executor) -> Verification: ...


class CommandVerifier:
    def __init__(self, argv: Sequence[str], timeout: float = 300):
        self.argv = tuple(argv)
        self.timeout = timeout

    def verify(self, executor: Executor) -> Verification:
        try:
            result = executor.run(self.argv, timeout=self.timeout, network=False)
        except ExecutionError as exc:
            return Verification(False, f"Verifier failed to run ({exc.code}): {exc}")
        output = result.output or "(no output)"
        if result.returncode == 0:
            return Verification(True, output)
        return Verification(False, f"{output}\n[exit code: {result.returncode}]")

