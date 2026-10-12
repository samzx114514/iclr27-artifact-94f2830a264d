"""Workspace boundary used by the SWE-specific graph."""

from __future__ import annotations

from typing import Any, Protocol


class SWEWorkspace(Protocol):
    """An isolated code workspace controlled by the benchmark runtime."""

    def execute(self, command: str, timeout: int | None = None) -> dict[str, Any]:
        """Execute a shell command inside the isolated workspace."""

    def git_diff(self) -> str:
        """Return the current repository diff."""
