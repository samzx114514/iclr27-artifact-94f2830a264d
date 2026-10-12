"""Narrow retrieval interface injected by the benchmark runtime."""

from __future__ import annotations

from typing import Any, Protocol


class SearchBackend(Protocol):
    """A read-only search backend with no shell or index control surface."""

    def search(self, query: str) -> list[dict[str, Any]]: ...
