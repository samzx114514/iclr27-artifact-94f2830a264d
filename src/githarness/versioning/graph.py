"""Branchable immutable version history."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .models import Version


@dataclass
class VersionGraph:
    _versions: dict[str, Version] = field(default_factory=dict)
    _active_version_id: str | None = None
    _next_index: int = 0

    @property
    def active_version_id(self) -> str | None:
        return self._active_version_id

    @property
    def versions(self) -> tuple[Version, ...]:
        return tuple(self._versions.values())

    def propose_version_id(self) -> str:
        """Return the next ID without mutating accepted graph state."""

        index = self._next_index
        while f"v{index}" in self._versions:
            index += 1
        return f"v{index}"

    def reserve_version_id(self) -> str:
        """Compatibility alias; proposals are intentionally non-mutating."""

        return self.propose_version_id()

    def get(self, version_id: str) -> Version:
        try:
            return self._versions[version_id]
        except KeyError as error:
            raise KeyError(f"unknown version: {version_id}") from error

    def add(self, version: Version, *, activate: bool = True) -> None:
        self.validate_add(version)
        self._versions[version.version_id] = version
        if version.version_id.startswith("v") and version.version_id[1:].isdigit():
            self._next_index = max(self._next_index, int(version.version_id[1:]) + 1)
        self._validate_acyclic()
        if activate:
            self._active_version_id = version.version_id

    def validate_add(self, version: Version) -> None:
        """Validate a prospective mutation before checkpoint persistence."""

        if version.version_id in self._versions:
            raise ValueError(f"version already exists: {version.version_id}")
        if version.parent_version_id is not None:
            self.get(version.parent_version_id)
            if version.parent_version_id == version.version_id:
                raise ValueError("version cannot be its own parent")

    def activate_existing(self, version_id: str) -> Version:
        version = self.get(version_id)
        self._active_version_id = version_id
        return version

    def ancestors(self, version_id: str) -> tuple[Version, ...]:
        ancestors: list[Version] = []
        current = self.get(version_id)
        while current.parent_version_id is not None:
            current = self.get(current.parent_version_id)
            ancestors.append(current)
        return tuple(ancestors)

    def lineage(self, version_id: str) -> tuple[Version, ...]:
        return (*reversed(self.ancestors(version_id)), self.get(version_id))

    def children(self, version_id: str) -> tuple[Version, ...]:
        self.get(version_id)
        return tuple(v for v in self._versions.values() if v.parent_version_id == version_id)

    def view(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "version_id": version.version_id,
                "parent_version_id": version.parent_version_id,
                "requirement": version.requirement_state.text,
                "source_turn": version.source_turn,
                "summary": version.summary,
                "active": version.version_id == self._active_version_id,
            }
            for version in self._versions.values()
        )

    def _validate_acyclic(self) -> None:
        for start in self._versions:
            seen: set[str] = set()
            current: str | None = start
            while current is not None:
                if current in seen:
                    raise ValueError("version graph contains a cycle")
                seen.add(current)
                current = self._versions[current].parent_version_id

    @classmethod
    def from_versions(cls, versions: Iterable[Version], active: str | None = None) -> "VersionGraph":
        graph = cls()
        pending = list(versions)
        while pending:
            progressed = False
            for version in list(pending):
                if version.parent_version_id is None or version.parent_version_id in graph._versions:
                    graph.add(version, activate=False)
                    pending.remove(version)
                    progressed = True
            if not progressed:
                raise ValueError("versions contain a missing parent or cycle")
        if active is not None:
            graph.activate_existing(active)
        return graph
