from __future__ import annotations

import pytest

from githarness.versioning import RequirementState, Version, VersionGraph


def _version(version_id: str, parent: str | None, turn: int) -> Version:
    return Version(version_id, RequirementState(f"requirement {version_id}"), parent, f"cp-{version_id}", turn)


def test_branch_lineage_and_active_version() -> None:
    graph = VersionGraph()
    graph.add(_version("v0", None, 0))
    graph.add(_version("v1", "v0", 1))
    graph.add(_version("v2", "v0", 2))

    assert graph.active_version_id == "v2"
    assert [item.version_id for item in graph.lineage("v2")] == ["v0", "v2"]
    assert {item.version_id for item in graph.children("v0")} == {"v1", "v2"}
    graph.activate_existing("v1")
    assert graph.active_version_id == "v1"


def test_missing_parent_is_rejected() -> None:
    graph = VersionGraph()
    with pytest.raises(KeyError, match="unknown version"):
        graph.add(_version("v1", "missing", 1))

