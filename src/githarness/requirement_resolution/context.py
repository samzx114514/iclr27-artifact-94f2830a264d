"""Compact, public-only context construction for the Git Agent."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from githarness.versioning.graph import VersionGraph

from .sql_context import compact_sql_feedback


def build_git_agent_context(
    *,
    feedback: str,
    source_turn: int,
    graph: VersionGraph,
    observations: Sequence[Mapping[str, Any]],
    domain: str | None = None,
) -> dict[str, Any]:
    active_id = graph.active_version_id
    active = graph.get(active_id) if active_id is not None else None
    return {
        "schema_version": "git-agent-context-v1",
        "source_turn": source_turn,
        "current_feedback": _prompt_safe(
            compact_sql_feedback(feedback) if domain == "sql" else feedback
        ),
        "active_version": (
            {
                "version_id": active.version_id,
                "parent_version_id": active.parent_version_id,
                "source_turn": active.source_turn,
                "summary": active.summary,
                "resolved_requirement": active.requirement_state.text,
            }
            if active is not None
            else None
        ),
        "history_index": [
            {
                "version_id": version.version_id,
                "parent_version_id": version.parent_version_id,
                "source_turn": version.source_turn,
                "summary": version.summary,
            }
            for version in graph.versions
            if version.version_id != active_id
        ],
        "observations": [dict(item) for item in observations],
        "allowed_actions": ["Inspect", "Commit", "Reuse"],
    }


def inspect_version_context(
    graph: VersionGraph,
    version_id: str,
    *,
    public_output: str | None,
) -> dict[str, Any]:
    version = graph.get(version_id)
    return {
        "version_id": version.version_id,
        "parent_version_id": version.parent_version_id,
        "source_turn": version.source_turn,
        "summary": version.summary,
        "resolved_requirement": version.requirement_state.text,
        "public_output": public_output,
        "ancestor_index": [
            {
                "version_id": item.version_id,
                "parent_version_id": item.parent_version_id,
                "source_turn": item.source_turn,
                "summary": item.summary,
            }
            for item in graph.ancestors(version_id)
        ],
        "children_index": [
            {
                "version_id": item.version_id,
                "source_turn": item.source_turn,
                "summary": item.summary,
            }
            for item in graph.children(version_id)
        ],
    }


def _prompt_safe(value: str) -> str:
    return value.replace("\\boxed{}", "boxed form")
