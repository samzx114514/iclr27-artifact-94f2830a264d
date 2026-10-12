from __future__ import annotations

import json
from pathlib import Path

import pytest

from githarness.requirement_resolution import Commit, GitAgent, Inspect, Reuse, VersionHandoff
from githarness.requirement_resolution.agent import ScriptedGitPolicy
from githarness.requirement_resolution.context import build_git_agent_context
from githarness.requirement_resolution.handoff import (
    handoff_from_dict,
    handoff_to_dict,
)
from githarness.requirement_resolution.qwen_policy import parse_text_action
from githarness.requirement_resolution.sql_context import SQL_REQUEST_PREAMBLE
from githarness.errors import ProtocolError
from githarness.versioning import Checkpoint, WorkState
from githarness.versioning import RequirementState, Version, VersionGraph


def _graph() -> VersionGraph:
    graph = VersionGraph()
    graph.add(Version("v0", RequirementState("Compute 2 + 3"), None, "checkpoint-v0", 0))
    return graph


def test_git_agent_default_budget_is_paper_aligned() -> None:
    assert GitAgent(ScriptedGitPolicy([])).max_steps == 4
    with pytest.raises(ValueError, match="positive integer"):
        GitAgent(ScriptedGitPolicy([]), max_steps=0)
    with pytest.raises(ValueError, match="positive integer"):
        GitAgent(ScriptedGitPolicy([]), max_steps=True)


def test_inspect_then_commit_reserves_target_without_mutating_graph() -> None:
    graph = _graph()
    agent = GitAgent(ScriptedGitPolicy([Inspect("v0"), Commit("Compute 2 + 5", "v0", "replace 3")]))
    handoff = agent.resolve(feedback="Use 5", source_turn=1, graph=graph, version_outputs={"v0": "5"})

    assert handoff.selected_base_version_id == "v0"
    assert handoff.target_version_id == "v1"
    assert len(graph.versions) == 1
    assert [event["action"] for event in handoff.events] == ["Inspect", "Commit"]


def test_reuse_targets_existing_version() -> None:
    graph = _graph()
    handoff = GitAgent(ScriptedGitPolicy([Reuse("v0")])).resolve(
        feedback="Return to the original", source_turn=1, graph=graph
    )
    assert handoff.reused_version_id == "v0"
    assert handoff.resolved_requirement == "Compute 2 + 3"


def test_compact_context_contains_active_requirement_once_and_indexes_history() -> None:
    graph = _graph()
    graph.add(
        Version(
            "v1",
            RequirementState("Compute 2 + 5 and return JSON"),
            "v0",
            "checkpoint-v1",
            1,
            summary="replace operand and request JSON",
        )
    )

    context = build_git_agent_context(
        feedback="Restore the original arithmetic task",
        source_turn=2,
        graph=graph,
        observations=(),
    )

    assert context["schema_version"] == "git-agent-context-v1"
    assert context["active_version"]["resolved_requirement"] == "Compute 2 + 5 and return JSON"
    assert context["history_index"] == [
        {
            "version_id": "v0",
            "parent_version_id": None,
            "source_turn": 0,
            "summary": "",
        }
    ]
    serialized = json.dumps(context)
    assert serialized.count("Compute 2 + 5 and return JSON") == 1
    assert "Compute 2 + 3" not in serialized


def test_only_one_inspect_is_allowed_then_commit_can_finish() -> None:
    graph = _graph()
    agent = GitAgent(
        ScriptedGitPolicy(
            [
                Inspect("v0"),
                Inspect("v0"),
                Commit("Compute 2 + 4", "v0", "replace operand"),
            ]
        )
    )

    handoff = agent.resolve(feedback="Use 4", source_turn=1, graph=graph)

    assert handoff.resolved_requirement == "Compute 2 + 4"
    assert [event["status"] for event in handoff.events] == ["ok", "error", "ok"]
    assert handoff.events[1]["code"] == "inspect_limit"


def test_invalid_policy_action_retries_without_mutating_graph() -> None:
    class InvalidThenCommit:
        def __init__(self) -> None:
            self.calls = 0

        def choose(self, context):
            del context
            self.calls += 1
            if self.calls == 1:
                raise ProtocolError("malformed tool arguments")
            return Commit("Compute 2 + 8", "v0", "replace operand")

    graph = _graph()
    handoff = GitAgent(InvalidThenCommit()).resolve(
        feedback="Use 8", source_turn=1, graph=graph
    )

    assert len(graph.versions) == 1
    assert handoff.events[0]["code"] == "invalid_action"
    assert handoff.target_version_id == "v1"


def test_failed_commit_proposal_does_not_advance_graph_identity() -> None:
    graph = _graph()
    first = GitAgent(ScriptedGitPolicy([Commit("Compute 2 + 8", "v0", "replace")])).resolve(
        feedback="Use 8", source_turn=1, graph=graph
    )
    second = GitAgent(ScriptedGitPolicy([Commit("Compute 2 + 9", "v0", "replace")])).resolve(
        feedback="Use 9", source_turn=1, graph=graph
    )

    assert first.target_version_id == second.target_version_id == "v1"
    assert len(graph.versions) == 1


def test_handoff_round_trip_binds_selected_checkpoint_and_detects_tampering() -> None:
    graph = _graph()
    checkpoint = Checkpoint(
        "checkpoint-v0",
        "v0",
        WorkState({"answer.md": "5"}),
        state0_id="state0-v0",
        origin_state0_id="state0-v0",
    )
    handoff = GitAgent(ScriptedGitPolicy([Commit("Compute 2 + 7", "v0", "replace 3")])).resolve(
        feedback="Use 7",
        source_turn=1,
        graph=graph,
        checkpoints={checkpoint.checkpoint_id: checkpoint},
        checkpoint_bindings={"v0": checkpoint.checkpoint_id},
        task_id="math-1",
        domain="math",
    )

    payload = handoff_to_dict(handoff)
    assert payload["selected_base_checkpoint"]["checkpoint_digest"] == checkpoint.digest
    assert handoff_from_dict(payload) == handoff

    payload["resolved_requirement"] = "tampered"
    with pytest.raises(ValueError, match="digest"):
        handoff_from_dict(payload)


def test_reuse_rejects_missing_checkpoint_binding_then_can_recover() -> None:
    graph = _graph()
    handoff = GitAgent(
        ScriptedGitPolicy([Reuse("v0"), Commit("Independent task", None, "new task")])
    ).resolve(
        feedback="start independently",
        source_turn=1,
        graph=graph,
        checkpoints={},
        checkpoint_bindings={},
    )

    assert handoff.resolved_requirement == "Independent task"
    assert handoff.events[0]["code"] == "invalid_base"


def test_strict_actions_reject_runtime_owned_fields_and_qwen_parser_shares_protocol() -> None:
    with pytest.raises(ProtocolError, match="invalid Git Agent tool call"):
        parse_text_action(
            json.dumps(
                {
                    "tool": "Commit",
                    "arguments": {
                        "resolved_requirement": "Task",
                        "base_version_id": None,
                        "update_hint": "new",
                        "target_version_id": "forged",
                    },
                }
            )
        )

    action = parse_text_action(
        json.dumps(
            {
                "tool": "Commit",
                "arguments": {
                    "resolved_requirement": "Complete task",
                    "base_version_id": None,
                    "update_hint": "initial",
                },
            }
        )
    )
    assert action == Commit("Complete task", None, "initial")


def test_sql_context_removes_fixed_schema_from_git_agent_prompt() -> None:
    context = build_git_agent_context(
        feedback=(
            SQL_REQUEST_PREAMBLE
            + "\n\nDatabase schema:\nCREATE TABLE private_table(value INTEGER);"
            + "\n\nEvidence:\n'value' means amount\n\nHow many rows?"
        ),
        source_turn=0,
        graph=VersionGraph(),
        observations=(),
        domain="sql",
    )

    assert context["current_feedback"] == "Evidence:\n'value' means amount\n\nHow many rows?"
    assert "CREATE TABLE" not in json.dumps(context)


def test_checked_in_handoff_schema_matches_serialized_contract() -> None:
    schema = json.loads(
        (Path(__file__).parents[1] / "schemas" / "resolved-requirement-handoff-v1.schema.json")
        .read_text(encoding="utf-8")
    )
    handoff = VersionHandoff("Complete task", None, "v0", "initial")

    assert set(schema["required"]) == set(handoff_to_dict(handoff))
