from __future__ import annotations

import pytest

from githarness.branch_execution import ConservativeExecutionRouter
from githarness.cli import _execute_case
from githarness.execution_schedule import RequirementExecutionPlan
from githarness.offline import DeterministicMathRootExecutor
from githarness.requirement_resolution import VersionHandoff
from githarness.version_interface import ExecutionResult


def test_necessary_path_keeps_bootstrap_and_target_lineage_only() -> None:
    plan = RequirementExecutionPlan()
    plan.record(0, VersionHandoff("root", None, "v0", ""))
    plan.record(1, VersionHandoff("branch one", "v0", "v1", "one"))
    plan.record(2, VersionHandoff("branch two", "v0", "v2", "two"))

    assert plan.necessary_versions([2]) == ("v0", "v2")


def test_independent_target_still_keeps_state0_bootstrap() -> None:
    plan = RequirementExecutionPlan()
    plan.record(0, VersionHandoff("root", None, "v0", ""))
    plan.record(1, VersionHandoff("independent", None, "v1", "fresh"))

    assert plan.necessary_versions([1]) == ("v0", "v1")


def test_deferred_target_turn_must_exist() -> None:
    plan = RequirementExecutionPlan()
    plan.record(0, VersionHandoff("root", None, "v0", ""))

    with pytest.raises(ValueError, match="out of range"):
        plan.necessary_versions([1])

    assert plan.necessary_versions([]) == ()


def test_deferred_schedule_skips_irrelevant_updates(tmp_path) -> None:
    turns = [
        {
            "feedback": "initial",
            "actions": [{
                "action": "Commit",
                "resolved_requirement": "Compute: 2 + 3",
                "base_version_id": None,
                "update_hint": "",
            }],
        },
        {
            "feedback": "branch",
            "actions": [{
                "action": "Commit",
                "resolved_requirement": "Compute: 2 + 5",
                "base_version_id": "v0",
                "update_hint": "change",
            }],
        },
        {
            "feedback": "return",
            "actions": [{"action": "Reuse", "version_id": "v0"}],
        },
    ]
    calls = {"eager": 0, "deferred": 0}

    def update_for(schedule):
        def update(workspace, requirement, hint):
            del requirement, hint
            calls[schedule] += 1
            artifact = "Final answer: \\boxed{7}\n"
            (workspace / "answer.md").write_text(artifact, encoding="utf-8")
            return ExecutionResult(
                True,
                artifact,
            )

        return update

    common = {
        "task_id": "schedule-test",
        "domain": "math",
        "executor": "offline_math",
        "turns": turns,
    }
    eager = _execute_case(
        common,
        tmp_path / "eager",
        update_for("eager"),
        DeterministicMathRootExecutor(),
        ConservativeExecutionRouter(),
        scripted=True,
    )
    deferred = _execute_case(
        {**common, "execution_schedule": "deferred_necessary_path"},
        tmp_path / "deferred",
        update_for("deferred"),
        DeterministicMathRootExecutor(),
        ConservativeExecutionRouter(),
        scripted=True,
    )

    assert calls == {"eager": 1, "deferred": 0}
    assert eager["turns"][1]["checkpoint_id"] is not None
    assert deferred["turns"][1]["materialized"] is False
    assert eager["turns"][2]["final_response"] == deferred["turns"][2]["final_response"]
