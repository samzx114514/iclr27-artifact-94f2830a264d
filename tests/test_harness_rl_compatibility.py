from __future__ import annotations

import json
from pathlib import Path

import pytest

from githarness.branch_execution import BranchExecutionRuntime, ConservativeExecutionRouter
from githarness.compatibility import export_harness_rl_turn, validate_harness_rl_export
from githarness.domains import get_domain_spec
from githarness.offline import DeterministicMathRootExecutor, deterministic_math_update
from githarness.requirement_resolution import VersionHandoff
from githarness.task_native_update import CallableTaskNativeUpdateAgent
from githarness.version_interface import FileWorkspaceVersionInterface
from githarness.versioning import VersionGraph


def _export(tmp_path: Path) -> dict:
    runtime = BranchExecutionRuntime(
        graph=VersionGraph(),
        checkpoints={},
        bindings={},
        router=ConservativeExecutionRouter(),
        interface=FileWorkspaceVersionInterface(
            tmp_path, get_domain_spec("math").finalizer
        ),
        domain="math",
        root_executor=DeterministicMathRootExecutor(),
        update_agent=CallableTaskNativeUpdateAgent(deterministic_math_update),
        public_runtime_assets={"capabilities": ["calculator"]},
    )
    handoff = VersionHandoff(
        "Compute: 2 + 3",
        None,
        "v0",
        "",
        events=({"step": 0, "action": "Commit", "status": "ok"},),
        task_id="compat-math",
        domain="math",
    )
    outcome = runtime.execute(task_id="compat-math", source_turn=0, handoff=handoff)
    checkpoint = runtime.checkpoints[outcome.checkpoint_id]
    return export_harness_rl_turn(
        task_id="compat-math",
        domain="math",
        source_turn=0,
        handoff=handoff,
        outcome=outcome,
        checkpoint=checkpoint,
        evaluator_result={
            "status": "correct",
            "finished": True,
            "correct": True,
            "score": 1.0,
        },
    )


def test_harness_rl_golden_fixture_matches_public_export(tmp_path: Path) -> None:
    exported = _export(tmp_path)
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "harness_rl_compat_v1.json")
        .read_text(encoding="utf-8")
    )
    assert exported == fixture
    validate_harness_rl_export(exported)


def test_harness_rl_schema_matches_export_fields(tmp_path: Path) -> None:
    exported = _export(tmp_path)
    schema = json.loads(
        (Path(__file__).parents[1] / "schemas" / "harness-rl-compat-v1.schema.json")
        .read_text(encoding="utf-8")
    )
    assert set(schema["required"]) == set(exported)


def test_harness_rl_export_rejects_digest_tampering(tmp_path: Path) -> None:
    exported = _export(tmp_path)
    exported["execution"]["final_response"] = "tampered"
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_harness_rl_export(exported)


@pytest.mark.parametrize(
    "hidden",
    ["gold_sql", "hidden_tests", "grader_feedback", "reference_report"],
)
def test_harness_rl_export_rejects_hidden_evaluator_canary(
    tmp_path: Path, hidden: str
) -> None:
    exported = _export(tmp_path)
    exported["evaluator_result"] = {hidden: "HIDDEN_CANARY"}
    exported.pop("export_digest")
    from githarness.versioning.models import digest

    exported["export_digest"] = digest(exported)
    with pytest.raises(ValueError, match="forbidden private field"):
        validate_harness_rl_export(exported)
