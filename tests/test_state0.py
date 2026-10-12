from __future__ import annotations

from pathlib import Path
import json

import pytest

from githarness.branch_execution import BranchExecutionRuntime, ConservativeExecutionRouter
from githarness.domains import get_domain_spec
from githarness.errors import (
    RootExecutionError,
    RuntimeFinalizationError,
    State0SerializationError,
)
from githarness.offline import DeterministicMathRootExecutor, deterministic_math_update
from githarness.requirement_resolution.agent import VersionHandoff
from githarness.state0 import (
    ExecutionFailure,
    StackPlannerRootResult,
    State0Request,
    state0_from_dict,
    state0_to_dict,
)
from githarness.task_native_update import CallableTaskNativeUpdateAgent
from githarness.version_interface import FileWorkspaceVersionInterface
from githarness.versioning import VersionGraph


def _runtime(tmp_path: Path, root=None) -> BranchExecutionRuntime:
    interface = FileWorkspaceVersionInterface(
        tmp_path,
        get_domain_spec("math").finalizer,
    )
    return BranchExecutionRuntime(
        graph=VersionGraph(),
        checkpoints={},
        bindings={},
        router=ConservativeExecutionRouter(),
        interface=interface,
        domain="math",
        root_executor=root or DeterministicMathRootExecutor(),
        update_agent=CallableTaskNativeUpdateAgent(deterministic_math_update),
        public_runtime_assets={"calculator": "decimal"},
    )


def _create_state0(tmp_path: Path):
    return _runtime(tmp_path).execute(
        task_id="math-case",
        source_turn=0,
        handoff=VersionHandoff("Compute: 2 + 3", None, "v0", ""),
    ).state0


def test_state0_schema_round_trip_and_checkpoint_binding(tmp_path: Path) -> None:
    state0 = _create_state0(tmp_path)
    assert state0 is not None
    payload = state0_to_dict(state0)
    restored = state0_from_dict(payload)

    assert state0_to_dict(restored) == payload
    assert restored.workspace_digest == restored.initial_workspace.digest
    assert restored.finalization_receipt["workspace_digest"] == restored.workspace_digest
    assert restored.initial_workspace.metadata["state0_root_work_state"] == {
        "calculator": "decimal"
    }
    assert restored.checkpoint_id == "checkpoint-v0"


def test_checked_in_state0_schema_matches_serialized_fields(tmp_path: Path) -> None:
    state0 = _create_state0(tmp_path)
    assert state0 is not None
    payload = state0_to_dict(state0)
    schema_path = Path(__file__).parents[1] / "schemas" / "state0-v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["properties"]["schema_version"]["const"] == 1
    assert set(schema["required"]) == set(payload)


def test_state0_schema_rejects_digest_tampering(tmp_path: Path) -> None:
    state0 = _create_state0(tmp_path)
    assert state0 is not None
    payload = state0_to_dict(state0)
    payload["workspace_digest"] = "forged"
    with pytest.raises(State0SerializationError, match="workspace digest"):
        state0_from_dict(payload)


def test_state0_workspace_and_checkpoint_digests_are_stable(tmp_path: Path) -> None:
    first = _create_state0(tmp_path / "first")
    second = _create_state0(tmp_path / "second")
    assert first is not None and second is not None
    assert first.workspace_digest == second.workspace_digest
    assert first.checkpoint_digest == second.checkpoint_digest


@pytest.mark.parametrize(
    "private_field",
    ["hidden_tests", "gold_sql", "reference_report", "grader_feedback", "api_key"],
)
def test_state0_request_rejects_hidden_runtime_assets(private_field: str) -> None:
    with pytest.raises(ValueError, match="forbidden private field"):
        State0Request(
            task_id="case",
            domain="math",
            resolved_requirement="Compute: 2 + 3",
            public_runtime_assets={private_field: "secret-canary"},
        )


def test_root_failure_discards_candidate_and_history(tmp_path: Path) -> None:
    class FailingRoot:
        def create_state0(self, candidate, request):
            return StackPlannerRootResult(
                False,
                "answer.md",
                "",
                failure=ExecutionFailure("root_failure", "root did not finish"),
            )

    runtime = _runtime(tmp_path, FailingRoot())
    with pytest.raises(RootExecutionError, match="root did not finish"):
        runtime.execute(
            task_id="case",
            source_turn=0,
            handoff=VersionHandoff("Compute: 2 + 3", None, "v0", ""),
        )
    assert runtime.graph.versions == ()
    assert runtime.checkpoints == {}
    assert runtime.bindings == {}
    assert runtime.state0s == {}
    assert not any((tmp_path / "candidates").iterdir())
    assert not any((tmp_path / "snapshots").iterdir())


def test_state0_finalization_failure_creates_no_checkpoint(tmp_path: Path) -> None:
    class EmptyArtifactRoot:
        def create_state0(self, candidate, request):
            (candidate.root / "answer.md").write_text("", encoding="utf-8")
            return StackPlannerRootResult(True, "answer.md", "")

    runtime = _runtime(tmp_path, EmptyArtifactRoot())
    with pytest.raises(RuntimeFinalizationError, match="empty artifact"):
        runtime.execute(
            task_id="case",
            source_turn=0,
            handoff=VersionHandoff("Compute: 2 + 3", None, "v0", ""),
        )
    assert runtime.graph.versions == ()
    assert runtime.checkpoints == {}
    assert runtime.bindings == {}
    assert not any((tmp_path / "snapshots").iterdir())


def test_root_result_rejects_hidden_canary() -> None:
    with pytest.raises(ValueError, match="forbidden private field"):
        StackPlannerRootResult(
            True,
            "answer.md",
            "public output",
            work_state={"hidden_tests": "secret-canary"},
        )


@pytest.mark.parametrize(
    ("domain", "graph_format", "capabilities"),
    [
        ("math", "sp_math", ("calculator",)),
        ("sql", "sp_sql", ()),
        ("search", "sp_search", ("search(query)",)),
        ("research", "sp_deep_research", ("search(query)",)),
        ("code", "sp_swe", ("/testbed",)),
    ],
)
def test_five_domain_specs_bind_root_graph_and_public_capability(
    domain: str,
    graph_format: str,
    capabilities: tuple[str, ...],
) -> None:
    spec = get_domain_spec(domain)
    assert spec.graph_format == graph_format
    assert spec.public_capabilities == capabilities
