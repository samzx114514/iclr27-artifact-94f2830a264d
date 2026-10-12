from __future__ import annotations

from pathlib import Path

import pytest

from githarness.branch_execution import BranchExecutionRuntime, ConservativeExecutionRouter, ExecutionMode
from githarness.domains import get_domain_spec
from githarness.errors import InfrastructureError, ProtocolError, RuntimeFinalizationError
from githarness.offline import DeterministicMathRootExecutor, deterministic_math_update
from githarness.requirement_resolution.handoff import CheckpointBinding, VersionHandoff
from githarness.task_native_update import CallableTaskNativeUpdateAgent
from githarness.version_interface import ExecutionResult, FileWorkspaceVersionInterface
from githarness.versioning import VersionGraph


def _runtime(tmp_path: Path, update=deterministic_math_update) -> BranchExecutionRuntime:
    interface = FileWorkspaceVersionInterface(tmp_path, get_domain_spec("math").finalizer)
    return BranchExecutionRuntime(
        graph=VersionGraph(),
        checkpoints={},
        bindings={},
        router=ConservativeExecutionRouter(),
        interface=interface,
        domain="math",
        root_executor=DeterministicMathRootExecutor(),
        update_agent=CallableTaskNativeUpdateAgent(update),
    )


def test_fresh_reuse_and_patch_and_exact_reuse(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    first = runtime.execute(
        task_id="case",
        source_turn=0,
        handoff=VersionHandoff("Compute: 2 + 3", None, "v0", ""),
    )
    assert first.mode is ExecutionMode.FRESH_SOLVE
    assert first.state0 is not None
    assert first.state0.checkpoint_id == first.checkpoint_id
    assert "\\boxed{5}" in first.work_state.files["answer.md"]

    second = runtime.execute(
        task_id="case",
        source_turn=1,
        handoff=VersionHandoff("Compute: 2 + 5", "v0", "v1", "replace operand"),
    )
    assert second.mode is ExecutionMode.REUSE_AND_PATCH
    second_checkpoint = runtime.checkpoints[second.checkpoint_id]
    assert second_checkpoint.parent_checkpoint_id == first.checkpoint_id
    assert second_checkpoint.origin_state0_id == first.state0.state0_id
    assert "\\boxed{7}" in second.work_state.files["answer.md"]
    assert "\\boxed{5}" in first.work_state.files["answer.md"]

    reused = runtime.execute(
        task_id="case",
        source_turn=2,
        handoff=VersionHandoff("Compute: 2 + 3", None, "v0", "", reused_version_id="v0"),
    )
    assert reused.mode is ExecutionMode.EXACT_REUSE
    assert reused.final_response == first.final_response


def test_failed_candidate_does_not_mutate_graph(tmp_path: Path) -> None:
    def fail(workspace: Path, requirement: str, update_hint: str) -> ExecutionResult:
        return ExecutionResult(False, "", error="expected failure")

    runtime = _runtime(tmp_path, fail)
    runtime.execute(
        task_id="case",
        source_turn=0,
        handoff=VersionHandoff("Compute: 2 + 3", None, "v0", ""),
    )
    with pytest.raises(RuntimeFinalizationError):
        runtime.execute(
            task_id="case",
            source_turn=1,
            handoff=VersionHandoff("Compute: 2 + 5", "v0", "v1", "change operand"),
        )
    assert [version.version_id for version in runtime.graph.versions] == ["v0"]
    assert set(runtime.bindings) == {"v0"}
    assert not any((tmp_path / "candidates").iterdir())


def test_ordinary_fresh_solve_calls_update_but_not_root(tmp_path: Path) -> None:
    class CountingRoot(DeterministicMathRootExecutor):
        def __init__(self):
            self.calls = 0

        def create_state0(self, candidate, request):
            self.calls += 1
            return super().create_state0(candidate, request)

    update_calls = []

    def update(workspace: Path, requirement: str, update_hint: str) -> ExecutionResult:
        update_calls.append((requirement, update_hint))
        return deterministic_math_update(workspace, requirement, update_hint)

    root = CountingRoot()
    interface = FileWorkspaceVersionInterface(
        tmp_path,
        get_domain_spec("math").finalizer,
    )
    runtime = BranchExecutionRuntime(
        VersionGraph(),
        {},
        {},
        ConservativeExecutionRouter(),
        interface,
        "math",
        root,
        CallableTaskNativeUpdateAgent(update),
    )
    runtime.execute(
        task_id="case",
        source_turn=0,
        handoff=VersionHandoff("Compute: 2 + 3", None, "v0", ""),
    )
    second = runtime.execute(
        task_id="case",
        source_turn=1,
        handoff=VersionHandoff("Compute: 8 + 1", "v0", "v1", ""),
    )

    assert second.mode is ExecutionMode.FRESH_SOLVE
    assert root.calls == 1
    assert update_calls == [("Compute: 8 + 1", "")]
    assert second.update_result is not None
    assert second.update_result.modified_work_state == second.work_state


def test_runtime_rejects_tampered_git_agent_checkpoint_binding(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    first = runtime.execute(
        task_id="task",
        source_turn=0,
        handoff=VersionHandoff("Compute: 2 + 3", None, "v0", ""),
    )
    binding = CheckpointBinding.from_checkpoint(runtime.checkpoints[first.checkpoint_id])
    tampered = CheckpointBinding(
        binding.version_id,
        binding.checkpoint_id,
        "0" * 64,
        binding.workspace_digest,
        binding.origin_state0_id,
    )

    with pytest.raises(ProtocolError, match="checkpoint binding mismatch"):
        runtime.execute(
            task_id="task",
            source_turn=1,
            handoff=VersionHandoff(
                "Compute: 2 + 4",
                "v0",
                "v1",
                "replace operand",
                selected_base_checkpoint=tampered,
            ),
        )

    assert len(runtime.graph.versions) == 1


def test_binding_failure_rolls_back_persisted_checkpoint(tmp_path: Path) -> None:
    class RejectingGraph(VersionGraph):
        def add(self, version, *, activate=True):
            raise RuntimeError("simulated binding failure")

    interface = FileWorkspaceVersionInterface(
        tmp_path,
        get_domain_spec("math").finalizer,
    )
    runtime = BranchExecutionRuntime(
        graph=RejectingGraph(),
        checkpoints={},
        bindings={},
        router=ConservativeExecutionRouter(),
        interface=interface,
        domain="math",
        root_executor=DeterministicMathRootExecutor(),
        update_agent=CallableTaskNativeUpdateAgent(deterministic_math_update),
    )

    with pytest.raises(InfrastructureError, match="checkpoint binding failed"):
        runtime.execute(
            task_id="case",
            source_turn=0,
            handoff=VersionHandoff("Compute: 2 + 3", None, "v0", ""),
        )

    assert runtime.graph.versions == ()
    assert runtime.checkpoints == {}
    assert runtime.bindings == {}
    assert runtime.state0s == {}
    assert not (tmp_path / "snapshots" / "checkpoint-v0").exists()
