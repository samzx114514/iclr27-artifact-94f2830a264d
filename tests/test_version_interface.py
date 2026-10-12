from __future__ import annotations

from pathlib import Path

import pytest

from githarness.domains import get_domain_spec
from githarness.version_interface import ExecutionResult, FileWorkspaceVersionInterface
from githarness.versioning import Checkpoint, WorkState


def test_failed_checkpoint_construction_removes_orphan_snapshot(tmp_path: Path) -> None:
    interface = FileWorkspaceVersionInterface(
        tmp_path,
        get_domain_spec("math").finalizer,
    )
    root = Checkpoint(
        checkpoint_id="checkpoint-v0",
        version_id="v0",
        work_state=WorkState({"answer.md": "\\boxed{5}"}),
        state0_id="state0-v0",
        origin_state0_id="state0-v0",
    )
    candidate = interface.fork("case", root)
    result = ExecutionResult(True, "\\boxed{7}")
    receipt = interface.finalize(candidate, result)

    with pytest.raises(ValueError, match="origin State 0"):
        interface.commit(
            candidate,
            checkpoint_id="checkpoint-v1",
            version_id="v1",
            receipt=receipt,
            final_response=result.final_response,
            origin_state0_id=None,
        )

    assert not (tmp_path / "snapshots" / "checkpoint-v1").exists()
    assert candidate.root.exists()
    interface.discard(candidate)
    assert not candidate.root.exists()


def test_rollback_commit_removes_persisted_snapshot(tmp_path: Path) -> None:
    interface = FileWorkspaceVersionInterface(
        tmp_path,
        get_domain_spec("math").finalizer,
    )
    candidate = interface.empty("case")
    (candidate.root / "answer.md").write_text("\\boxed{5}", encoding="utf-8")
    result = ExecutionResult(True, "\\boxed{5}")
    receipt = interface.finalize(candidate, result)
    checkpoint = interface.commit(
        candidate,
        checkpoint_id="checkpoint-v0",
        version_id="v0",
        receipt=receipt,
        final_response=result.final_response,
        state0_id="state0-v0",
        origin_state0_id="state0-v0",
    )
    snapshot = tmp_path / "snapshots" / checkpoint.checkpoint_id
    assert snapshot.exists()

    interface.rollback_commit(checkpoint)

    assert not snapshot.exists()


def test_forked_nested_metadata_cannot_mutate_accepted_checkpoint(tmp_path: Path) -> None:
    interface = FileWorkspaceVersionInterface(
        tmp_path,
        get_domain_spec("search").finalizer,
    )
    source_metadata = {"search_receipts": [{"docid": "ancestor"}]}
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint-v0",
        version_id="v0",
        work_state=WorkState(
            {"answer.json": '{"exact_answer":"old"}'},
            source_metadata,
        ),
        state0_id="state0-v0",
        origin_state0_id="state0-v0",
    )
    source_metadata["search_receipts"].append({"docid": "outside"})
    candidate = interface.fork("case", checkpoint)
    candidate.metadata["search_receipts"].append({"docid": "descendant"})

    assert checkpoint.work_state.metadata["search_receipts"] == [
        {"docid": "ancestor"}
    ]
