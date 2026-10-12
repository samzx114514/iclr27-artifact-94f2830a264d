"""State 0 candidate finalization and accepted-checkpoint binding."""

from __future__ import annotations

from typing import Any, Mapping

from githarness.versioning.models import Checkpoint

from .contracts import State0, State0Candidate


def accept_state0(
    candidate: State0Candidate,
    *,
    checkpoint: Checkpoint,
    receipt: Mapping[str, Any],
) -> State0:
    """Build an accepted State 0 only after checkpoint persistence succeeds."""

    result = candidate.root_result
    request = candidate.request
    if not result.finished:
        raise ValueError("failed State 0 candidate cannot be accepted")
    if checkpoint.state0_id != candidate.state0_id:
        raise ValueError("checkpoint is not bound to the State 0 candidate")
    if checkpoint.origin_state0_id != candidate.state0_id:
        raise ValueError("root checkpoint must originate from its own State 0")
    if checkpoint.work_state.files.get(result.artifact_path) != result.artifact:
        raise ValueError("checkpoint artifact does not match root result")
    return State0(
        state0_id=candidate.state0_id,
        task_id=request.task_id,
        domain=request.domain,
        resolved_requirement=request.resolved_requirement,
        requirement_digest=request.requirement_digest,
        public_runtime_assets=request.public_runtime_assets,
        initial_workspace=checkpoint.work_state,
        root_plan=result.root_plan,
        root_work_state=result.work_state,
        artifact_path=result.artifact_path,
        artifact=result.artifact,
        root_trace=result.trace,
        finalization_receipt=receipt,
        checkpoint_id=checkpoint.checkpoint_id,
        checkpoint_digest=checkpoint.digest,
        workspace_digest=checkpoint.work_state.digest,
    )
