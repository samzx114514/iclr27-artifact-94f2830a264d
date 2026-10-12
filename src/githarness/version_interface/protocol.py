"""Unified Version Interface from Algorithm 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from githarness.versioning.models import Checkpoint, WorkState


@dataclass
class CandidateWorkState:
    candidate_id: str
    root: Path
    base_checkpoint_id: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionResult:
    finished: bool
    final_response: str
    trace: tuple[Mapping[str, Any], ...] = ()
    error: str | None = None


class UnifiedVersionInterface(Protocol):
    def restore(self, checkpoint: Checkpoint) -> WorkState: ...
    def inspect(self, candidate: CandidateWorkState) -> WorkState: ...
    def empty(self, task_id: str) -> CandidateWorkState: ...
    def fork(self, task_id: str, checkpoint: Checkpoint) -> CandidateWorkState: ...
    def finalize(self, candidate: CandidateWorkState, result: ExecutionResult) -> Mapping[str, Any]: ...
    def commit(
        self,
        candidate: CandidateWorkState,
        *,
        checkpoint_id: str,
        version_id: str,
        receipt: Mapping[str, Any],
        final_response: str,
        state0_id: str | None = None,
        origin_state0_id: str | None = None,
    ) -> Checkpoint: ...
    def rollback_commit(self, checkpoint: Checkpoint) -> None: ...
    def discard(self, candidate: CandidateWorkState) -> None: ...
