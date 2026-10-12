"""Filesystem implementation of the Unified Version Interface."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping
import uuid

from githarness.errors import RuntimeFinalizationError
from githarness.versioning.models import Checkpoint, WorkState

from .protocol import CandidateWorkState, ExecutionResult


Finalizer = Callable[[Path, ExecutionResult], Mapping[str, Any]]


def read_workspace(root: Path, metadata: Mapping[str, Any] | None = None) -> WorkState:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            files[relative] = path.read_text(encoding="utf-8")
    return WorkState(files=files, metadata=metadata or {})


@dataclass
class FileWorkspaceVersionInterface:
    root: Path
    finalizer: Finalizer

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.candidates = self.root / "candidates"
        self.snapshots = self.root / "snapshots"
        self.candidates.mkdir(parents=True, exist_ok=True)
        self.snapshots.mkdir(parents=True, exist_ok=True)

    def restore(self, checkpoint: Checkpoint) -> WorkState:
        return checkpoint.work_state

    def inspect(self, candidate: CandidateWorkState) -> WorkState:
        return read_workspace(candidate.root, candidate.metadata)

    def empty(self, task_id: str) -> CandidateWorkState:
        return self._candidate(task_id, None, None)

    def fork(self, task_id: str, checkpoint: Checkpoint) -> CandidateWorkState:
        candidate = self._candidate(task_id, checkpoint.checkpoint_id, checkpoint.work_state)
        candidate.metadata["origin_state0_id"] = checkpoint.origin_state0_id
        return candidate

    def _candidate(
        self,
        task_id: str,
        base_checkpoint_id: str | None,
        work_state: WorkState | None,
    ) -> CandidateWorkState:
        candidate_id = f"candidate-{uuid.uuid4().hex[:12]}"
        root = self.candidates / candidate_id
        root.mkdir(parents=True, exist_ok=False)
        if work_state is not None:
            for relative, content in work_state.files.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
        metadata = copy.deepcopy(dict(work_state.metadata)) if work_state is not None else {}
        metadata["task_id"] = task_id
        return CandidateWorkState(candidate_id, root, base_checkpoint_id, metadata)

    def finalize(self, candidate: CandidateWorkState, result: ExecutionResult) -> Mapping[str, Any]:
        if not result.finished:
            raise RuntimeFinalizationError(result.error or "update agent did not finish")
        receipt = dict(self.finalizer(candidate.root, result))
        if receipt.get("passed") is not True:
            raise RuntimeFinalizationError(str(receipt.get("error") or "candidate finalization failed"))
        receipt["workspace_digest"] = self.inspect(candidate).digest
        return receipt

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
    ) -> Checkpoint:
        work_state = self.inspect(candidate)
        snapshot = self.snapshots / checkpoint_id
        if snapshot.exists():
            raise ValueError(f"checkpoint snapshot already exists: {checkpoint_id}")
        try:
            shutil.copytree(candidate.root, snapshot)
            checkpoint = Checkpoint(
                checkpoint_id=checkpoint_id,
                version_id=version_id,
                work_state=work_state,
                parent_checkpoint_id=candidate.base_checkpoint_id,
                receipt=receipt,
                final_response=final_response,
                state0_id=state0_id,
                origin_state0_id=origin_state0_id,
            )
        except Exception:
            shutil.rmtree(snapshot, ignore_errors=True)
            raise
        self.discard(candidate)
        return checkpoint

    def rollback_commit(self, checkpoint: Checkpoint) -> None:
        """Remove a persisted snapshot whose Version binding was not accepted."""

        shutil.rmtree(self.snapshots / checkpoint.checkpoint_id, ignore_errors=True)

    def discard(self, candidate: CandidateWorkState) -> None:
        shutil.rmtree(candidate.root, ignore_errors=True)
