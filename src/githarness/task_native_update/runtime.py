"""Candidate-only Task-Native Update Agent runtime adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from githarness.version_interface.protocol import ExecutionResult
from githarness.versioning.models import WorkState

from .contracts import UpdateAgentRequest, UpdateAgentResult, UpdateExecutionFailure


LegacyUpdateCallable = Callable[[Path, str, str], ExecutionResult]


@dataclass
class CallableTaskNativeUpdateAgent:
    """Compatibility boundary for deterministic or legacy domain executors."""

    callable: LegacyUpdateCallable

    def execute(self, request: UpdateAgentRequest) -> UpdateAgentResult:
        try:
            raw = self.callable(
                request.candidate.root,
                request.resolved_requirement,
                request.update_hint,
            )
        except Exception as error:
            return self._failure(request, "infrastructure_failure", type(error).__name__, True)
        work_state = _read_candidate(request)
        if not raw.finished:
            return UpdateAgentResult(
                False,
                request.domain_spec.artifact_path,
                raw.final_response,
                work_state,
                trace=raw.trace,
                failure=UpdateExecutionFailure(
                    "update_failure", raw.error or "Task-Native Update Agent did not finish"
                ),
            )
        return UpdateAgentResult(
            True,
            request.domain_spec.artifact_path,
            raw.final_response,
            work_state,
            trace=raw.trace,
        )

    @staticmethod
    def _failure(
        request: UpdateAgentRequest,
        kind: str,
        message: str,
        retryable: bool,
    ) -> UpdateAgentResult:
        return UpdateAgentResult(
            False,
            request.domain_spec.artifact_path,
            "",
            _read_candidate(request),
            failure=UpdateExecutionFailure(kind, message, retryable),
        )


def _read_candidate(request: UpdateAgentRequest) -> WorkState:
    files: dict[str, str] = {}
    for path in sorted(request.candidate.root.rglob("*")):
        if path.is_file():
            files[path.relative_to(request.candidate.root).as_posix()] = path.read_text(
                encoding="utf-8"
            )
    return WorkState(files, request.candidate.metadata)


def candidate_work_state(request: UpdateAgentRequest) -> WorkState:
    return _read_candidate(request)
