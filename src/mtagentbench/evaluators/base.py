"""Shared evaluator adapter contract and runtime dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from githarness.errors import TaskEvaluationError

from ..contracts import Domain, EvaluationResult, PrivateEvaluationRecord


ResearchJudge = Callable[[str, PrivateEvaluationRecord], str | Mapping[str, Any]]
CodeGrader = Callable[[str, str], Mapping[str, Any]]
Artifact = str | Mapping[str, Any]


@dataclass(frozen=True)
class EvaluationContext:
    """External evaluator assets supplied only after generation finishes."""

    db_root: Path | None = None
    research_judge: ResearchJudge | None = None
    code_grader: CodeGrader | None = None


class EvaluatorAdapter(Protocol):
    domain: Domain
    verifier: str

    def evaluate(
        self, artifact: Artifact, record: PrivateEvaluationRecord,
        context: EvaluationContext,
    ) -> EvaluationResult: ...


def require_record(
    record: PrivateEvaluationRecord, *, domain: Domain, verifier: str,
) -> None:
    if record.domain != domain:
        raise TaskEvaluationError(
            f"evaluator domain mismatch: expected {domain}, got {record.domain}"
        )
    if record.verifier != verifier:
        raise TaskEvaluationError(
            f"evaluator verifier mismatch: expected {verifier}, got {record.verifier}"
        )
