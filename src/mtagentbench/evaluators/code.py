"""Adapter for the separately installed official SWE-bench harness."""

from __future__ import annotations

from typing import Any, Mapping

from githarness.errors import InfrastructureError, TaskEvaluationError

from ..contracts import EvaluationResult, PrivateEvaluationRecord
from .base import Artifact, CodeGrader, EvaluationContext, require_record


def evaluate_code(
    task_id: str, patch: str, record: PrivateEvaluationRecord,
    grader: CodeGrader,
) -> EvaluationResult:
    instance_id = record.asset_bindings.get("instance_id")
    if not isinstance(instance_id, str):
        raise TaskEvaluationError("Code evaluator record has no instance_id")
    try:
        outcome: Mapping[str, Any] = dict(grader(instance_id, patch))
    except Exception as error:
        raise InfrastructureError(
            f"SWE-bench grader failed: {type(error).__name__}: {error}"
        ) from error
    if outcome.get("infrastructure_failure"):
        return EvaluationResult(
            task_id, "code", "infrastructure_failure", True, None, None,
            outcome,
        )
    correct = bool(outcome.get("resolved"))
    return EvaluationResult(
        task_id, "code", "correct" if correct else "incorrect", True,
        correct, float(correct), outcome,
    )


class CodeEvaluator:
    domain = "code"
    verifier = "swebench_official"

    def evaluate(
        self, artifact: Artifact, record: PrivateEvaluationRecord,
        context: EvaluationContext,
    ) -> EvaluationResult:
        require_record(record, domain=self.domain, verifier=self.verifier)
        if context.code_grader is None:
            raise InfrastructureError("Code evaluation requires --code-grader")
        return evaluate_code(
            record.task_id, str(artifact), record, context.code_grader,
        )
