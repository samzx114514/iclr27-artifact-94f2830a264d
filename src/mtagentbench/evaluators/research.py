"""Official-compatible DeepResearch Bench RACE evaluator adapter."""

from __future__ import annotations

from githarness.errors import InfrastructureError, TaskEvaluationError

from ..contracts import EvaluationResult, PrivateEvaluationRecord
from .base import Artifact, EvaluationContext, ResearchJudge, require_record
from .race import (
    UPSTREAM_COMMIT,
    parse_race_judgment,
    race_input_from_record,
    render_race_prompt,
    score_race_judgment,
)


def evaluate_research(
    task_id: str, article: str, record: PrivateEvaluationRecord,
    judge: ResearchJudge,
) -> EvaluationResult:
    if not article.strip():
        raise TaskEvaluationError("Research artifact is empty")
    race_input = race_input_from_record(record)
    prompt = render_race_prompt(article, race_input)
    try:
        judgment = judge(prompt, record)
    except InfrastructureError:
        raise
    except Exception as error:
        raise TaskEvaluationError(
            f"Research judge failed: {type(error).__name__}: {error}"
        ) from error
    judgment = parse_race_judgment(judgment)
    score = score_race_judgment(judgment, race_input.criteria)
    return EvaluationResult(
        task_id,
        "research",
        "scored",
        True,
        None,
        score.overall_score,
        {
            "metric": "race",
            "benchmark_id": race_input.benchmark_id,
            "upstream_commit": UPSTREAM_COMMIT,
            "dimension_scores": dict(score.dimension_scores),
            "target_dimension_scores": dict(score.target_dimension_scores),
            "reference_dimension_scores": dict(score.reference_dimension_scores),
            "target_weighted_score": score.target_weighted_score,
            "reference_weighted_score": score.reference_weighted_score,
            "criterion_scores": {
                dimension: list(rows)
                for dimension, rows in score.criterion_scores.items()
            },
        },
    )


class ResearchEvaluator:
    domain = "research"
    verifier = "deep_research_race"

    def evaluate(
        self, artifact: Artifact, record: PrivateEvaluationRecord,
        context: EvaluationContext,
    ) -> EvaluationResult:
        require_record(record, domain=self.domain, verifier=self.verifier)
        if context.research_judge is None:
            raise InfrastructureError(
                "Research evaluation requires --research-judge"
            )
        return evaluate_research(
            record.task_id, str(artifact), record, context.research_judge,
        )
