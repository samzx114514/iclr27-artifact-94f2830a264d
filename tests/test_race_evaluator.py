from __future__ import annotations

import hashlib
from typing import Any, Mapping

import pytest

from githarness.errors import TaskEvaluationError
from mtagentbench.batch_evaluation import summarize
from mtagentbench.contracts import PrivateEvaluationRecord
from mtagentbench.evaluators import EvaluationContext, get_evaluator
from mtagentbench.evaluators.race import (
    DIMENSIONS,
    _PROMPT_EN,
    _PROMPT_ZH,
    parse_race_judgment,
    race_input_from_record,
    render_race_prompt,
    score_race_judgment,
)


TARGET = {
    "comprehensiveness": 8.0,
    "insight": 6.0,
    "instruction_following": 9.0,
    "readability": 4.0,
}
REFERENCE = {
    "comprehensiveness": 4.0,
    "insight": 6.0,
    "instruction_following": 3.0,
    "readability": 8.0,
}
DIMENSION_WEIGHTS = {
    "comprehensiveness": 0.4,
    "insight": 0.3,
    "instruction_following": 0.2,
    "readability": 0.1,
}


def test_official_prompt_snapshots_are_pinned() -> None:
    assert hashlib.sha256(_PROMPT_EN.encode()).hexdigest() == (
        "29c0dc8365d047b293c6e6dce80c840402003aa5171c47e2a6f19a064ca77a1f"
    )
    assert hashlib.sha256(_PROMPT_ZH.encode()).hexdigest() == (
        "d324077bbe7218ad1df5c49cf3d0207f66416526c73063be883203d7e80610c4"
    )


def _criteria() -> dict[str, Any]:
    return {
        "dimension_weight": dict(DIMENSION_WEIGHTS),
        "criterions": {
            dimension: [{
                "criterion": f"{dimension} criterion",
                "explanation": f"Evaluate {dimension}.",
                "weight": 1.0,
            }]
            for dimension in DIMENSIONS
        },
    }


def _record(*, language: str = "en") -> PrivateEvaluationRecord:
    return PrivateEvaluationRecord(
        "deep-research-bench-7",
        "research",
        "deep_research_race",
        {
            "prompt": "Assess the evidence and write a report.",
            "reference_article": "The official evaluator-only reference report.",
            "language": language,
            "criteria": _criteria(),
        },
        {"benchmark_id": 7},
    )


def _judgment() -> dict[str, Any]:
    return {
        dimension: [{
            "criterion": f"{dimension} criterion",
            "analysis": f"Comparative analysis for {dimension}.",
            "article_1_score": TARGET[dimension],
            "article_2_score": REFERENCE[dimension],
        }]
        for dimension in DIMENSIONS
    }


def test_official_race_formula_and_dimension_normalization() -> None:
    score = score_race_judgment(_judgment(), _criteria())

    assert score.target_weighted_score == pytest.approx(7.2)
    assert score.reference_weighted_score == pytest.approx(4.8)
    assert score.overall_score == pytest.approx(0.6)
    assert score.dimension_scores == pytest.approx({
        "comprehensiveness": 8 / 12,
        "insight": 0.5,
        "instruction_following": 0.75,
        "readability": 4 / 12,
    })


def test_race_parser_accepts_fenced_json_without_repairing_fields() -> None:
    import json

    parsed = parse_race_judgment("```json\n" + json.dumps(_judgment()) + "\n```")
    assert parsed == _judgment()
    with pytest.raises(TaskEvaluationError, match="invalid RACE JSON"):
        parse_race_judgment("not json")


@pytest.mark.parametrize("language", ["en", "zh"])
def test_race_prompt_contains_private_inputs_but_hides_weights(language: str) -> None:
    race_input = race_input_from_record(_record(language=language))
    prompt = render_race_prompt("The generated target report.", race_input)

    assert "The generated target report." in prompt
    assert "The official evaluator-only reference report." in prompt
    assert "comprehensiveness criterion" in prompt
    assert '"weight"' not in prompt


def test_research_evaluator_returns_continuous_scored_result() -> None:
    seen: dict[str, Any] = {}

    def judge(prompt: str, record: PrivateEvaluationRecord) -> Mapping[str, Any]:
        seen["prompt"] = prompt
        seen["task_id"] = record.task_id
        return _judgment()

    result = get_evaluator("research").evaluate(
        "The generated target report.",
        _record(),
        EvaluationContext(research_judge=judge),
    )

    assert seen["task_id"] == "deep-research-bench-7"
    assert "<article_1>" in seen["prompt"]
    assert result.status == "scored"
    assert result.correct is None
    assert result.score == pytest.approx(0.6)
    assert result.details["metric"] == "race"
    assert result.details["dimension_scores"]["insight"] == pytest.approx(0.5)


def test_race_summary_macro_averages_scores_and_dimensions() -> None:
    evaluator = get_evaluator("research")
    first = evaluator.evaluate(
        "Target one", _record(),
        EvaluationContext(research_judge=lambda prompt, record: _judgment()),
    )
    second = evaluator.evaluate(
        "Target two", _record(),
        EvaluationContext(research_judge=lambda prompt, record: _judgment()),
    )

    summary = summarize([first, second])
    assert summary["accuracy"] is None
    assert summary["scored"] == 2
    assert summary["research_race"]["num_completed"] == 2
    assert summary["research_race"]["num_errors"] == 0
    assert summary["research_race"]["mean_race"] == pytest.approx(0.6)
    assert summary["research_race"]["dimension_scores"]["readability"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("mutation", ["missing_dimension", "wrong_criterion", "score_out_of_range"])
def test_race_rejects_malformed_judgments(mutation: str) -> None:
    judgment = _judgment()
    if mutation == "missing_dimension":
        del judgment["readability"]
    elif mutation == "wrong_criterion":
        judgment["insight"][0]["criterion"] = "substitute criterion"
    else:
        judgment["comprehensiveness"][0]["article_1_score"] = 11

    with pytest.raises(TaskEvaluationError, match="RACE"):
        score_race_judgment(judgment, _criteria())


def test_race_rejects_legacy_arbitrary_score_plugin() -> None:
    def legacy(prompt: str, record: PrivateEvaluationRecord):
        return 0.9, {"claimed": "race"}

    with pytest.raises(TaskEvaluationError, match="empty RACE judgment"):
        get_evaluator("research").evaluate(
            "Target", _record(), EvaluationContext(research_judge=legacy),
        )
