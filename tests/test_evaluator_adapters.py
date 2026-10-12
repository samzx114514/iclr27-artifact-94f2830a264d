from __future__ import annotations

import json
from pathlib import Path

import pytest

from githarness.errors import TaskEvaluationError
from mtagentbench.batch_evaluation import evaluate_files
from mtagentbench.contracts import PrivateEvaluationRecord
from mtagentbench.evaluators import EvaluationContext, get_evaluator
from mtagentbench.evaluators.registry import EVALUATORS


ROOT = Path(__file__).resolve().parents[1]


def test_evaluator_registry_is_closed_and_matches_private_verifiers() -> None:
    assert set(EVALUATORS) == {"math", "sql", "search", "research", "code"}
    assert {domain: adapter.verifier for domain, adapter in EVALUATORS.items()} == {
        "math": "normalized_exact_match",
        "sql": "bird_execution",
        "search": "browsecomp_normalized_em",
        "research": "deep_research_race",
        "code": "swebench_official",
    }


def test_adapter_rejects_a_mismatched_private_verifier() -> None:
    record = PrivateEvaluationRecord(
        "math-1", "math", "wrong_verifier", {"answer": "1"},
    )
    with pytest.raises(TaskEvaluationError, match="verifier mismatch"):
        get_evaluator("math").evaluate("Answer: 1", record, EvaluationContext())


def test_batch_maps_verifier_mismatch_to_verifier_failure(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    private = tmp_path / "private.jsonl"
    predictions.write_text(
        '{"task_id":"math-1","finished":true,"artifact":"Answer: 1"}\n',
        encoding="utf-8",
    )
    private.write_text(
        '{"task_id":"math-1","domain":"math","verifier":"wrong",'
        '"reference":{"answer":"1"}}\n',
        encoding="utf-8",
    )
    results, summary = evaluate_files(predictions, private)
    assert results[0].status == "verifier_failure"
    assert summary["verifier_failure"] == 1


def test_evaluation_result_schema_matches_the_adapter_boundary() -> None:
    schema = json.loads(
        (ROOT / "schemas/evaluation-result-v1.schema.json").read_text()
    )
    assert set(schema["required"]) == {
        "task_id", "domain", "status", "finished", "correct", "score",
        "details",
    }
    assert set(schema["properties"]["domain"]["enum"]) == set(EVALUATORS)
    assert "scored" in schema["properties"]["status"]["enum"]
