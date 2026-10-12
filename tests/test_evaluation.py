from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mtagentbench.contracts import PrivateEvaluationRecord
from mtagentbench.batch_evaluation import evaluate_files
from mtagentbench.evaluation import evaluate_math, evaluate_search, evaluate_sql


def test_math_nested_box_and_search_em() -> None:
    math = PrivateEvaluationRecord("m", "math", "normalized_exact_match", {"answer": "42"})
    assert evaluate_math("m", r"Final: \boxed{\mathrm{42}}", math).correct
    search = PrivateEvaluationRecord("s", "search", "browsecomp_normalized_em", {"answer": "The Example University"})
    assert evaluate_search("s", {"exact_answer": "Example University"}, search).correct


@pytest.mark.parametrize(
    ("artifact", "expected"),
    [
        (r"Final: \boxed{6000\text{ gallons}}", "6000"),
        (r"Final: \boxed{\$18{,}000}", "18000"),
        (r"Final: \boxed{\frac{8}{3}\text{ bolts}}", "8/3"),
        (r"Final: \boxed{56 \text{ total items: } 20 \text{ flashlights}}", "56"),
    ],
)
def test_math_normalizes_boxed_units_currency_and_fractions(
    artifact: str, expected: str,
) -> None:
    record = PrivateEvaluationRecord(
        "m", "math", "normalized_exact_match", {"answer": expected},
    )
    assert evaluate_math("m", artifact, record).correct


def test_sql_execution_match(tmp_path: Path) -> None:
    db = tmp_path / "tiny.sqlite"
    connection = sqlite3.connect(db)
    connection.executescript("CREATE TABLE items(value INTEGER); INSERT INTO items VALUES (1), (2);")
    connection.commit()
    connection.close()
    record = PrivateEvaluationRecord("q", "sql", "bird_execution", {"gold_sql": "SELECT value FROM items"}, {"db_id": "tiny"})
    result = evaluate_sql("q", "```sql\nSELECT value FROM items;\n```", record, db)
    assert result.correct


def test_sql_prediction_timeout_is_incorrect(tmp_path: Path) -> None:
    db = tmp_path / "tiny.sqlite"
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE items(value INTEGER)")
    connection.execute("INSERT INTO items VALUES (1)")
    connection.commit()
    connection.close()
    record = PrivateEvaluationRecord(
        "q",
        "sql",
        "bird_execution",
        {"gold_sql": "SELECT value FROM items"},
        {"db_id": "tiny"},
    )
    result = evaluate_sql(
        "q",
        (
            "WITH RECURSIVE cnt(x) AS "
            "(SELECT 1 UNION ALL SELECT x + 1 FROM cnt) SELECT x FROM cnt"
        ),
        record,
        db,
        query_timeout_seconds=0,
    )
    assert result.status == "incorrect"
    assert "query exceeded" in result.details["model_sql_error"]


def test_batch_summary_removes_infrastructure_failures(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    private = tmp_path / "private.jsonl"
    predictions.write_text(
        '{"task_id":"ok","artifact":"Final answer: 2","finished":true}\n'
        '{"task_id":"infra","finished":false,"failure_kind":"infrastructure_failure"}\n',
        encoding="utf-8",
    )
    private.write_text(
        '{"task_id":"ok","domain":"math","verifier":"normalized_exact_match","reference":{"answer":"2"}}\n'
        '{"task_id":"infra","domain":"math","verifier":"normalized_exact_match","reference":{"answer":"3"}}\n',
        encoding="utf-8",
    )
    results, summary = evaluate_files(predictions, private)
    assert [result.status for result in results] == ["correct", "infrastructure_failure"]
    assert summary["accuracy"] == 1.0
    assert summary["accuracy_denominator"] == 1
    assert summary["infrastructure_failure_removed"] == 1


def test_batch_summary_counts_generation_failures_as_unsuccessful(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    private = tmp_path / "private.jsonl"
    predictions.write_text(
        '{"task_id":"ok","artifact":"Final answer: 2","finished":true}\n'
        '{"task_id":"model","finished":false,"failure_kind":"model_failure"}\n',
        encoding="utf-8",
    )
    private.write_text(
        '{"task_id":"ok","domain":"math","verifier":"normalized_exact_match",'
        '"reference":{"answer":"2"}}\n'
        '{"task_id":"model","domain":"math","verifier":"normalized_exact_match",'
        '"reference":{"answer":"3"}}\n'
        '{"task_id":"missing","domain":"math","verifier":"normalized_exact_match",'
        '"reference":{"answer":"4"}}\n',
        encoding="utf-8",
    )
    _, summary = evaluate_files(predictions, private)
    assert summary["accuracy_denominator"] == 3
    assert summary["accuracy"] == pytest.approx(1 / 3)
