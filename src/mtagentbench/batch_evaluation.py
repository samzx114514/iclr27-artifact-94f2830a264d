"""Batch evaluation that opens private references only after generation."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from githarness.errors import InfrastructureError, TaskEvaluationError

from .contracts import EvaluationResult, PrivateEvaluationRecord, to_plain
from .evaluators import EvaluationContext, get_evaluator


def read_json_or_jsonl(path: Path) -> list[Mapping[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        values = payload if isinstance(payload, list) else payload.get("cases") or payload.get("records")
    if not isinstance(values, list) or not all(isinstance(value, Mapping) for value in values):
        raise ValueError(f"{path} must contain JSON objects")
    return values


def private_record(value: Mapping[str, Any]) -> PrivateEvaluationRecord:
    return PrivateEvaluationRecord(
        task_id=str(value["task_id"]),
        domain=str(value["domain"]),  # type: ignore[arg-type]
        verifier=str(value["verifier"]),
        reference=dict(value.get("reference") or {}),
        asset_bindings=dict(value.get("asset_bindings") or {}),
        schema_version=int(value.get("schema_version", 1)),
    )


def load_callable(spec: str | None) -> Callable[..., Any] | None:
    if not spec:
        return None
    if ":" not in spec:
        raise ValueError("plugin must use module:function syntax")
    module_name, attribute = spec.split(":", 1)
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise TypeError(f"plugin is not callable: {spec}")
    return value


def _failure_result(prediction: Mapping[str, Any], record: PrivateEvaluationRecord) -> EvaluationResult | None:
    kind = prediction.get("failure_kind")
    if not kind and prediction.get("finished", True):
        return None
    allowed = {"protocol_error", "infrastructure_failure", "model_failure", "verifier_failure"}
    status = str(kind or "protocol_error")
    if status not in allowed:
        status = "protocol_error"
    return EvaluationResult(
        record.task_id,
        record.domain,
        status,  # type: ignore[arg-type]
        bool(prediction.get("finished", False)),
        None,
        None,
        {"error": str(prediction.get("error") or "generation did not finish")},
    )


def _artifact(prediction: Mapping[str, Any], root: Path) -> str | Mapping[str, Any]:
    if "artifact" in prediction:
        value = prediction["artifact"]
        if isinstance(value, (str, Mapping)):
            return value
    artifact_path = prediction.get("artifact_path")
    if isinstance(artifact_path, str):
        path = (root / artifact_path).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("artifact_path escapes prediction directory")
        return path.read_text(encoding="utf-8")
    raise ValueError("prediction has no artifact or artifact_path")


def evaluate_prediction(
    prediction: Mapping[str, Any],
    record: PrivateEvaluationRecord,
    *,
    prediction_root: Path,
    db_root: Path | None = None,
    research_judge: Callable[..., Any] | None = None,
    code_grader: Callable[..., Any] | None = None,
) -> EvaluationResult:
    failure = _failure_result(prediction, record)
    if failure is not None:
        return failure
    artifact = _artifact(prediction, prediction_root)
    context = EvaluationContext(
        db_root=db_root,
        research_judge=research_judge,
        code_grader=code_grader,
    )
    return get_evaluator(record.domain).evaluate(artifact, record, context)


def evaluate_files(
    predictions_path: Path,
    private_path: Path,
    *,
    db_root: Path | None = None,
    research_judge: Callable[..., Any] | None = None,
    code_grader: Callable[..., Any] | None = None,
) -> tuple[list[EvaluationResult], dict[str, Any]]:
    predictions = {str(row["task_id"]): row for row in read_json_or_jsonl(predictions_path)}
    records = [private_record(row) for row in read_json_or_jsonl(private_path)]
    results = []
    for record in records:
        prediction = predictions.get(record.task_id)
        if prediction is None:
            result = EvaluationResult(
                record.task_id,
                record.domain,
                "protocol_error",
                False,
                None,
                None,
                {"error": "missing prediction"},
            )
        else:
            try:
                result = evaluate_prediction(
                    prediction,
                    record,
                    prediction_root=predictions_path.parent,
                    db_root=db_root,
                    research_judge=research_judge,
                    code_grader=code_grader,
                )
            except InfrastructureError as error:
                result = EvaluationResult(
                    record.task_id,
                    record.domain,
                    "infrastructure_failure",
                    True,
                    None,
                    None,
                    {"error": str(error)},
                )
            except TaskEvaluationError as error:
                result = EvaluationResult(
                    record.task_id,
                    record.domain,
                    "verifier_failure",
                    True,
                    None,
                    None,
                    {"error": str(error)},
                )
        results.append(result)
    summary = summarize(results)
    return results, summary


def summarize(results: Iterable[EvaluationResult]) -> dict[str, Any]:
    rows = list(results)
    accuracy_statuses = {
        "correct",
        "incorrect",
        "protocol_error",
        "model_failure",
        "verifier_failure",
    }
    correctness_rows = [
        row for row in rows
        if row.domain != "research" and row.status in accuracy_statuses
    ]
    summary = {
        "schema_version": 1,
        "total": len(rows),
        "finished": sum(row.finished for row in rows),
        "correct": sum(row.status == "correct" for row in rows),
        "incorrect": sum(row.status == "incorrect" for row in rows),
        "scored": sum(row.status == "scored" for row in rows),
        "protocol_error": sum(row.status == "protocol_error" for row in rows),
        "model_failure": sum(row.status == "model_failure" for row in rows),
        "verifier_failure": sum(row.status == "verifier_failure" for row in rows),
        "infrastructure_failure_removed": sum(row.status == "infrastructure_failure" for row in rows),
        "accuracy_denominator": len(correctness_rows),
        "accuracy": (
            sum(row.status == "correct" for row in correctness_rows) / len(correctness_rows)
            if correctness_rows else None
        ),
    }
    research_rows = [row for row in rows if row.domain == "research"]
    completed_race = [
        row for row in research_rows
        if row.status == "scored" and row.score is not None
    ]
    if research_rows:
        dimension_scores: dict[str, float | None] = {}
        for dimension in (
            "comprehensiveness", "insight", "instruction_following", "readability",
        ):
            values = [
                row.details.get("dimension_scores", {}).get(dimension)
                for row in completed_race
            ]
            numeric = [float(value) for value in values if isinstance(value, (int, float))]
            dimension_scores[dimension] = (
                sum(numeric) / len(numeric) if len(numeric) == len(completed_race) and numeric else None
            )
        summary["research_race"] = {
            "num_completed": len(completed_race),
            "num_errors": len(research_rows) - len(completed_race),
            "mean_race": (
                sum(float(row.score) for row in completed_race) / len(completed_race)
                if completed_race else None
            ),
            "dimension_scores": dimension_scores,
        }
    return summary


def write_evaluation(output: Path, results: list[EvaluationResult], summary: Mapping[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.jsonl").write_text(
        "".join(json.dumps(to_plain(result), ensure_ascii=False) + "\n" for result in results),
        encoding="utf-8",
    )
    (output / "summary.json").write_text(json.dumps(dict(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
