"""Read-only SQLite execution-equivalence evaluator."""

from __future__ import annotations

from pathlib import Path
import re
import sqlite3
import time
from typing import Any

from githarness.errors import InfrastructureError, TaskEvaluationError

from ..contracts import EvaluationResult, PrivateEvaluationRecord
from ..sql_assets import find_database
from .base import Artifact, EvaluationContext, require_record


SQL_QUERY_TIMEOUT_SECONDS = 30.0


class SQLQueryTimeout(sqlite3.OperationalError):
    """Raised when SQLite exceeds the evaluator's wall-clock query budget."""


def extract_sql(text: str) -> str | None:
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.I | re.S)
    value = fenced.group(1).strip() if fenced else text.strip()
    match = re.search(r"((?:WITH\s+.+?\)\s+)?SELECT\s+.+?)(?:;|\Z)", value, re.I | re.S)
    if not match:
        return None
    return " ".join(match.group(1).split()).rstrip(";")


def _query(
    db_path: Path,
    sql: str,
    *,
    timeout_seconds: float = SQL_QUERY_TIMEOUT_SECONDS,
) -> list[tuple[Any, ...]]:
    if not db_path.is_file():
        raise InfrastructureError(f"SQLite database not found: {db_path}")
    if timeout_seconds < 0:
        raise ValueError("timeout_seconds must be non-negative")
    connection = sqlite3.connect(
        f"file:{db_path}?mode=ro",
        uri=True,
        timeout=min(timeout_seconds, 10.0),
    )
    deadline = time.monotonic() + timeout_seconds
    timed_out = False

    def interrupt_after_deadline() -> int:
        nonlocal timed_out
        timed_out = time.monotonic() >= deadline
        return int(timed_out)

    connection.set_progress_handler(interrupt_after_deadline, 1_000)
    try:
        try:
            return connection.execute(sql).fetchall()
        except sqlite3.OperationalError as error:
            if timed_out and "interrupted" in str(error).lower():
                raise SQLQueryTimeout(
                    f"query exceeded {timeout_seconds:g} seconds"
                ) from error
            raise
    finally:
        connection.set_progress_handler(None, 0)
        connection.close()


def evaluate_sql(
    task_id: str,
    artifact: str,
    record: PrivateEvaluationRecord,
    db_path: Path,
    *,
    query_timeout_seconds: float = SQL_QUERY_TIMEOUT_SECONDS,
) -> EvaluationResult:
    predicted = extract_sql(artifact)
    gold = record.reference.get("gold_sql")
    if not isinstance(gold, str):
        raise TaskEvaluationError("private SQL record has no gold_sql")
    if predicted is None:
        return EvaluationResult(
            task_id, "sql", "incorrect", True, False, 0.0,
            {"error": "no SQL extracted"},
        )
    try:
        gold_rows = _query(db_path, gold, timeout_seconds=query_timeout_seconds)
    except SQLQueryTimeout as error:
        raise InfrastructureError(f"gold SQL timed out: {error}") from error
    except sqlite3.Error as error:
        raise InfrastructureError(f"gold SQL failed: {error}") from error
    try:
        predicted_rows = _query(
            db_path,
            predicted,
            timeout_seconds=query_timeout_seconds,
        )
    except SQLQueryTimeout as error:
        return EvaluationResult(
            task_id, "sql", "incorrect", True, False, 0.0,
            {"model_sql_error": str(error)},
        )
    except sqlite3.Error as error:
        return EvaluationResult(
            task_id, "sql", "incorrect", True, False, 0.0,
            {"model_sql_error": str(error)},
        )
    correct = set(predicted_rows) == set(gold_rows)
    return EvaluationResult(
        task_id, "sql", "correct" if correct else "incorrect", True,
        correct, float(correct), {"rows": len(predicted_rows)},
    )


class SQLEvaluator:
    domain = "sql"
    verifier = "bird_execution"

    def evaluate(
        self, artifact: Artifact, record: PrivateEvaluationRecord,
        context: EvaluationContext,
    ) -> EvaluationResult:
        require_record(record, domain=self.domain, verifier=self.verifier)
        if context.db_root is None:
            raise InfrastructureError("SQL evaluation requires --db-root")
        db_id = record.asset_bindings.get("db_id")
        if not isinstance(db_id, str):
            raise TaskEvaluationError("private SQL record has no db_id")
        try:
            db_path = find_database(context.db_root, db_id)
        except FileNotFoundError as error:
            raise InfrastructureError(
                f"database asset is missing for db_id={db_id}"
            ) from error
        return evaluate_sql(
            record.task_id, str(artifact), record, db_path,
        )
