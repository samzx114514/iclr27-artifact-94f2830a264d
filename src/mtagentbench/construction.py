"""Five-domain projection into isolated public and private records."""

from __future__ import annotations

import re
from typing import Any, Mapping

from githarness.errors import DatasetConstructionError

from .contracts import PrivateEvaluationRecord, PublicCase, to_plain
from .leakage import assert_public_safe


def _turns(values: Any, *, fields: tuple[str, ...] = ("content", "query", "query_zh")) -> tuple[dict[str, str], ...]:
    if not isinstance(values, list) or not values:
        raise DatasetConstructionError("source row requires public turns")
    turns = []
    for raw in values:
        if isinstance(raw, str):
            content = raw
        elif isinstance(raw, Mapping):
            content = next((raw[name] for name in fields if isinstance(raw.get(name), str)), None)
        else:
            content = None
        if not isinstance(content, str) or not content.strip():
            raise DatasetConstructionError("source row contains an invalid public turn")
        turns.append({"role": "user", "content": content.strip()})
    return tuple(turns)


def _task_id(row: Mapping[str, Any]) -> str:
    value = row.get("task_id") or row.get("instance_id") or row.get("id")
    if not isinstance(value, (str, int)) or not str(value).strip():
        raise DatasetConstructionError("source row has no task identifier")
    return str(value)


def project_math(row: Mapping[str, Any]) -> tuple[PublicCase, PrivateEvaluationRecord]:
    task_id = _task_id(row)
    turns = _turns(row.get("turns") or [{"content": row.get("question", "")}])
    answer = row.get("label") or (row.get("source") or {}).get("label") or (row.get("metadata") or {}).get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise DatasetConstructionError(f"{task_id}: Math source lacks private answer")
    return (
        PublicCase(task_id, "math", turns),
        PrivateEvaluationRecord(task_id, "math", "normalized_exact_match", {"answer": answer}),
    )


def project_sql(row: Mapping[str, Any]) -> tuple[PublicCase, PrivateEvaluationRecord]:
    task_id = _task_id(row)
    metadata = row.get("metadata") or {}
    turns = _turns(row.get("turns") or [{"content": row.get("question", "")}])
    schema = metadata.get("schema") or row.get("schema")
    db_id = metadata.get("db_id") or row.get("db_id")
    gold_sql = metadata.get("gold_sql") or row.get("gold_sql")
    if not isinstance(schema, str) or not schema.strip():
        raise DatasetConstructionError(f"{task_id}: SQL source lacks complete public schema")
    if not isinstance(db_id, str) or not isinstance(gold_sql, str):
        raise DatasetConstructionError(f"{task_id}: SQL source lacks private evaluator fields")
    return (
        PublicCase(task_id, "sql", turns, {"db_id": db_id, "schema": schema}),
        PrivateEvaluationRecord(task_id, "sql", "bird_execution", {"gold_sql": gold_sql}, {"db_id": db_id}),
    )


def project_search(row: Mapping[str, Any]) -> tuple[PublicCase, PrivateEvaluationRecord]:
    task_id = _task_id(row)
    turns = _turns(row.get("turns") or [{"content": row.get("source_question", row.get("question", ""))}])
    answer = row.get("source_answer") or row.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise DatasetConstructionError(f"{task_id}: Search source lacks private answer")
    return (
        PublicCase(task_id, "search", turns),
        PrivateEvaluationRecord(task_id, "search", "browsecomp_normalized_em", {"answer": answer}),
    )


def project_research(row: Mapping[str, Any]) -> tuple[PublicCase, PrivateEvaluationRecord]:
    task_id = _task_id(row)
    if not row.get("turns") and (row.get("evolution_v2") or row.get("manual_turns")):
        raise DatasetConstructionError(
            f"{task_id}: versioned Research construction source requires "
            "the domain-specific Research preparation path; generic projection would collapse it to one turn"
        )
    canonical = row.get("canonical_prompt") or row.get("prompt")
    turns = _turns(row.get("turns") or [{"content": canonical or ""}])
    benchmark_id = row.get("benchmark_id") or row.get("id")
    if benchmark_id is None:
        raise DatasetConstructionError(f"{task_id}: Research source lacks benchmark id")
    return (
        PublicCase(task_id, "research", turns, {"benchmark_id": benchmark_id}),
        PrivateEvaluationRecord(
            task_id,
            "research",
            "deep_research_race",
            {"canonical_prompt": canonical},
            {"benchmark_id": benchmark_id},
        ),
    )


def project_code(row: Mapping[str, Any]) -> tuple[PublicCase, PrivateEvaluationRecord]:
    task_id = _task_id(row)
    metadata = row.get("swe_bench_metadata") or row.get("metadata") or {}
    validation = row.get("dataset_validation")
    if isinstance(validation, Mapping) and validation.get("valid") is not True:
        raise DatasetConstructionError(f"{task_id}: Code source did not pass dataset validation")
    instance_id = row.get("original_id") or row.get("instance_id") or metadata.get("original_id")
    if not instance_id:
        prefix = "extracted-swe_bench_verified-test-"
        encoded = task_id.removeprefix(prefix) if task_id.startswith(prefix) else ""
        if re.fullmatch(r"[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-\d+", encoded):
            instance_id = encoded
    turns = _turns(row.get("turns") or row.get("user_messages") or [{"content": row.get("problem_statement", "")}])
    if not isinstance(instance_id, str) or not instance_id.strip():
        raise DatasetConstructionError(f"{task_id}: Code source lacks instance id")
    return (
        PublicCase(task_id, "code", turns, {"instance_id": instance_id}),
        PrivateEvaluationRecord(task_id, "code", "swebench_official", {}, {"instance_id": instance_id}),
    )


PROJECTORS = {
    "math": project_math,
    "sql": project_sql,
    "search": project_search,
    "research": project_research,
    "code": project_code,
}


def build_dataset(
    rows: list[Mapping[str, Any]],
    *,
    domain: str,
) -> tuple[list[PublicCase], list[PrivateEvaluationRecord], list[dict[str, Any]]]:
    try:
        projector = PROJECTORS[domain]
    except KeyError as error:
        raise DatasetConstructionError(f"unsupported domain: {domain}") from error
    public: list[PublicCase] = []
    private: list[PrivateEvaluationRecord] = []
    exclusions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        try:
            pub, priv = projector(row)
            if pub.task_id in seen:
                raise DatasetConstructionError(f"duplicate task_id: {pub.task_id}")
            seen.add(pub.task_id)
            assert_public_safe(to_plain(pub))
            public.append(pub)
            private.append(priv)
        except (DatasetConstructionError, ValueError) as error:
            exclusions.append({"index": index, "reason": str(error)})
    return public, private, exclusions
