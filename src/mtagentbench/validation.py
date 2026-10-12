"""Deterministic validation for paired MTAgentBench release artifacts."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from githarness.integrations.stackplanner.swe import image_name_for_instance
from githarness.errors import DataLeakageError

from .batch_evaluation import read_json_or_jsonl
from .contracts import PrivateEvaluationRecord, PublicCase
from .leakage import assert_public_safe


_PUBLIC_KEYS = {"task_id", "domain", "turns", "runtime_assets", "schema_version"}
_PRIVATE_KEYS = {"task_id", "domain", "verifier", "reference", "asset_bindings", "schema_version"}
_DOMAINS = {"math", "sql", "search", "research", "code"}


def _public(value: Mapping[str, Any]) -> PublicCase:
    if not isinstance(value, Mapping):
        raise ValueError("public row must be an object")
    unknown = set(value) - _PUBLIC_KEYS
    if unknown:
        raise ValueError(f"unknown public fields: {sorted(unknown)}")
    if int(value.get("schema_version", 0)) != 1:
        raise ValueError("unsupported public schema_version")
    if value.get("domain") not in _DOMAINS:
        raise ValueError("unsupported public domain")
    assert_public_safe(dict(value))
    return PublicCase(
        task_id=str(value.get("task_id", "")),
        domain=str(value.get("domain", "")),  # type: ignore[arg-type]
        turns=tuple(value.get("turns") or ()),
        runtime_assets=dict(value.get("runtime_assets") or {}),
        schema_version=1,
    )


def _private(value: Mapping[str, Any]) -> PrivateEvaluationRecord:
    if not isinstance(value, Mapping):
        raise ValueError("private row must be an object")
    unknown = set(value) - _PRIVATE_KEYS
    if unknown:
        raise ValueError(f"unknown private fields: {sorted(unknown)}")
    if int(value.get("schema_version", 0)) != 1:
        raise ValueError("unsupported private schema_version")
    if value.get("domain") not in _DOMAINS:
        raise ValueError("unsupported private domain")
    return PrivateEvaluationRecord(
        task_id=str(value.get("task_id", "")),
        domain=str(value.get("domain", "")),  # type: ignore[arg-type]
        verifier=str(value.get("verifier", "")),
        reference=dict(value.get("reference") or {}),
        asset_bindings=dict(value.get("asset_bindings") or {}),
        schema_version=1,
    )


def _duplicates(values: list[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def _db_path(root: Path, db_id: str) -> Path | None:
    candidates = (
        root / db_id / f"{db_id}.sqlite",
        root / f"{db_id}.sqlite",
        root / f"{db_id}.db",
    )
    return next((path for path in candidates if path.is_file()), None)


def validate_search_corpus(path: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    seen: set[str] = set()
    count = 0
    try:
        rows = read_json_or_jsonl(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return 0, [f"search corpus is invalid: {error}"]
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"search corpus row {index} must be an object")
            count += 1
            continue
        docid = row.get("docid")
        text = row.get("text")
        if not isinstance(docid, str) or not docid.strip():
            errors.append(f"search corpus row {index} has no docid")
        elif docid in seen:
            errors.append(f"duplicate search docid: {docid}")
        else:
            seen.add(docid)
        if not isinstance(text, str) or not text.strip():
            errors.append(f"search corpus row {index} has no text")
        count += 1
    if not count:
        errors.append("search corpus is empty")
    return count, errors


def validate_dataset(
    public_path: Path,
    private_path: Path,
    *,
    db_root: Path | None = None,
    search_corpus: Path | None = None,
    check_swe_images: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    raw_public = read_json_or_jsonl(public_path)
    raw_private = read_json_or_jsonl(private_path)
    public: list[PublicCase] = []
    private: list[PrivateEvaluationRecord] = []
    for index, value in enumerate(raw_public):
        try:
            public.append(_public(value))
        except (DataLeakageError, TypeError, ValueError) as error:
            errors.append(f"public row {index}: {error}")
    for index, value in enumerate(raw_private):
        try:
            private.append(_private(value))
        except (TypeError, ValueError) as error:
            errors.append(f"private row {index}: {error}")

    public_ids = [record.task_id for record in public]
    private_ids = [record.task_id for record in private]
    for label, values in (("public", public_ids), ("private", private_ids)):
        duplicates = _duplicates(values)
        if duplicates:
            errors.append(f"duplicate {label} task IDs: {duplicates}")
    if public_ids != private_ids:
        errors.append("public/private task IDs or ordering differ")
    for pub, priv in zip(public, private):
        if pub.task_id != priv.task_id or pub.domain != priv.domain:
            errors.append(f"paired record mismatch: {pub.task_id}/{priv.task_id}")

    sql_records = [(pub, priv) for pub, priv in zip(public, private) if pub.domain == "sql"]
    if sql_records and db_root is None:
        warnings.append("SQL database assets were not checked; pass --db-root")
    if db_root is not None:
        for pub, priv in sql_records:
            db_id = priv.asset_bindings.get("db_id") or pub.runtime_assets.get("db_id")
            if not isinstance(db_id, str) or _db_path(db_root, db_id) is None:
                errors.append(f"{pub.task_id}: SQL database asset is missing")

    if any(record.domain == "search" for record in public):
        if search_corpus is None:
            warnings.append("Search corpus was not checked; pass --search-corpus")
        else:
            _, corpus_errors = validate_search_corpus(search_corpus)
            errors.extend(corpus_errors)

    code_records = [(pub, priv) for pub, priv in zip(public, private) if pub.domain == "code"]
    if code_records and not check_swe_images:
        warnings.append("SWE-bench Docker images were not checked; pass --check-swe-images")
    if check_swe_images:
        for pub, priv in code_records:
            instance_id = priv.asset_bindings.get("instance_id")
            if not isinstance(instance_id, str):
                errors.append(f"{pub.task_id}: Code instance_id is missing")
                continue
            image = image_name_for_instance(instance_id)
            try:
                result = subprocess.run(
                    ["docker", "image", "inspect", image],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                errors.append(f"Docker image check failed: {error}")
                break
            if result.returncode:
                errors.append(f"{pub.task_id}: local SWE-bench image is missing: {image}")

    return {
        "schema_version": 1,
        "valid": not errors,
        "public_records": len(public),
        "private_records": len(private),
        "domains": dict(sorted(Counter(record.domain for record in public).items())),
        "errors": errors,
        "warnings": warnings,
    }
