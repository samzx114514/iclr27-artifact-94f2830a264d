"""SQL-specific frozen-release checks."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .common import read_json_object


def validate_release(
    root: Path,
    train: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    splits = {"train": train, "validation": validation, "test": test}
    for name, rows in splits.items():
        if len(rows) != 100:
            errors.append(f"sql/{name}: expected 100 records, found {len(rows)}")
    seen: dict[str, str] = {}
    for name, rows in splits.items():
        for row in rows:
            task_id = row["task_id"]
            if task_id in seen:
                errors.append(f"sql: task ID {task_id} overlaps {seen[task_id]} and {name}")
            else:
                seen[task_id] = name
    for row in [*train, *validation, *test]:
        assets = row.get("runtime_assets") or {}
        if not assets.get("db_id") or not assets.get("schema"):
            errors.append(f"sql: {row.get('task_id')} lacks its public database schema")
            break
    assets = read_json_object(root / "datasets/sql/runtime_assets.json")
    databases = assets.get("databases") or {}
    sha256 = re.compile(r"^[0-9a-f]{64}$")
    if (
        not isinstance(databases, Mapping)
        or len(databases) != 11
        or any(not sha256.fullmatch(str(value.get("sha256", ""))) for value in databases.values())
    ):
        errors.append("sql: the 11 database asset hashes are not fully pinned")
    return errors
