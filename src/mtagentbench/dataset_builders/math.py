"""Math-specific frozen-release checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


def validate_release(
    root: Path,
    train: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
) -> list[str]:
    del root
    errors: list[str] = []
    splits = {"train": train, "validation": validation, "test": test}
    for name, rows in splits.items():
        if len(rows) != 100:
            errors.append(f"math/{name}: expected 100 records, found {len(rows)}")
    seen: dict[str, str] = {}
    for name, rows in splits.items():
        for row in rows:
            task_id = row["task_id"]
            if task_id in seen:
                errors.append(f"math: task ID {task_id} overlaps {seen[task_id]} and {name}")
            else:
                seen[task_id] = name
    return errors
