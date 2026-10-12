"""Recursive leakage checks for model-visible benchmark artifacts."""

from __future__ import annotations

from typing import Any

from githarness.errors import DataLeakageError


FORBIDDEN_PUBLIC_KEYS = {
    "answer",
    "gold",
    "gold_answer",
    "gold_sql",
    "gold_patch",
    "reference",
    "reference_result",
    "reference_report",
    "criteria",
    "criteria_id",
    "test_patch",
    "fail_to_pass",
    "pass_to_pass",
    "grader_feedback",
    "db_path",
    "api_key",
    "token",
    "cookie",
}


def _normalized(key: str) -> str:
    return key.lower().strip().replace("-", "_")


def find_forbidden_public_fields(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}"
            if _normalized(str(key)) in FORBIDDEN_PUBLIC_KEYS:
                found.append(child)
            found.extend(find_forbidden_public_fields(item, child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(find_forbidden_public_fields(item, f"{path}[{index}]"))
    return found


def assert_public_safe(value: Any) -> None:
    found = find_forbidden_public_fields(value)
    if found:
        raise DataLeakageError(f"private fields in public artifact: {', '.join(found)}")

