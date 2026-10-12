"""Code-specific frozen-release checks."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping, Sequence

CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def validate_release(root: Path, rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    instance_ids = [(row.get("runtime_assets") or {}).get("instance_id") for row in rows]
    if len(set(instance_ids)) != 100 or None in instance_ids:
        errors.append("code/test: SWE-bench instance IDs are not 100 unique values")
    if any(CJK.search(turn["content"]) for row in rows for turn in row["turns"]):
        errors.append("code/test: CJK remains in the English release")
    return errors
