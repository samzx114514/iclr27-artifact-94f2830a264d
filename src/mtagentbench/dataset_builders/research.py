"""Research-specific frozen-release checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

def validate_release(root: Path, rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    benchmark_ids = [(row.get("runtime_assets") or {}).get("benchmark_id") for row in rows]
    if len(set(benchmark_ids)) != 100 or None in benchmark_ids:
        errors.append("research/test: benchmark IDs are not 100 unique values")
    return errors
