"""Search-specific frozen-release checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import read_json_object


def validate_release(root: Path, rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    retrieval = read_json_object(root / "datasets/search/retrieval_assets.json")
    for key in ("corpus", "index", "embedding_model"):
        if not (retrieval.get(key) or {}).get("revision"):
            errors.append(f"search/test: {key} revision is not pinned")
    if (retrieval.get("corpus") or {}).get("expected_documents") != 100195:
        errors.append("search/test: expected corpus population is not pinned")
    if any(row.get("runtime_assets") for row in rows):
        errors.append("search/test: corpus or retrieval content leaked into public runtime assets")
    return errors
