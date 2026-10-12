from __future__ import annotations

import json
from pathlib import Path

import pytest

from mtagentbench.dataset_builders.common import (
    CANONICAL_SPLITS,
    records_by_task_id,
    write_json,
    write_jsonl,
)


def test_canonical_splits_are_unique_and_public_private_are_separate() -> None:
    assert len(CANONICAL_SPLITS) == len({spec.key for spec in CANONICAL_SPLITS}) == 9
    assert all(spec.public_directory.startswith("datasets/") for spec in CANONICAL_SPLITS)
    assert all(spec.private_directory.startswith(".local/") for spec in CANONICAL_SPLITS)
    assert all(spec.private_path.endswith("/private_eval.jsonl") for spec in CANONICAL_SPLITS)


def test_shared_writers_and_task_index_are_deterministic(tmp_path: Path) -> None:
    rows = [{"task_id": "two", "value": 2}, {"task_id": "one", "value": 1}]
    output = tmp_path / "nested" / "rows.jsonl"
    write_jsonl(output, rows)
    assert [json.loads(line) for line in output.read_text().splitlines()] == rows
    metadata = tmp_path / "metadata.json"
    write_json(metadata, {"z": 2, "a": 1})
    assert list(json.loads(metadata.read_text())) == ["a", "z"]
    assert list(records_by_task_id(rows, label="fixture")) == ["two", "one"]
    with pytest.raises(ValueError, match="duplicate task ID"):
        records_by_task_id([rows[0], rows[0]], label="fixture")
