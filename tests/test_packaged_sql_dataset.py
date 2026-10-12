from __future__ import annotations

import json
from pathlib import Path

from mtagentbench.batch_evaluation import read_json_or_jsonl
from mtagentbench.leakage import assert_public_safe


ROOT = Path(__file__).parents[1]


def test_committed_sql_dataset_is_public_only_and_complete() -> None:
    sql = ROOT / "datasets" / "sql"
    directory = sql / "test"
    rows = read_json_or_jsonl(directory / "public_cases.jsonl")
    assert len(rows) == 100
    assert len({row["task_id"] for row in rows}) == 100
    assert len({row["runtime_assets"]["db_id"] for row in rows}) == 11
    for row in rows:
        assert set(row) == {"domain", "runtime_assets", "schema_version", "task_id", "turns"}
        assert row["domain"] == "sql" and len(row["turns"]) == 7
        assert set(row["runtime_assets"]) == {"db_id", "schema"}
        assert row["runtime_assets"]["schema"].strip()
        assert_public_safe(row)
        serialized = json.dumps(row, ensure_ascii=False)
        for private_key in ("gold_sql", "per_turn_gold", "reference_result", "db_path"):
            assert private_key not in serialized


def test_sql_runtime_assets_are_fully_pinned() -> None:
    assets = json.loads((ROOT / "datasets" / "sql" / "runtime_assets.json").read_text(encoding="utf-8"))
    assert assets["bundle"]["bytes"] == 436442964
    assert len(assets["bundle"]["sha256"]) == 64
    assert len(assets["databases"]) == 11
    assert all(value["bytes"] > 0 and len(value["sha256"]) == 64 for value in assets["databases"].values())
    assert assets["distribution"]["assets_committed_here"] is False
