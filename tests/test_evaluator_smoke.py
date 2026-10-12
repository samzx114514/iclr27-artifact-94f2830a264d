from __future__ import annotations

from pathlib import Path

from mtagentbench.evaluator_smoke import run_smoke


def test_all_five_evaluator_smokes_are_offline_and_correct(tmp_path: Path) -> None:
    results = run_smoke(tmp_path)
    assert [result["domain"] for result in results] == [
        "math", "sql", "search", "research", "code",
    ]
    assert [result["status"] for result in results] == [
        "correct", "correct", "correct", "scored", "correct",
    ]
    assert all(result["finished"] is True for result in results)
