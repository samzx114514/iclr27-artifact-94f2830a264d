from __future__ import annotations

import json
from pathlib import Path

import pytest

from githarness.cli import run_case
from mtagentbench.batch_evaluation import read_json_or_jsonl
from mtagentbench.dataset_builders.common import write_json
from mtagentbench.leakage import assert_public_safe
from mtagentbench.e2e_smoke import generation_fixtures, run_e2e_smoke


def test_generation_fixtures_are_five_public_seven_turn_cases() -> None:
    fixtures = generation_fixtures()
    assert [fixture.public_case.domain for fixture in fixtures] == [
        "math", "sql", "search", "research", "code",
    ]
    for fixture in fixtures:
        assert len(fixture.public_case.turns) == 7
        assert_public_safe({
            "task_id": fixture.public_case.task_id,
            "domain": fixture.public_case.domain,
            "turns": list(fixture.public_case.turns),
            "runtime_assets": dict(fixture.public_case.runtime_assets),
        })


def test_offline_fixture_executor_is_restricted_to_synthetic_configs(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.json"
    write_json(config, {
        "task_id": "unsafe",
        "domain": "math",
        "executor": "offline_fixture",
        "fixture_artifact": {"path": "answer.md", "content": "Answer: 1"},
        "turns": [],
    })
    with pytest.raises(SystemExit, match="synthetic_fixture_only=true"):
        run_case(config, tmp_path / "run")


def test_five_domain_publiccase_to_evaluationresult_smoke(tmp_path: Path) -> None:
    report = run_e2e_smoke(tmp_path)
    assert report["valid"] is True
    assert report["private_visible_to_generation"] is False
    assert report["external_services_called"] is False
    assert report["docker_used"] is False
    assert report["public_cases"] == report["private_records"] == 5
    assert report["summary"]["correct"] == 4
    assert report["summary"]["scored"] == 1
    assert report["summary"]["accuracy"] == 1.0
    assert report["summary"]["research_race"]["mean_race"] == 0.5
    assert [row["turns"] for row in report["generation"]] == [7] * 5
    assert all(
        row["modes"] == ["fresh_solve", *("reuse_and_patch" for _ in range(6))]
        for row in report["generation"]
    )
    assert [row["status"] for row in report["evaluation_results"]] == [
        "correct", "correct", "correct", "scored", "correct",
    ]
    assert [Path(row["artifact"]).name for row in report["generation"]] == [
        "answer.md", "answer.sql", "answer.json", "report.md", "patch.diff",
    ]

    public_rows = read_json_or_jsonl(tmp_path / "public/public_cases.jsonl")
    private_rows = read_json_or_jsonl(tmp_path / "evaluator/private_eval.jsonl")
    assert [row["task_id"] for row in public_rows] == [
        row["task_id"] for row in private_rows
    ]
    public_text = (tmp_path / "public/public_cases.jsonl").read_text(encoding="utf-8")
    assert '"reference"' not in public_text
    assert '"verifier"' not in public_text
    assert len(read_json_or_jsonl(tmp_path / "evaluation/results.jsonl")) == 5
    assert json.loads((tmp_path / "report.json").read_text())["valid"] is True
