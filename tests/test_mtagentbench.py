from __future__ import annotations

import pytest

from githarness.errors import DataLeakageError
from mtagentbench.construction import build_dataset
from mtagentbench.contracts import to_plain
from mtagentbench.leakage import assert_public_safe


def test_math_projection_separates_answer() -> None:
    public, private, exclusions = build_dataset(
        [{"task_id": "m1", "turns": [{"query": "Compute 2 + 3"}], "label": "5"}],
        domain="math",
    )
    assert public[0].turns[0]["content"] == "Compute 2 + 3"
    assert "answer" not in str(to_plain(public[0])).lower()
    assert private[0].reference["answer"] == "5"
    assert exclusions == []


def test_leakage_scan_rejects_nested_gold() -> None:
    with pytest.raises(DataLeakageError, match="gold_sql"):
        assert_public_safe({"runtime_assets": {"gold_sql": "SELECT 1"}})


def test_invalid_rows_are_reported_as_exclusions() -> None:
    public, private, exclusions = build_dataset(
        [{"task_id": "bad", "turns": []}],
        domain="search",
    )
    assert not public and not private
    assert len(exclusions) == 1


def test_remote_swe_schema_projects_only_public_turns() -> None:
    public, private, exclusions = build_dataset(
        [
            {
                "task_id": "code-1",
                "original_id": "owner__repo-1",
                "turns": [{"turn": 1, "query": "Inspect the public issue."}],
                "swe_bench_metadata": {
                    "FAIL_TO_PASS": ["hidden-test"],
                    "PASS_TO_PASS": ["hidden-regression"],
                    "base_commit": "private-evaluator-binding",
                },
                "dataset_validation": {"valid": True},
            }
        ],
        domain="code",
    )
    rendered = str(to_plain(public[0]))
    assert public[0].turns[0]["content"] == "Inspect the public issue."
    assert "hidden-test" not in rendered and "base_commit" not in rendered
    assert private[0].asset_bindings["instance_id"] == "owner__repo-1"
    assert exclusions == []


def test_remote_swe_schema_rejects_unvalidated_case() -> None:
    public, private, exclusions = build_dataset(
        [{"task_id": "bad-code", "original_id": "owner__repo-1", "turns": [{"query": "x"}], "dataset_validation": {"valid": False}}],
        domain="code",
    )
    assert not public and not private and len(exclusions) == 1


def test_remote_swe_schema_recovers_valid_instance_id_from_task_id() -> None:
    public, private, exclusions = build_dataset(
        [{
            "task_id": "extracted-swe_bench_verified-test-django__django-10973",
            "turns": [{"query": "Fix the public issue."}],
            "dataset_validation": {"valid": True},
        }],
        domain="code",
    )
    assert exclusions == []
    assert public[0].runtime_assets["instance_id"] == "django__django-10973"
    assert private[0].asset_bindings["instance_id"] == "django__django-10973"
