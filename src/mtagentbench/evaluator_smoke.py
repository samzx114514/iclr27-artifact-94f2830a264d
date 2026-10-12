#!/usr/bin/env python3
"""Run one offline synthetic smoke through each evaluator adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Mapping

from mtagentbench.contracts import PrivateEvaluationRecord, to_plain
from mtagentbench.evaluators import EvaluationContext, get_evaluator


def _research_judge(
    prompt: str, record: PrivateEvaluationRecord,
) -> Mapping[str, Any]:
    assert "A synthetic research report." in prompt
    return {
        dimension: [{
            "criterion": f"{dimension} criterion",
            "analysis": "The synthetic target and reference are comparable.",
            "article_1_score": 8,
            "article_2_score": 8,
        }]
        for dimension in _RACE_DIMENSIONS
    }


_RACE_DIMENSIONS = (
    "comprehensiveness", "insight", "instruction_following", "readability",
)


def _race_reference() -> Mapping[str, Any]:
    return {
        "prompt": "Write a short synthetic report.",
        "reference_article": "A synthetic reference report.",
        "language": "en",
        "criteria": {
            "dimension_weight": {dimension: 0.25 for dimension in _RACE_DIMENSIONS},
            "criterions": {
                dimension: [{
                    "criterion": f"{dimension} criterion",
                    "explanation": f"Evaluate {dimension}.",
                    "weight": 1.0,
                }]
                for dimension in _RACE_DIMENSIONS
            },
        },
    }


def _code_grader(instance_id: str, patch: str) -> Mapping[str, Any]:
    return {
        "resolved": instance_id == "owner__repo-1" and patch.startswith("diff --git"),
        "mock": True,
    }


def run_smoke(workspace: Path) -> list[dict[str, Any]]:
    workspace.mkdir(parents=True, exist_ok=True)
    database = workspace / "tiny.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript(
        "CREATE TABLE items(value INTEGER); INSERT INTO items VALUES (1), (2);"
    )
    connection.commit()
    connection.close()
    context = EvaluationContext(
        db_root=workspace,
        research_judge=_research_judge,
        code_grader=_code_grader,
    )
    fixtures = (
        (
            PrivateEvaluationRecord(
                "math-smoke", "math", "normalized_exact_match", {"answer": "42"},
            ),
            r"Final answer: \boxed{42}",
        ),
        (
            PrivateEvaluationRecord(
                "sql-smoke", "sql", "bird_execution",
                {"gold_sql": "SELECT value FROM items"}, {"db_id": "tiny"},
            ),
            "```sql\nSELECT value FROM items;\n```",
        ),
        (
            PrivateEvaluationRecord(
                "search-smoke", "search", "browsecomp_normalized_em",
                {"answer": "The Example University"},
            ),
            {"exact_answer": "Example University"},
        ),
        (
            PrivateEvaluationRecord(
                "research-smoke", "research", "deep_research_race",
                _race_reference(), {"benchmark_id": "synthetic"},
            ),
            "A synthetic research report.",
        ),
        (
            PrivateEvaluationRecord(
                "code-smoke", "code", "swebench_official", {},
                {"instance_id": "owner__repo-1"},
            ),
            "diff --git a/a.py b/a.py\n",
        ),
    )
    return [
        to_plain(get_evaluator(record.domain).evaluate(artifact, record, context))
        for record, artifact in fixtures
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args(argv)
    if args.workspace is not None:
        results = run_smoke(args.workspace)
    else:
        with tempfile.TemporaryDirectory(prefix="githarness-evaluator-smoke-") as value:
            results = run_smoke(Path(value))
    report = {
        "schema_version": 1,
        "valid": all(
            result["status"] == ("scored" if result["domain"] == "research" else "correct")
            for result in results
        ),
        "domains": [result["domain"] for result in results],
        "results": results,
        "external_services_called": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
