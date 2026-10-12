#!/usr/bin/env python3
"""Run five offline PublicCase-to-EvaluationResult integration smokes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Mapping

from githarness.cli import run_case
from githarness.domains import get_domain_spec
from mtagentbench.batch_evaluation import evaluate_files, write_evaluation
from mtagentbench.contracts import PrivateEvaluationRecord, PublicCase, to_plain
from mtagentbench.dataset_builders.common import write_json, write_jsonl


@dataclass(frozen=True)
class GenerationFixture:
    public_case: PublicCase
    artifact: str


def _turns(domain: str) -> tuple[Mapping[str, str], ...]:
    prompts = {
        "math": "Compute 6 multiplied by 7 and provide the final answer.",
        "sql": "Return every value from the public items table.",
        "search": "Identify the synthetic institution described in the local fixture.",
        "research": "Write a short synthetic report using the supplied public brief.",
        "code": "Produce a minimal patch for the synthetic repository fixture.",
    }
    return tuple(
        {"role": "user", "content": f"{prompts[domain]} Step {index + 1} of 7."}
        for index in range(7)
    )


def generation_fixtures() -> tuple[GenerationFixture, ...]:
    """Model-visible fixtures; no evaluator reference is present here."""
    artifacts = {
        "math": "Final answer: \\boxed{42}\n",
        "sql": "SELECT value FROM items;\n",
        "search": json.dumps({"exact_answer": "Example University"}) + "\n",
        "research": "# Synthetic report\n\nThe public evidence supports the requested conclusion.\n",
        "code": (
            "diff --git a/example.py b/example.py\n"
            "--- a/example.py\n+++ b/example.py\n"
            "@@ -1 +1 @@\n-old = 1\n+new = 1\n"
        ),
    }
    return tuple(
        GenerationFixture(
            PublicCase(
                task_id=f"{domain}-e2e-smoke",
                domain=domain,  # type: ignore[arg-type]
                turns=_turns(domain),
                runtime_assets=(
                    {"db_id": "tiny", "schema": "CREATE TABLE items(value INTEGER);"}
                    if domain == "sql"
                    else {"benchmark_id": "synthetic"} if domain == "research"
                    else {"instance_id": "owner__repo-1"} if domain == "code"
                    else {}
                ),
            ),
            artifacts[domain],
        )
        for domain in ("math", "sql", "search", "research", "code")
    )


def private_fixtures() -> tuple[PrivateEvaluationRecord, ...]:
    """Evaluator-only fixtures, constructed only after all generation runs."""
    return (
        PrivateEvaluationRecord(
            "math-e2e-smoke", "math", "normalized_exact_match", {"answer": "42"},
        ),
        PrivateEvaluationRecord(
            "sql-e2e-smoke", "sql", "bird_execution",
            {"gold_sql": "SELECT value FROM items"}, {"db_id": "tiny"},
        ),
        PrivateEvaluationRecord(
            "search-e2e-smoke", "search", "browsecomp_normalized_em",
            {"answer": "The Example University"},
        ),
        PrivateEvaluationRecord(
            "research-e2e-smoke", "research", "deep_research_race",
            _race_reference(), {"benchmark_id": "synthetic"},
        ),
        PrivateEvaluationRecord(
            "code-e2e-smoke", "code", "swebench_official", {},
            {"instance_id": "owner__repo-1"},
        ),
    )


def _scripted_turns(case: PublicCase) -> list[dict[str, Any]]:
    resolved: list[str] = []
    result = []
    for index, turn in enumerate(case.turns):
        resolved.append(turn["content"])
        result.append({
            "feedback": turn["content"],
            "actions": [{
                "action": "Commit",
                "resolved_requirement": "\n".join(resolved),
                "base_version_id": None if index == 0 else f"v{index - 1}",
                "update_hint": "Apply the latest public user turn.",
            }],
        })
    return result


def _research_judge(
    prompt: str, record: PrivateEvaluationRecord,
) -> Mapping[str, Any]:
    assert "# Synthetic report" in prompt
    return {
        dimension: [{
            "criterion": f"{dimension} criterion",
            "analysis": "The synthetic target satisfies the criterion.",
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
        "prompt": "Write a short synthetic report using the supplied public brief.",
        "reference_article": "# Synthetic reference\n\nA reference report.",
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
        "resolved": instance_id == "owner__repo-1" and "diff --git" in patch,
        "mock": True,
    }


def _database(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(root / "tiny.sqlite")
    connection.executescript(
        "CREATE TABLE items(value INTEGER); INSERT INTO items VALUES (1), (2);"
    )
    connection.commit()
    connection.close()


def run_e2e_smoke(output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"smoke output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    public_root = output / "public"
    generation_root = output / "generation"
    fixtures = generation_fixtures()
    write_jsonl(public_root / "public_cases.jsonl", [fixture.public_case for fixture in fixtures])

    predictions: list[dict[str, Any]] = []
    generation_reports: list[dict[str, Any]] = []
    for fixture in fixtures:
        case = fixture.public_case
        domain = case.domain
        spec = get_domain_spec(domain)
        config = {
            "task_id": case.task_id,
            "domain": domain,
            "executor": "offline_fixture",
            "synthetic_fixture_only": True,
            "fixture_artifact": {"path": spec.artifact_path, "content": fixture.artifact},
            "turns": _scripted_turns(case),
        }
        config_path = public_root / "configs" / f"{domain}.json"
        write_json(config_path, config)
        run_root = generation_root / domain
        run = run_case(config_path, run_root)
        if len(run["turns"]) != 7:
            raise RuntimeError(f"{domain}: generation did not process seven turns")
        final = run["turns"][-1]
        artifact_path = (
            run_root / "workspace" / "snapshots" / final["checkpoint_id"]
            / spec.artifact_path
        )
        if not artifact_path.is_file():
            raise RuntimeError(f"{domain}: final artifact is missing")
        predictions.append({
            "task_id": case.task_id,
            "finished": True,
            "artifact_path": artifact_path.relative_to(output).as_posix(),
        })
        generation_reports.append({
            "task_id": case.task_id,
            "domain": domain,
            "turns": len(run["turns"]),
            "modes": [turn["mode"] for turn in run["turns"]],
            "artifact": artifact_path.relative_to(output).as_posix(),
        })

    # Private records and evaluator assets are created only after generation.
    evaluator_root = output / "evaluator"
    private_path = evaluator_root / "private_eval.jsonl"
    write_jsonl(private_path, private_fixtures())
    predictions_path = output / "predictions.jsonl"
    write_jsonl(predictions_path, predictions)
    db_root = evaluator_root / "databases"
    _database(db_root)
    results, summary = evaluate_files(
        predictions_path,
        private_path,
        db_root=db_root,
        research_judge=_research_judge,
        code_grader=_code_grader,
    )
    write_evaluation(output / "evaluation", results, summary)
    report = {
        "schema_version": 1,
        "valid": (
            len(results) == 5
            and all(
                result.status == ("scored" if result.domain == "research" else "correct")
                for result in results
            )
        ),
        "public_cases": len(fixtures),
        "private_records": len(results),
        "generation": generation_reports,
        "evaluation_results": [to_plain(result) for result in results],
        "summary": summary,
        "private_visible_to_generation": False,
        "external_services_called": False,
        "docker_used": False,
    }
    write_json(output / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output is not None:
        report = run_e2e_smoke(args.output.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="githarness-dataset-e2e-") as value:
            report = run_e2e_smoke(Path(value))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
