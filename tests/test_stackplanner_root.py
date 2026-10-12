from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from githarness.integrations.stackplanner.root import CanonicalStackPlannerRootExecutor
from githarness.state0 import State0Request
from githarness.version_interface import CandidateWorkState


class FakeGraph:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def ainvoke(self, state, config):
        self.calls.append((state, config))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_root_module_keeps_stackplanner_dependencies_lazy() -> None:
    repository = Path(__file__).parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import githarness.integrations.stackplanner.root; "
                "assert not any(name == 'langgraph' or name.startswith('langgraph.') "
                "for name in sys.modules)"
            ),
        ],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("domain", "graph_format", "state", "artifact_path", "artifact"),
    [
        ("math", "sp_math", {"final_report": "Final answer: 5"}, "answer.md", "Final answer: 5"),
        ("sql", "sp_sql", {"final_report": "```sql\nSELECT 1;\n```"}, "answer.sql", "SELECT 1"),
        ("search", "sp_search", {"search_exact_answer": "Ada"}, "answer.json", '{"exact_answer": "Ada"}'),
        ("research", "sp_deep_research", {"final_report": "# Report"}, "report.md", "# Report"),
        ("code", "sp_swe", {"swe_patch": "diff --git a/a b/a\n"}, "patch.diff", "diff --git a/a b/a\n"),
    ],
)
def test_canonical_root_supports_five_domains(
    tmp_path, domain, graph_format, state, artifact_path, artifact
) -> None:
    state = {
        **state,
        "current_plan": {"steps": ["public step"]},
    }
    graph = FakeGraph(state)
    observed = []

    def factory(selected, vendor_root):
        observed.append(selected)
        return graph

    executor = CanonicalStackPlannerRootExecutor(domain=domain, graph_factory=factory)
    candidate = CandidateWorkState("candidate-1", tmp_path, None)
    request = State0Request("task", domain, "complete public requirement")
    result = executor.create_state0(candidate, request)

    assert result.finished is True
    assert observed == [graph_format]
    assert result.artifact_path == artifact_path
    assert result.artifact == artifact
    assert (tmp_path / artifact_path).read_text(encoding="utf-8") == artifact
    assert result.root_plan["plan"] == {"steps": ["public step"]}
    prompt_state, config = graph.calls[0]
    assert prompt_state["user_query"] == "complete public requirement"
    assert config["configurable"]["graph_format"] == graph_format
    assert "secret-canary" not in json.dumps(result.trace)


def test_root_failure_is_typed_and_writes_no_artifact(tmp_path) -> None:
    executor = CanonicalStackPlannerRootExecutor(
        domain="math",
        graph_factory=lambda *_: FakeGraph(RuntimeError("provider unavailable")),
    )
    result = executor.create_state0(
        CandidateWorkState("candidate-1", tmp_path, None),
        State0Request("task", "math", "Compute the public task"),
    )
    assert result.finished is False
    assert result.failure is not None
    assert result.failure.kind == "infrastructure_failure"
    assert not (tmp_path / "answer.md").exists()


def test_root_configuration_rejects_hidden_evaluator_fields() -> None:
    with pytest.raises(ValueError, match="forbidden private field"):
        CanonicalStackPlannerRootExecutor(
            domain="math",
            configurable={"grader_feedback": "secret-canary"},
        )
