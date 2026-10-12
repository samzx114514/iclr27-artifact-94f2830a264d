from __future__ import annotations

import pytest

from githarness.domains import get_domain_spec
from githarness.integrations.stackplanner import StackPlannerTaskNativeUpdateAgent
from githarness.task_native_update import UpdateAgentRequest
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


def _request(tmp_path, domain):
    spec = get_domain_spec(domain)
    return UpdateAgentRequest(
        task_id="task",
        domain=domain,
        candidate=CandidateWorkState(
            "candidate-1",
            tmp_path,
            "checkpoint-v0",
            {"state0_root_work_state": {"memory_stack": ["accepted public state"]}},
        ),
        resolved_requirement="Complete revised public requirement",
        selected_base_checkpoint=None,
        update_hint="change one public slot",
        domain_spec=spec,
        public_runtime_assets={"capabilities": list(spec.public_capabilities)},
    )


@pytest.mark.parametrize(
    ("domain", "graph_format", "state", "artifact_path", "artifact_fragment"),
    [
        ("math", "sp_math", {"final_report": "Final answer: 7"}, "answer.md", "7"),
        ("sql", "sp_sql", {"final_report": "```sql\nSELECT title FROM books\n```"}, "answer.sql", "SELECT"),
        ("search", "sp_search", {"search_exact_answer": "1997"}, "answer.json", "1997"),
        ("research", "sp_deep_research", {"final_report": "# Public report"}, "report.md", "Public report"),
        ("code", "sp_swe", {"swe_patch": "diff --git a/a b/a\n"}, "patch.diff", "diff --git"),
    ],
)
def test_stackplanner_update_supports_five_domains_with_one_typed_boundary(
    tmp_path, domain, graph_format, state, artifact_path, artifact_fragment
) -> None:
    graph = FakeGraph(state)
    agent = StackPlannerTaskNativeUpdateAgent(
        domain=domain,
        graph_factory=lambda selected, vendor: graph,
    )

    result = agent.execute(_request(tmp_path, domain))

    assert result.finished
    assert result.artifact_path == artifact_path
    assert artifact_fragment in result.artifact
    assert result.modified_work_state.files[artifact_path] == result.artifact
    state_input, config = graph.calls[0]
    assert state_input["memory_stack"] == ["accepted public state"]
    assert "Non-authoritative update hint" in state_input["user_query"]
    assert config["configurable"]["graph_format"] == graph_format


def test_stackplanner_update_persists_projected_native_work_state(tmp_path) -> None:
    graph = FakeGraph({"final_report": "Final answer: 9", "memory_stack": ["updated"]})
    request = _request(tmp_path, "math")
    result = StackPlannerTaskNativeUpdateAgent(
        domain="math", graph_factory=lambda *_: graph
    ).execute(request)

    assert result.modified_work_state.metadata["task_native_work_state"] == {
        "memory_stack": ["updated"],
        "final_report": "Final answer: 9",
    }


def test_stackplanner_update_failure_is_typed_and_redacted(tmp_path) -> None:
    result = StackPlannerTaskNativeUpdateAgent(
        domain="math",
        graph_factory=lambda *_: FakeGraph(RuntimeError("private endpoint secret-canary")),
    ).execute(_request(tmp_path, "math"))

    assert not result.finished
    assert result.failure is not None
    assert result.failure.kind == "infrastructure_failure"
    assert "secret-canary" not in result.failure.message


def test_stackplanner_update_rejects_hidden_configuration() -> None:
    with pytest.raises(ValueError, match="forbidden private field"):
        StackPlannerTaskNativeUpdateAgent(
            domain="search",
            configurable={"grader_feedback": "secret-canary"},
        )
