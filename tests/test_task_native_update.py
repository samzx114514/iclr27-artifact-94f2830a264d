from __future__ import annotations

import json
from pathlib import Path

import pytest

from githarness.domains import get_domain_spec
from githarness.task_native_update import (
    CallableTaskNativeUpdateAgent,
    UpdateAgentRequest,
    UpdateAgentResult,
    UpdateExecutionFailure,
    get_update_domain_contract,
    update_result_from_dict,
    update_result_to_dict,
)
from githarness.version_interface import CandidateWorkState, ExecutionResult
from githarness.versioning import Checkpoint, WorkState


def _request(tmp_path: Path, domain: str = "math") -> UpdateAgentRequest:
    candidate = CandidateWorkState("candidate-1", tmp_path, "checkpoint-v0")
    checkpoint = Checkpoint(
        "checkpoint-v0",
        "v0",
        WorkState({"old.txt": "accepted"}),
        state0_id="state0-v0",
        origin_state0_id="state0-v0",
    )
    return UpdateAgentRequest(
        task_id="task-1",
        domain=domain,
        candidate=candidate,
        resolved_requirement="Complete public requirement",
        selected_base_checkpoint=checkpoint,
        update_hint="non-authoritative hint",
        domain_spec=get_domain_spec(domain),
        public_runtime_assets={
            "capabilities": list(get_domain_spec(domain).public_capabilities)
        },
    )


def test_callable_update_agent_modifies_only_candidate_and_returns_typed_state(tmp_path: Path) -> None:
    request = _request(tmp_path)
    base_before = request.selected_base_checkpoint.work_state

    def execute(workspace, requirement, update_hint):
        assert requirement == request.resolved_requirement
        assert update_hint == request.update_hint
        (workspace / "answer.md").write_text("Final answer: 7", encoding="utf-8")
        return ExecutionResult(
            True,
            "Final answer: 7",
            trace=({"event": "updated"},),
        )

    result = CallableTaskNativeUpdateAgent(execute).execute(request)

    assert result.finished
    assert result.artifact == "Final answer: 7"
    assert result.modified_work_state.files["answer.md"] == "Final answer: 7"
    assert request.selected_base_checkpoint.work_state == base_before
    assert result.failure is None


def test_update_result_schema_round_trip_and_digest_tampering(tmp_path: Path) -> None:
    result = UpdateAgentResult(
        finished=True,
        artifact_path="answer.md",
        artifact="Final answer: 7",
        modified_work_state=WorkState({"answer.md": "Final answer: 7"}),
        trace=({"event": "updated"},),
    )
    payload = update_result_to_dict(result)

    assert update_result_from_dict(payload) == result
    schema = json.loads(
        (Path(__file__).parents[1] / "schemas" / "task-native-update-result-v1.schema.json")
        .read_text(encoding="utf-8")
    )
    assert set(schema["required"]) == set(payload)

    payload["artifact"] = "tampered"
    with pytest.raises(ValueError, match="digest"):
        update_result_from_dict(payload)


@pytest.mark.parametrize(
    ("domain", "capabilities", "artifact"),
    [
        ("math", ("calculator",), "answer.md"),
        ("sql", (), "answer.sql"),
        ("search", ("search(query)",), "answer.json"),
        ("research", ("search(query)",), "report.md"),
        ("code", ("/testbed",), "patch.diff"),
    ],
)
def test_five_domain_update_contracts(
    domain: str, capabilities: tuple[str, ...], artifact: str
) -> None:
    contract = get_update_domain_contract(domain)
    assert contract.artifact_path == artifact
    assert contract.public_capabilities == capabilities


@pytest.mark.parametrize(
    "private_field",
    ["hidden_tests", "gold_sql", "reference_report", "gold_patch", "grader_feedback"],
)
def test_update_request_rejects_hidden_evaluator_assets(tmp_path: Path, private_field: str) -> None:
    with pytest.raises(ValueError, match="forbidden private field"):
        UpdateAgentRequest(
            task_id="task",
            domain="math",
            candidate=CandidateWorkState("candidate", tmp_path, None),
            resolved_requirement="Compute public task",
            selected_base_checkpoint=None,
            update_hint="",
            domain_spec=get_domain_spec("math"),
            public_runtime_assets={private_field: "secret-canary"},
        )


def test_update_request_public_projection_does_not_serialize_capability_objects(tmp_path: Path) -> None:
    class SearchCapability:
        def __init__(self) -> None:
            self.private_value = "secret-canary"

        def search(self, query):
            return []

    spec = get_domain_spec("search")
    request = UpdateAgentRequest(
        task_id="task",
        domain="search",
        candidate=CandidateWorkState("candidate", tmp_path, None),
        resolved_requirement="Public task",
        selected_base_checkpoint=None,
        update_hint="",
        domain_spec=spec,
        public_runtime_assets={
            "capabilities": list(spec.public_capabilities),
            "search": SearchCapability(),
        },
    )
    projection = request.public_projection()
    assert projection["public_runtime_assets"]["provided_capability_keys"] == ["search"]
    assert "secret-canary" not in json.dumps(projection)


def test_update_failure_is_typed(tmp_path: Path) -> None:
    def fail(workspace, requirement, update_hint):
        del workspace, requirement, update_hint
        return ExecutionResult(False, "", error="model did not finish")

    result = CallableTaskNativeUpdateAgent(fail).execute(_request(tmp_path))

    assert not result.finished
    assert result.failure == UpdateExecutionFailure(
        "update_failure", "model did not finish", retryable=False
    )


@pytest.mark.parametrize(
    ("domain", "forbidden_asset"),
    [
        ("math", "host_shell"),
        ("sql", "db_path"),
        ("search", "corpus_path"),
        ("research", "reference_report"),
        ("code", "docker_socket"),
    ],
)
def test_domain_update_contract_rejects_unscoped_runtime_handles(
    tmp_path: Path, domain: str, forbidden_asset: str
) -> None:
    spec = get_domain_spec(domain)
    with pytest.raises(ValueError, match="forbidden private field|unsupported"):
        UpdateAgentRequest(
            task_id="task",
            domain=domain,
            candidate=CandidateWorkState("candidate", tmp_path, None),
            resolved_requirement="Public task",
            selected_base_checkpoint=None,
            update_hint="",
            domain_spec=spec,
            public_runtime_assets={
                "capabilities": list(spec.public_capabilities),
                forbidden_asset: "secret-canary",
            },
        )


def test_update_result_rejects_hidden_canary_in_trace_or_work_state() -> None:
    with pytest.raises(ValueError, match="forbidden private field"):
        UpdateAgentResult(
            True,
            "answer.md",
            "answer",
            WorkState({"answer.md": "answer"}),
            trace=({"grader_feedback": "secret-canary"},),
        )
    with pytest.raises(ValueError, match="forbidden private field"):
        UpdateAgentResult(
            True,
            "answer.md",
            "answer",
            WorkState({"answer.md": "answer"}, {"hidden_tests": "secret-canary"}),
        )
