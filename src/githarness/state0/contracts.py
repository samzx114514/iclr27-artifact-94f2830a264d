"""Typed public contracts for canonical StackPlanner State 0."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping

from githarness.versioning.models import RequirementState, WorkState


STATE0_SCHEMA_VERSION = 1
FailureKind = Literal[
    "protocol_error",
    "model_failure",
    "root_failure",
    "finalization_failure",
    "serialization_failure",
    "checkpoint_failure",
    "infrastructure_failure",
]

_FORBIDDEN_KEYS = {
    "answer_key",
    "api_key",
    "credential",
    "docker_socket",
    "expected_answer",
    "gold_patch",
    "gold_sql",
    "grader_feedback",
    "hidden_answer",
    "hidden_tests",
    "private_endpoint",
    "private_path",
    "reference_answer",
    "reference_report",
    "secret",
}


def _key(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def assert_public_payload(value: Any, *, location: str = "payload") -> None:
    """Reject evaluator-only data before it can enter runtime state."""

    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = _key(raw_key)
            if key in _FORBIDDEN_KEYS or key.endswith("_api_key"):
                raise ValueError(f"forbidden private field in {location}: {raw_key}")
            assert_public_payload(item, location=f"{location}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_public_payload(item, location=f"{location}[{index}]")


def _frozen(value: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _events(value: tuple[Mapping[str, Any], ...]) -> tuple[Mapping[str, Any], ...]:
    return tuple(dict(item) for item in value)


@dataclass(frozen=True)
class ExecutionFailure:
    kind: FailureKind
    message: str
    retryable: bool = False

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError("failure message must be non-empty")


@dataclass(frozen=True)
class State0Request:
    task_id: str
    domain: str
    resolved_requirement: str
    public_runtime_assets: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = STATE0_SCHEMA_VERSION
    requirement_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != STATE0_SCHEMA_VERSION:
            raise ValueError("unsupported State 0 request schema")
        if not self.task_id.strip() or not self.domain.strip():
            raise ValueError("State 0 task and domain are required")
        requirement = RequirementState(self.resolved_requirement)
        assets = dict(self.public_runtime_assets)
        assert_public_payload(assets, location="public_runtime_assets")
        object.__setattr__(self, "resolved_requirement", requirement.text)
        object.__setattr__(self, "requirement_digest", requirement.digest)
        object.__setattr__(self, "public_runtime_assets", _frozen(assets))


@dataclass(frozen=True)
class StackPlannerRootResult:
    finished: bool
    artifact_path: str
    artifact: str
    root_plan: Mapping[str, Any] = field(default_factory=dict)
    work_state: Mapping[str, Any] = field(default_factory=dict)
    trace: tuple[Mapping[str, Any], ...] = ()
    failure: ExecutionFailure | None = None

    def __post_init__(self) -> None:
        if self.finished and self.failure is not None:
            raise ValueError("finished root result cannot contain a failure")
        if not self.finished and self.failure is None:
            raise ValueError("failed root result requires a typed failure")
        if not self.artifact_path or self.artifact_path.startswith("/"):
            raise ValueError("root artifact path must be a safe relative path")
        if ".." in self.artifact_path.split("/"):
            raise ValueError("root artifact path must be a safe relative path")
        root_plan = dict(self.root_plan)
        work_state = dict(self.work_state)
        trace = tuple(dict(item) for item in self.trace)
        assert_public_payload(root_plan, location="root_plan")
        assert_public_payload(work_state, location="root_work_state")
        assert_public_payload(trace, location="root_trace")
        object.__setattr__(self, "root_plan", _frozen(root_plan))
        object.__setattr__(self, "work_state", _frozen(work_state))
        object.__setattr__(self, "trace", _events(trace))


@dataclass(frozen=True)
class State0Candidate:
    state0_id: str
    candidate_id: str
    request: State0Request
    root_result: StackPlannerRootResult

    def __post_init__(self) -> None:
        if not self.state0_id or not self.candidate_id:
            raise ValueError("State 0 and candidate identifiers are required")


@dataclass(frozen=True)
class State0:
    state0_id: str
    task_id: str
    domain: str
    resolved_requirement: str
    requirement_digest: str
    public_runtime_assets: Mapping[str, Any]
    initial_workspace: WorkState
    root_plan: Mapping[str, Any]
    root_work_state: Mapping[str, Any]
    artifact_path: str
    artifact: str
    root_trace: tuple[Mapping[str, Any], ...]
    finalization_receipt: Mapping[str, Any]
    checkpoint_id: str
    checkpoint_digest: str
    workspace_digest: str
    failure: ExecutionFailure | None = None
    schema_version: int = STATE0_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STATE0_SCHEMA_VERSION:
            raise ValueError("unsupported State 0 schema")
        if self.failure is not None:
            raise ValueError("accepted State 0 cannot contain a failure")
        if not self.state0_id or not self.task_id or not self.domain:
            raise ValueError("State 0 identity fields are required")
        requirement = RequirementState(self.resolved_requirement)
        if self.requirement_digest != requirement.digest:
            raise ValueError("State 0 requirement digest does not match")
        if self.workspace_digest != self.initial_workspace.digest:
            raise ValueError("State 0 workspace digest does not match")
        if not self.checkpoint_id or not self.checkpoint_digest:
            raise ValueError("State 0 checkpoint binding is required")
        assets = dict(self.public_runtime_assets)
        plan = dict(self.root_plan)
        work_state = dict(self.root_work_state)
        trace = tuple(dict(item) for item in self.root_trace)
        receipt = dict(self.finalization_receipt)
        assert_public_payload(assets, location="public_runtime_assets")
        assert_public_payload(plan, location="root_plan")
        assert_public_payload(work_state, location="root_work_state")
        assert_public_payload(trace, location="root_trace")
        assert_public_payload(receipt, location="finalization_receipt")
        object.__setattr__(self, "resolved_requirement", requirement.text)
        object.__setattr__(self, "public_runtime_assets", _frozen(assets))
        object.__setattr__(self, "root_plan", _frozen(plan))
        object.__setattr__(self, "root_work_state", _frozen(work_state))
        object.__setattr__(self, "root_trace", _events(trace))
        object.__setattr__(self, "finalization_receipt", _frozen(receipt))
