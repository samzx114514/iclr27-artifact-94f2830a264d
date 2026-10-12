"""Typed public contracts for Task-Native Update Agent execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from githarness.domains.specs import DomainSpec
from githarness.state0.contracts import assert_public_payload
from githarness.version_interface.protocol import CandidateWorkState
from githarness.versioning.models import Checkpoint, WorkState, digest


UPDATE_RESULT_SCHEMA_VERSION = "task-native-update-result-v1"


@dataclass(frozen=True)
class UpdateExecutionFailure:
    kind: str
    message: str
    retryable: bool = False

    def __post_init__(self) -> None:
        if not self.kind or not self.message:
            raise ValueError("update failure kind and message are required")


@dataclass(frozen=True)
class UpdateAgentRequest:
    task_id: str
    domain: str
    candidate: CandidateWorkState
    resolved_requirement: str
    selected_base_checkpoint: Checkpoint | None
    update_hint: str
    domain_spec: DomainSpec
    public_runtime_assets: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id or not self.resolved_requirement.strip():
            raise ValueError("task ID and complete resolved requirement are required")
        if self.domain != self.domain_spec.name:
            raise ValueError("Update Agent domain does not match DomainSpec")
        if self.selected_base_checkpoint is not None:
            base_id = self.candidate.base_checkpoint_id
            if base_id not in {None, self.selected_base_checkpoint.checkpoint_id}:
                raise ValueError("candidate/base checkpoint binding mismatch")
        assets = dict(self.public_runtime_assets)
        assert_public_payload(assets, location="update_public_runtime_assets")
        from .domains import get_update_domain_contract

        domain_contract = get_update_domain_contract(self.domain)
        if (
            domain_contract.artifact_path != self.domain_spec.artifact_path
            or domain_contract.public_capabilities != self.domain_spec.public_capabilities
        ):
            raise ValueError("Update Agent domain contract does not match DomainSpec")
        domain_contract.validate_runtime_assets(assets)
        object.__setattr__(self, "resolved_requirement", self.resolved_requirement.strip())
        object.__setattr__(self, "update_hint", self.update_hint.strip())
        object.__setattr__(self, "public_runtime_assets", assets)

    def public_projection(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "candidate_id": self.candidate.candidate_id,
            "candidate_base_checkpoint_id": self.candidate.base_checkpoint_id,
            "resolved_requirement": self.resolved_requirement,
            "selected_base_checkpoint_id": (
                self.selected_base_checkpoint.checkpoint_id
                if self.selected_base_checkpoint else None
            ),
            "update_hint": self.update_hint,
            "artifact_path": self.domain_spec.artifact_path,
            "public_runtime_assets": {
                "capabilities": list(self.domain_spec.public_capabilities),
                "provided_capability_keys": sorted(
                    key for key in self.public_runtime_assets if key != "capabilities"
                ),
            },
        }


@dataclass(frozen=True)
class UpdateAgentResult:
    finished: bool
    artifact_path: str
    artifact: str
    modified_work_state: WorkState
    trace: tuple[Mapping[str, Any], ...] = ()
    failure: UpdateExecutionFailure | None = None
    schema_version: str = UPDATE_RESULT_SCHEMA_VERSION
    result_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != UPDATE_RESULT_SCHEMA_VERSION:
            raise ValueError(f"unsupported Update Agent schema: {self.schema_version}")
        if not self.artifact_path:
            raise ValueError("Update Agent artifact path is required")
        if self.finished and self.failure is not None:
            raise ValueError("finished Update Agent result cannot contain a failure")
        if not self.finished and self.failure is None:
            raise ValueError("unfinished Update Agent result requires a typed failure")
        trace = tuple(dict(item) for item in self.trace)
        assert_public_payload(trace, location="update_trace")
        assert_public_payload(
            dict(self.modified_work_state.metadata),
            location="update_modified_work_state.metadata",
        )
        assert_public_payload(
            {path: None for path in self.modified_work_state.files},
            location="update_modified_work_state.files",
        )
        object.__setattr__(self, "trace", trace)
        object.__setattr__(self, "result_digest", digest(_result_payload(self, False)))


def update_result_to_dict(result: UpdateAgentResult) -> dict[str, Any]:
    return _result_payload(result, True)


def update_result_from_dict(value: Mapping[str, Any]) -> UpdateAgentResult:
    expected = {
        "schema_version", "finished", "artifact_path", "artifact",
        "modified_work_state", "trace", "failure", "result_digest",
    }
    if set(value) != expected:
        raise ValueError("Update Agent result fields do not match schema v1")
    raw_state = value["modified_work_state"]
    if not isinstance(raw_state, Mapping):
        raise ValueError("modified work state must be an object")
    work_state = WorkState(
        files=dict(raw_state.get("files") or {}),
        metadata=dict(raw_state.get("metadata") or {}),
    )
    if raw_state.get("digest") != work_state.digest:
        raise ValueError("Update Agent work-state digest mismatch")
    raw_failure = value["failure"]
    failure = UpdateExecutionFailure(**raw_failure) if isinstance(raw_failure, dict) else None
    result = UpdateAgentResult(
        finished=bool(value["finished"]),
        artifact_path=str(value["artifact_path"]),
        artifact=str(value["artifact"]),
        modified_work_state=work_state,
        trace=tuple(dict(item) for item in value["trace"]),
        failure=failure,
        schema_version=str(value["schema_version"]),
    )
    if value["result_digest"] != result.result_digest:
        raise ValueError("Update Agent result digest mismatch")
    return result


def _result_payload(result: UpdateAgentResult, include_digest: bool) -> dict[str, Any]:
    failure = result.failure
    value: dict[str, Any] = {
        "schema_version": result.schema_version,
        "finished": result.finished,
        "artifact_path": result.artifact_path,
        "artifact": result.artifact,
        "modified_work_state": {
            "files": dict(result.modified_work_state.files),
            "metadata": dict(result.modified_work_state.metadata),
            "digest": result.modified_work_state.digest,
        },
        "trace": [dict(item) for item in result.trace],
        "failure": (
            {"kind": failure.kind, "message": failure.message, "retryable": failure.retryable}
            if failure else None
        ),
    }
    if include_digest:
        value["result_digest"] = result.result_digest
    return value
