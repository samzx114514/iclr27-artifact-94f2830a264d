"""Versioned, auditable Git Agent to Branch Execution handoff."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from githarness.versioning.models import Checkpoint, RequirementState, digest


HANDOFF_SCHEMA_VERSION = "resolved-requirement-handoff-v1"


@dataclass(frozen=True)
class CheckpointBinding:
    version_id: str
    checkpoint_id: str
    checkpoint_digest: str
    workspace_digest: str
    origin_state0_id: str | None

    @classmethod
    def from_checkpoint(cls, checkpoint: Checkpoint) -> "CheckpointBinding":
        return cls(
            checkpoint.version_id,
            checkpoint.checkpoint_id,
            checkpoint.digest,
            checkpoint.work_state.digest,
            checkpoint.origin_state0_id,
        )


@dataclass(frozen=True)
class VersionHandoff:
    resolved_requirement: str
    selected_base_version_id: str | None
    target_version_id: str
    update_hint: str
    reused_version_id: str | None = None
    events: tuple[dict[str, Any], ...] = ()
    task_id: str | None = None
    domain: str | None = None
    selected_base_checkpoint: CheckpointBinding | None = None
    exact_reuse_identity: str | None = None
    schema_version: str = HANDOFF_SCHEMA_VERSION
    requirement_digest: str = field(init=False)
    handoff_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != HANDOFF_SCHEMA_VERSION:
            raise ValueError(f"unsupported handoff schema: {self.schema_version}")
        requirement = RequirementState(self.resolved_requirement)
        object.__setattr__(self, "resolved_requirement", requirement.text)
        if not self.target_version_id:
            raise ValueError("target version ID must be non-empty")
        if self.selected_base_checkpoint is not None:
            if self.selected_base_checkpoint.version_id != self.selected_base_version_id:
                raise ValueError("selected base checkpoint/version mismatch")
        requirement_digest = requirement.digest
        object.__setattr__(self, "requirement_digest", requirement_digest)
        object.__setattr__(self, "handoff_digest", digest(_payload(self, include_digest=False)))


def handoff_to_dict(handoff: VersionHandoff) -> dict[str, Any]:
    return _payload(handoff, include_digest=True)


def handoff_from_dict(value: Mapping[str, Any]) -> VersionHandoff:
    expected = {
        "schema_version", "task_id", "domain", "resolved_requirement",
        "requirement_digest", "selected_base_version_id", "selected_base_checkpoint",
        "target_version_id", "update_hint", "reused_version_id",
        "exact_reuse_identity", "events", "handoff_digest",
    }
    if set(value) != expected:
        raise ValueError("handoff fields do not match schema v1")
    raw_binding = value["selected_base_checkpoint"]
    if raw_binding is not None and not isinstance(raw_binding, dict):
        raise ValueError("selected base checkpoint must be an object or null")
    binding = CheckpointBinding(**raw_binding) if isinstance(raw_binding, dict) else None
    handoff = VersionHandoff(
        resolved_requirement=str(value["resolved_requirement"]),
        selected_base_version_id=value["selected_base_version_id"],
        target_version_id=str(value["target_version_id"]),
        update_hint=str(value["update_hint"]),
        reused_version_id=value["reused_version_id"],
        events=tuple(dict(item) for item in value["events"]),
        task_id=value["task_id"],
        domain=value["domain"],
        selected_base_checkpoint=binding,
        exact_reuse_identity=value["exact_reuse_identity"],
        schema_version=str(value["schema_version"]),
    )
    if value["requirement_digest"] != handoff.requirement_digest:
        raise ValueError("handoff requirement digest mismatch")
    if value["handoff_digest"] != handoff.handoff_digest:
        raise ValueError("handoff digest mismatch")
    return handoff


def _payload(handoff: VersionHandoff, *, include_digest: bool) -> dict[str, Any]:
    binding = handoff.selected_base_checkpoint
    result: dict[str, Any] = {
        "schema_version": handoff.schema_version,
        "task_id": handoff.task_id,
        "domain": handoff.domain,
        "resolved_requirement": handoff.resolved_requirement,
        "requirement_digest": handoff.requirement_digest,
        "selected_base_version_id": handoff.selected_base_version_id,
        "selected_base_checkpoint": (
            {
                "version_id": binding.version_id,
                "checkpoint_id": binding.checkpoint_id,
                "checkpoint_digest": binding.checkpoint_digest,
                "workspace_digest": binding.workspace_digest,
                "origin_state0_id": binding.origin_state0_id,
            }
            if binding else None
        ),
        "target_version_id": handoff.target_version_id,
        "update_hint": handoff.update_hint,
        "reused_version_id": handoff.reused_version_id,
        "exact_reuse_identity": handoff.exact_reuse_identity,
        "events": [dict(item) for item in handoff.events],
    }
    if include_digest:
        result["handoff_digest"] = handoff.handoff_digest
    return result
