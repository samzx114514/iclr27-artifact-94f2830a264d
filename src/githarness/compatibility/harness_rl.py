"""Stable, public-only export consumed by external Harness-RL systems."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Mapping

from githarness.branch_execution.runtime import ExecutionOutcome
from githarness.requirement_resolution.handoff import VersionHandoff, handoff_to_dict
from githarness.state0 import assert_public_payload, state0_to_dict
from githarness.task_native_update import update_result_to_dict
from githarness.versioning.models import Checkpoint, digest


HARNESS_RL_SCHEMA_VERSION = "harness-rl-compat-v1"
_FIELDS = {
    "schema_version",
    "task_id",
    "domain",
    "source_turn",
    "git_agent_trajectory",
    "handoff",
    "execution",
    "evaluator_result",
    "export_digest",
}


def export_harness_rl_turn(
    *,
    task_id: str,
    domain: str,
    source_turn: int,
    handoff: VersionHandoff,
    outcome: ExecutionOutcome,
    checkpoint: Checkpoint,
    evaluator_result: Mapping[str, Any] | Any | None = None,
) -> dict[str, Any]:
    if checkpoint.checkpoint_id != outcome.checkpoint_id:
        raise ValueError("Harness-RL export checkpoint/outcome mismatch")
    if checkpoint.version_id != outcome.version_id:
        raise ValueError("Harness-RL export version/outcome mismatch")
    if checkpoint.work_state.digest != outcome.work_state.digest:
        raise ValueError("Harness-RL export workspace/outcome mismatch")
    evaluator = _public_evaluator(evaluator_result)
    payload: dict[str, Any] = {
        "schema_version": HARNESS_RL_SCHEMA_VERSION,
        "task_id": task_id,
        "domain": domain,
        "source_turn": source_turn,
        "git_agent_trajectory": [{
            "policy_role": "git_agent",
            "segment_index": 0,
            "events": [dict(item) for item in handoff.events],
        }],
        "handoff": handoff_to_dict(handoff),
        "execution": {
            "mode": outcome.mode.value,
            "finished": outcome.finished,
            "version_id": outcome.version_id,
            "checkpoint_binding": {
                "checkpoint_id": checkpoint.checkpoint_id,
                "checkpoint_digest": checkpoint.digest,
                "workspace_digest": checkpoint.work_state.digest,
                "parent_checkpoint_id": checkpoint.parent_checkpoint_id,
                "state0_id": checkpoint.state0_id,
                "origin_state0_id": checkpoint.origin_state0_id,
            },
            "final_response": outcome.final_response,
            "receipt": dict(outcome.receipt),
            "state0": state0_to_dict(outcome.state0) if outcome.state0 else None,
            "update_result": (
                update_result_to_dict(outcome.update_result)
                if outcome.update_result else None
            ),
        },
        "evaluator_result": evaluator,
    }
    assert_public_payload(payload, location="harness_rl_export")
    payload["export_digest"] = digest(payload)
    return payload


def validate_harness_rl_export(value: Mapping[str, Any]) -> None:
    if set(value) != _FIELDS:
        raise ValueError("Harness-RL export fields do not match schema v1")
    if value["schema_version"] != HARNESS_RL_SCHEMA_VERSION:
        raise ValueError("unsupported Harness-RL export schema")
    payload = dict(value)
    supplied = payload.pop("export_digest")
    if supplied != digest(payload):
        raise ValueError("Harness-RL export digest mismatch")
    execution = payload.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("Harness-RL execution export must be an object")
    binding = execution.get("checkpoint_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("Harness-RL checkpoint binding must be an object")
    receipt = execution.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Harness-RL finalization receipt must be an object")
    if binding.get("workspace_digest") != receipt.get("workspace_digest"):
        raise ValueError("Harness-RL receipt/workspace binding mismatch")
    segments = payload.get("git_agent_trajectory")
    if not isinstance(segments, list) or not segments:
        raise ValueError("Harness-RL Git Agent trajectory is required")
    if any(not isinstance(item, Mapping) for item in segments):
        raise ValueError("Harness-RL trajectory segments must be objects")
    if any(item.get("policy_role") != "git_agent" for item in segments):
        raise ValueError("Harness-RL trajectory role mismatch")
    assert_public_payload(payload, location="harness_rl_export")


def _public_evaluator(value: Mapping[str, Any] | Any | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if is_dataclass(value) and not isinstance(value, type):
        result = {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    elif isinstance(value, Mapping):
        result = {str(key): _plain(item) for key, item in value.items()}
    else:
        raise TypeError("evaluator result must be a dataclass, mapping, or null")
    assert_public_payload(result, location="evaluator_result")
    return result


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value
