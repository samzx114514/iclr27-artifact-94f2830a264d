"""Strict State 0 v1 JSON-compatible serialization."""

from __future__ import annotations

from typing import Any, Mapping

from githarness.errors import State0SerializationError
from githarness.versioning.models import WorkState

from .contracts import STATE0_SCHEMA_VERSION, State0


_FIELDS = {
    "schema_version",
    "state0_id",
    "task_id",
    "domain",
    "resolved_requirement",
    "requirement_digest",
    "public_runtime_assets",
    "initial_workspace",
    "root_plan",
    "root_work_state",
    "artifact_path",
    "artifact",
    "root_trace",
    "finalization_receipt",
    "checkpoint_id",
    "checkpoint_digest",
    "workspace_digest",
    "failure",
}


def state0_to_dict(state0: State0) -> dict[str, Any]:
    return {
        "schema_version": state0.schema_version,
        "state0_id": state0.state0_id,
        "task_id": state0.task_id,
        "domain": state0.domain,
        "resolved_requirement": state0.resolved_requirement,
        "requirement_digest": state0.requirement_digest,
        "public_runtime_assets": dict(state0.public_runtime_assets),
        "initial_workspace": {
            "files": dict(state0.initial_workspace.files),
            "metadata": dict(state0.initial_workspace.metadata),
            "digest": state0.initial_workspace.digest,
        },
        "root_plan": dict(state0.root_plan),
        "root_work_state": dict(state0.root_work_state),
        "artifact_path": state0.artifact_path,
        "artifact": state0.artifact,
        "root_trace": [dict(item) for item in state0.root_trace],
        "finalization_receipt": dict(state0.finalization_receipt),
        "checkpoint_id": state0.checkpoint_id,
        "checkpoint_digest": state0.checkpoint_digest,
        "workspace_digest": state0.workspace_digest,
        "failure": None,
    }


def state0_from_dict(value: Mapping[str, Any]) -> State0:
    try:
        if set(value) != _FIELDS:
            raise ValueError("State 0 fields do not match schema v1")
        if value["schema_version"] != STATE0_SCHEMA_VERSION:
            raise ValueError("unsupported State 0 schema")
        if value["failure"] is not None:
            raise ValueError("serialized accepted State 0 cannot contain failure")
        workspace_value = _mapping(value["initial_workspace"], "initial_workspace")
        if set(workspace_value) != {"files", "metadata", "digest"}:
            raise ValueError("invalid initial workspace schema")
        workspace = WorkState(
            files=_mapping(workspace_value["files"], "files"),
            metadata=_mapping(workspace_value["metadata"], "metadata"),
        )
        if workspace.digest != workspace_value["digest"]:
            raise ValueError("serialized workspace digest does not match")
        trace_value = value["root_trace"]
        if not isinstance(trace_value, list):
            raise ValueError("root_trace must be a list")
        return State0(
            schema_version=STATE0_SCHEMA_VERSION,
            state0_id=_string(value["state0_id"], "state0_id"),
            task_id=_string(value["task_id"], "task_id"),
            domain=_string(value["domain"], "domain"),
            resolved_requirement=_string(value["resolved_requirement"], "resolved_requirement"),
            requirement_digest=_string(value["requirement_digest"], "requirement_digest"),
            public_runtime_assets=_mapping(value["public_runtime_assets"], "public_runtime_assets"),
            initial_workspace=workspace,
            root_plan=_mapping(value["root_plan"], "root_plan"),
            root_work_state=_mapping(value["root_work_state"], "root_work_state"),
            artifact_path=_string(value["artifact_path"], "artifact_path"),
            artifact=str(value["artifact"]),
            root_trace=tuple(_mapping(item, "root_trace item") for item in trace_value),
            finalization_receipt=_mapping(value["finalization_receipt"], "finalization_receipt"),
            checkpoint_id=_string(value["checkpoint_id"], "checkpoint_id"),
            checkpoint_digest=_string(value["checkpoint_digest"], "checkpoint_digest"),
            workspace_digest=_string(value["workspace_digest"], "workspace_digest"),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise State0SerializationError(str(error)) from error


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value
