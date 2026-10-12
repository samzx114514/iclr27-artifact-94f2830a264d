"""Immutable requirement, work-state, version, and checkpoint models."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import hashlib
import json
from types import MappingProxyType
from typing import Any, Literal, Mapping


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def frozen_mapping(value: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True)
class RequirementState:
    text: str
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        normalized = "\n".join(line.rstrip() for line in self.text.strip().splitlines())
        if not normalized:
            raise ValueError("requirement state must be non-empty")
        object.__setattr__(self, "text", normalized)
        object.__setattr__(self, "digest", digest({"text": normalized}))


@dataclass(frozen=True)
class WorkState:
    files: Mapping[str, str]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        safe_files = {str(k): str(v) for k, v in self.files.items()}
        if any(not path or path.startswith("/") or ".." in path.split("/") for path in safe_files):
            raise ValueError("work-state paths must be safe relative paths")
        safe_metadata = copy.deepcopy(dict(self.metadata))
        object.__setattr__(self, "files", frozen_mapping(safe_files))
        object.__setattr__(self, "metadata", frozen_mapping(safe_metadata))
        object.__setattr__(
            self,
            "digest",
            digest({"files": safe_files, "metadata": safe_metadata}),
        )


@dataclass(frozen=True)
class Checkpoint:
    checkpoint_id: str
    version_id: str
    work_state: WorkState
    parent_checkpoint_id: str | None = None
    receipt: Mapping[str, Any] = field(default_factory=dict)
    final_response: str = ""
    state0_id: str | None = None
    origin_state0_id: str | None = None
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.checkpoint_id or not self.version_id:
            raise ValueError("checkpoint and version identifiers are required")
        receipt = copy.deepcopy(dict(self.receipt))
        if self.state0_id is not None and self.parent_checkpoint_id is not None:
            raise ValueError("State 0 checkpoint cannot have a parent checkpoint")
        if self.state0_id is not None and self.origin_state0_id != self.state0_id:
            raise ValueError("State 0 checkpoint must bind its own origin")
        if self.parent_checkpoint_id is not None and self.origin_state0_id is None:
            raise ValueError("descendant checkpoint requires an origin State 0")
        object.__setattr__(self, "receipt", frozen_mapping(receipt))
        object.__setattr__(
            self,
            "digest",
            digest(
                {
                    "checkpoint_id": self.checkpoint_id,
                    "version_id": self.version_id,
                    "work_state_digest": self.work_state.digest,
                    "parent_checkpoint_id": self.parent_checkpoint_id,
                    "receipt": receipt,
                    "final_response": self.final_response,
                    "state0_id": self.state0_id,
                    "origin_state0_id": self.origin_state0_id,
                }
            ),
        )


@dataclass(frozen=True)
class Version:
    version_id: str
    requirement_state: RequirementState
    parent_version_id: str | None
    checkpoint_id: str
    source_turn: int
    created_by: Literal["commit", "reuse"] = "commit"
    summary: str = ""

    def __post_init__(self) -> None:
        if not self.version_id or not self.checkpoint_id:
            raise ValueError("version and checkpoint identifiers are required")
        if self.source_turn < 0:
            raise ValueError("source_turn must be non-negative")
        if not isinstance(self.summary, str):
            raise TypeError("version summary must be a string")
