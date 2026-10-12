"""Versioned public, private, manifest, and evaluation records."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Literal, Mapping

from githarness.versioning.models import frozen_mapping


Domain = Literal["math", "sql", "search", "research", "code"]
EvaluationStatus = Literal[
    "correct",
    "incorrect",
    "scored",
    "protocol_error",
    "infrastructure_failure",
    "model_failure",
    "verifier_failure",
]


@dataclass(frozen=True)
class PublicCase:
    task_id: str
    domain: Domain
    turns: tuple[Mapping[str, str], ...]
    runtime_assets: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.task_id or not self.turns:
            raise ValueError("public case requires task_id and turns")
        clean_turns = []
        for turn in self.turns:
            if set(turn) != {"role", "content"} or turn["role"] != "user" or not turn["content"].strip():
                raise ValueError("public turns must be non-empty user role/content objects")
            clean_turns.append(frozen_mapping(turn))
        object.__setattr__(self, "turns", tuple(clean_turns))
        object.__setattr__(self, "runtime_assets", frozen_mapping(self.runtime_assets))


@dataclass(frozen=True)
class PrivateEvaluationRecord:
    task_id: str
    domain: Domain
    verifier: str
    reference: Mapping[str, Any]
    asset_bindings: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.task_id or not self.verifier:
            raise ValueError("private evaluation record requires identifiers")
        object.__setattr__(self, "reference", frozen_mapping(self.reference))
        object.__setattr__(self, "asset_bindings", frozen_mapping(self.asset_bindings))


@dataclass(frozen=True)
class EvaluationResult:
    task_id: str
    domain: Domain
    status: EvaluationStatus
    finished: bool
    correct: bool | None
    score: float | None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", frozen_mapping(self.details))


def to_plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: to_plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    return value
