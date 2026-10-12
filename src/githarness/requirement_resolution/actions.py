"""Strict model-visible Git Agent actions."""

from dataclasses import dataclass
from typing import Any, Mapping

from githarness.errors import ProtocolError


@dataclass(frozen=True)
class Inspect:
    version_id: str

    def __post_init__(self) -> None:
        _identifier(self.version_id, "version_id")


@dataclass(frozen=True)
class Commit:
    resolved_requirement: str
    base_version_id: str | None
    update_hint: str = ""

    def __post_init__(self) -> None:
        _text(self.resolved_requirement, "resolved_requirement")
        if self.base_version_id is not None:
            _identifier(self.base_version_id, "base_version_id")
        if not isinstance(self.update_hint, str):
            raise ProtocolError("update_hint must be a string")


@dataclass(frozen=True)
class Reuse:
    version_id: str

    def __post_init__(self) -> None:
        _identifier(self.version_id, "version_id")


GitAction = Inspect | Commit | Reuse


def parse_git_action(name: object, arguments: object) -> GitAction:
    """Parse one action while rejecting runtime-owned or unknown fields."""

    if not isinstance(arguments, Mapping):
        raise ProtocolError("Git Agent tool arguments must be an object")
    data = dict(arguments)
    if name == "Inspect" and set(data) == {"version_id"}:
        return Inspect(_identifier(data["version_id"], "version_id"))
    if name == "Reuse" and set(data) == {"version_id"}:
        return Reuse(_identifier(data["version_id"], "version_id"))
    if name == "Commit" and set(data) == {
        "resolved_requirement",
        "base_version_id",
        "update_hint",
    }:
        base = data["base_version_id"]
        if base is not None:
            base = _identifier(base, "base_version_id")
        hint = data["update_hint"]
        if not isinstance(hint, str):
            raise ProtocolError("update_hint must be a string")
        return Commit(
            _text(data["resolved_requirement"], "resolved_requirement"),
            base,
            hint.strip(),
        )
    raise ProtocolError(f"invalid Git Agent tool call: {name}")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{name} must be a non-empty string")
    if len(value) > 100_000:
        raise ProtocolError(f"{name} exceeds the size limit")
    for character in value:
        if character in {"\n", "\t"}:
            continue
        if ord(character) < 32 or character in {"\u200b", "\ufeff"}:
            raise ProtocolError(f"{name} contains unsupported control characters")
    return value.strip()


def _identifier(value: Any, name: str) -> str:
    result = _text(value, name)
    if len(result) > 128:
        raise ProtocolError(f"{name} exceeds the size limit")
    return result
