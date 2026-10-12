"""Strict tools for the independent Task-Native Update Agent loop."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping


@dataclass(frozen=True)
class TaskNativeToolCall:
    call_id: str
    name: str
    arguments: Mapping[str, Any] | str


@dataclass(frozen=True)
class TaskNativePolicyResponse:
    tool_calls: tuple[TaskNativeToolCall, ...] = ()
    content: str | None = None


def parse_tool_call(call: TaskNativeToolCall, *, domain: str) -> tuple[str, dict[str, Any]]:
    arguments = call.arguments
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise ValueError("tool arguments are not valid JSON") from error
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be an object")
    data = dict(arguments)
    common = {
        "glob_workspace",
        "grep_workspace",
        "read_workspace",
        "write_artifact",
        "patch_artifact",
        "delete_workspace",
        "finish",
    }
    domain_tools = {
        "math": {"calculate"},
        "sql": set(),
        "search": {"search"},
        "research": {"search"},
        "code": {"testbed_execute"},
    }
    if call.name not in common | domain_tools[domain]:
        raise ValueError(f"tool is not permitted for {domain}: {call.name}")
    required = {
        "glob_workspace": {"pattern"},
        "grep_workspace": {"query", "pattern"},
        "read_workspace": {"path"},
        "write_artifact": {"content"},
        "patch_artifact": {"old_text", "new_text"},
        "delete_workspace": {"path", "reason"},
        "calculate": {"expression"},
        "search": {"query"},
        "testbed_execute": {"command"},
        "finish": set(),
    }[call.name]
    if set(data) != required:
        raise ValueError(f"invalid arguments for {call.name}")
    for name, value in data.items():
        if not isinstance(value, str) or (name != "new_text" and not value.strip()):
            raise ValueError(f"{name} must be a non-empty string")
        if len(value) > 100_000:
            raise ValueError(f"{name} exceeds the size limit")
    return call.name, data


def task_native_tool_schemas(domain: str) -> list[dict[str, Any]]:
    tools = [
        _tool(
            "glob_workspace",
            "List bounded candidate-relative files matching one safe glob pattern.",
            {"pattern": _string()},
        ),
        _tool(
            "grep_workspace",
            "Find bounded literal text matches in candidate files selected by a safe glob.",
            {"query": _string(), "pattern": _string()},
        ),
        _tool("read_workspace", "Read one candidate-relative public file.", {"path": _string()}),
        _tool("write_artifact", "Write the domain's one model-owned artifact.", {"content": _string()}),
        _tool(
            "patch_artifact",
            "Replace one exact substring in the domain artifact.",
            {"old_text": _string(), "new_text": {"type": "string", "maxLength": 100000}},
        ),
        _tool(
            "delete_workspace",
            "Delete one non-protected inherited candidate file only when the requirement explicitly withdraws it.",
            {"path": _string(), "reason": _string()},
        ),
    ]
    if domain == "math":
        tools.append(_tool("calculate", "Evaluate one public arithmetic expression.", {"expression": _string()}))
    elif domain in {"search", "research"}:
        tools.append(_tool("search", "Search through the injected narrow public provider.", {"query": _string()}))
    elif domain == "code":
        tools.append(_tool("testbed_execute", "Run one command inside the network-isolated /testbed.", {"command": _string()}))
    tools.append(_tool("finish", "Finish after the public artifact is ready.", {}))
    return tools


def _string() -> dict[str, Any]:
    return {"type": "string", "minLength": 1, "maxLength": 100000}


def _tool(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }
