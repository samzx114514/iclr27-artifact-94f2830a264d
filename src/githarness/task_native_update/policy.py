"""Provider-neutral policy adapters for Task-Native Update Agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from githarness.errors import ProtocolError
from githarness.model import OpenAICompatibleClient

from .tools import TaskNativePolicyResponse, TaskNativeToolCall


class TaskNativePolicy(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> TaskNativePolicyResponse: ...


@dataclass
class OpenAITaskNativePolicy:
    client: OpenAICompatibleClient

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> TaskNativePolicyResponse:
        response = self.client.complete(messages, tools=tools)
        raw_calls = response.message.get("tool_calls")
        if raw_calls is None:
            raw_calls = []
        if not isinstance(raw_calls, list):
            raise ProtocolError("Task-Native Update Agent returned malformed tool calls")
        calls = []
        for index, raw in enumerate(raw_calls):
            if not isinstance(raw, dict) or not isinstance(raw.get("function"), dict):
                raise ProtocolError("Task-Native Update Agent returned a malformed tool call")
            function = raw["function"]
            calls.append(TaskNativeToolCall(
                str(raw.get("id") or f"call-{index}"),
                str(function.get("name") or ""),
                function.get("arguments"),
            ))
        content = response.message.get("content")
        return TaskNativePolicyResponse(
            tuple(calls),
            content if isinstance(content, str) else None,
        )
