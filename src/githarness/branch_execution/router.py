"""Execution Router contracts and an offline deterministic policy."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping, Protocol

from githarness.errors import ModelError, ProtocolError
from githarness.model import OpenAICompatibleClient
from githarness.versioning.models import Checkpoint

from .modes import ExecutionMode


@dataclass(frozen=True)
class ExecutionDecision:
    mode: ExecutionMode
    reason: str


class ExecutionRouter(Protocol):
    def decide(
        self,
        *,
        requirement: str,
        base_requirement: str,
        update_hint: str,
        base_checkpoint: Checkpoint,
    ) -> ExecutionDecision: ...


@dataclass
class ConservativeExecutionRouter:
    """Offline policy: reuse only when the caller provides an explicit update hint."""

    def decide(
        self,
        *,
        requirement: str,
        base_requirement: str,
        update_hint: str,
        base_checkpoint: Checkpoint,
    ) -> ExecutionDecision:
        if update_hint.strip() and base_checkpoint.work_state.files:
            return ExecutionDecision(
                ExecutionMode.REUSE_AND_PATCH,
                "A verified non-empty base and explicit update hint are available.",
            )
        return ExecutionDecision(
            ExecutionMode.FRESH_SOLVE,
            "No explicit safe reuse signal is available.",
        )


@dataclass
class OpenAIExecutionRouter:
    """Choose FreshSolve or ReuseAndPatch without seeing private references."""

    client: OpenAICompatibleClient

    def decide(
        self,
        *,
        requirement: str,
        base_requirement: str,
        update_hint: str,
        base_checkpoint: Checkpoint,
    ) -> ExecutionDecision:
        context = {
            "requirement": requirement,
            "base_requirement": base_requirement,
            "update_hint": update_hint,
            "base_workspace_digest": base_checkpoint.work_state.digest,
            "base_artifacts": sorted(base_checkpoint.work_state.files),
            "base_verified": base_checkpoint.receipt.get("passed") is True,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Branch Execution Router. Select ReuseAndPatch only when the verified "
                    "base workspace is compatible and materially useful for the new complete requirement. "
                    "Select FreshSolve when inheritance is unsafe, irrelevant, or likely to retain stale state. "
                    "Do not solve the task or infer hidden evaluation data."
                ),
            },
            {"role": "user", "content": json.dumps(context, ensure_ascii=False, sort_keys=True)},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "route_execution",
                    "description": "Choose one execution mode.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mode": {"type": "string", "enum": ["reuse_and_patch", "fresh_solve"]},
                            "reason": {"type": "string", "minLength": 1},
                        },
                        "required": ["mode", "reason"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        response = self.client.complete(messages, tools=tools)
        calls = response.message.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 1:
            raise ModelError("Execution Router must return exactly one tool call")
        function = calls[0].get("function") if isinstance(calls[0], Mapping) else None
        if not isinstance(function, Mapping) or function.get("name") != "route_execution":
            raise ProtocolError("Execution Router returned an invalid tool")
        raw = function.get("arguments")
        try:
            arguments = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError as error:
            raise ProtocolError("Execution Router arguments are not valid JSON") from error
        if not isinstance(arguments, dict) or set(arguments) != {"mode", "reason"}:
            raise ProtocolError("Execution Router returned invalid arguments")
        try:
            mode = ExecutionMode(str(arguments["mode"]))
        except ValueError as error:
            raise ProtocolError("Execution Router returned an unsupported mode") from error
        if mode not in {ExecutionMode.REUSE_AND_PATCH, ExecutionMode.FRESH_SOLVE}:
            raise ProtocolError("Execution Router cannot choose ExactReuse")
        reason = arguments["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise ProtocolError("Execution Router reason is empty")
        return ExecutionDecision(mode, reason.strip())
