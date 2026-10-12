"""Model-driven Git Agent policy using native tool calls."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

from githarness.errors import ProtocolError
from githarness.model import OpenAICompatibleClient

from .actions import GitAction, parse_git_action


SYSTEM_PROMPT = """You are the Git Agent for requirement resolution. Do not solve the task.
Maintain one complete, self-contained requirement state under evolving user feedback.
The active version is the mandatory default base for cumulative edits. Preserve every
unmentioned fact, constraint, requested outcome, dependency, and presentation contract.
Treat entities, attributes, numbers, relationships, filters, projections, and outcomes
as semantic slots: replace every stale occurrence of an explicitly corrected slot and
remove every explicitly withdrawn clause. Never calculate or add derived facts.

The context contains one complete active requirement plus compact navigation summaries.
Historical summaries are not authoritative wording. Use Inspect only when one exact
historical requirement is needed; only one Inspect is allowed per turn, and Inspect does
not activate or copy a version. A comparison begins with the complete active task as
scenario A and changes only the named slots in scenario B. For withdrawal/restoration,
select the parent of the version that first introduced cancelled content, then replay
only later persistent edits that remain requested.

Commit creates a pending handoff from the selected base; execution acceptance happens
later. Its resolved_requirement must contain the entire resulting task, never a delta.
Use JSON null for an independent task. Reuse is allowed only when an accepted historical
version exactly equals the desired complete task; it is a zero-execution path. The
update_hint is a short non-authoritative execution hint and never replaces the complete
requirement. Never invent version/checkpoint IDs, answers, digests, tool results, or
hidden evaluator facts. End with exactly one successful Commit or Reuse."""


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "Inspect",
            "description": "Read one existing version's complete requirement and output.",
            "parameters": {
                "type": "object",
                "properties": {"version_id": {"type": "string", "minLength": 1, "maxLength": 128}},
                "required": ["version_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Commit",
            "description": "Commit a complete resolved requirement from a selected base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "resolved_requirement": {"type": "string", "minLength": 1, "maxLength": 100000},
                    "base_version_id": {"type": ["string", "null"], "maxLength": 128},
                    "update_hint": {"type": "string", "maxLength": 2000},
                },
                "required": ["resolved_requirement", "base_version_id", "update_hint"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Reuse",
            "description": "Activate an existing version only when it exactly matches the desired task.",
            "parameters": {
                "type": "object",
                "properties": {"version_id": {"type": "string", "minLength": 1, "maxLength": 128}},
                "required": ["version_id"],
                "additionalProperties": False,
            },
        },
    },
]


@dataclass
class OpenAIGitPolicy:
    client: OpenAICompatibleClient

    def choose(self, context: dict[str, Any]) -> GitAction:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False, sort_keys=True)},
        ]
        response = self.client.complete(messages, tools=TOOLS)
        calls = response.message.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 1:
            raise ProtocolError("Git Agent must return exactly one tool call")
        function = calls[0].get("function") if isinstance(calls[0], Mapping) else None
        if not isinstance(function, Mapping):
            raise ProtocolError("Git Agent returned a malformed tool call")
        name = function.get("name")
        raw_arguments = function.get("arguments")
        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        except json.JSONDecodeError as error:
            raise ProtocolError("Git Agent tool arguments are not valid JSON") from error
        if not isinstance(arguments, dict):
            raise ProtocolError("Git Agent tool arguments must be an object")
        return parse_git_action(name, arguments)
