"""Optional text-action compatibility transport for tool-call-fragile models."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from githarness.errors import ProtocolError
from githarness.model import OpenAICompatibleClient

from .actions import GitAction, parse_git_action
from .openai_policy import SYSTEM_PROMPT


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def parse_text_action(text: str) -> GitAction:
    if not isinstance(text, str) or not text.strip():
        raise ProtocolError("empty Git Agent text action")
    candidate = text.strip()
    fenced = _FENCE_RE.findall(candidate)
    if fenced:
        if len(fenced) != 1 or not candidate.startswith("```") or not candidate.endswith("```"):
            raise ProtocolError("Git Agent must return exactly one JSON action")
        candidate = fenced[0].strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ProtocolError("Git Agent text action is not valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"tool", "arguments"}:
        raise ProtocolError("Git Agent text action must contain only tool and arguments")
    return parse_git_action(payload["tool"], payload["arguments"])


@dataclass
class QwenTextGitPolicy:
    client: OpenAICompatibleClient

    def choose(self, context: dict[str, Any]) -> GitAction:
        instruction = (
            SYSTEM_PROMPT
            + "\nReturn exactly one JSON object: "
            + '{"tool":"Inspect|Commit|Reuse","arguments":{...}}. '
            + "Do not include explanations, runtime IDs, digests, or a FINISH action."
        )
        response = self.client.complete(
            [
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False, sort_keys=True)},
            ]
        )
        content = response.message.get("content")
        if not isinstance(content, str):
            raise ProtocolError("Git Agent text response has no content")
        return parse_text_action(content)
