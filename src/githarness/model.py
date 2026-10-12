"""Provider-neutral OpenAI-compatible model configuration and transport."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Mapping
from urllib import error, request

from .errors import InfrastructureError, ModelError


@dataclass(frozen=True)
class ModelConfig:
    model: str
    base_url: str
    api_key_env: str
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_seconds: float = 120.0
    tool_choice: str = "required"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ModelConfig":
        required = ("model", "base_url", "api_key_env")
        missing = [name for name in required if not isinstance(value.get(name), str) or not value[name].strip()]
        if missing:
            raise ValueError(f"model config is missing: {', '.join(missing)}")
        tool_choice = str(value.get("tool_choice", "required"))
        if tool_choice not in {"required", "auto"}:
            raise ValueError("model tool_choice must be required or auto")
        return cls(
            model=str(value["model"]),
            base_url=str(value["base_url"]),
            api_key_env=str(value["api_key_env"]),
            temperature=float(value.get("temperature", 0.0)),
            max_tokens=int(value.get("max_tokens", 2048)),
            timeout_seconds=float(value.get("timeout_seconds", 120.0)),
            tool_choice=tool_choice,
        )

    @property
    def resolved_base_url(self) -> str:
        value = os.path.expandvars(self.base_url).rstrip("/")
        if "$" in value:
            raise InfrastructureError("model base_url references an unset environment variable")
        if not value.startswith(("http://", "https://")):
            raise ValueError("model base_url must be an HTTP(S) URL")
        return value

    @property
    def endpoint(self) -> str:
        base = self.resolved_base_url
        return base if base.endswith("/chat/completions") else f"{base}/chat/completions"

    def stackplanner_mapping(self) -> dict[str, dict[str, Any]]:
        """Map the one public model config onto every StackPlanner model role."""
        base_url = self.base_url
        if base_url.startswith("${") and base_url.endswith("}"):
            base_url = "$" + base_url[2:-1]
        common = {
            "model": self.model,
            "base_url": base_url,
            "api_key": f"${self.api_key_env}",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout_seconds,
        }
        return {"BASIC_MODEL": dict(common), "REPORT_MODEL": dict(common), "REASONING_MODEL": dict(common)}

    def write_stackplanner_config(self, path: str | os.PathLike[str]) -> None:
        """Write env references only; this method never reads or serializes a key."""
        from pathlib import Path

        Path(path).write_text(
            json.dumps(self.stackplanner_mapping(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True)
class ModelResponse:
    message: Mapping[str, Any]


class OpenAICompatibleClient:
    """Minimal chat-completions client; secret values never enter traces."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def complete(
        self,
        messages: list[Mapping[str, Any]],
        *,
        tools: list[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
    ) -> ModelResponse:
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise InfrastructureError(f"required API key environment variable is unset: {self.config.api_key_env}")
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or self.config.tool_choice
            payload["parallel_tool_calls"] = False
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        http_request = request.Request(
            self.config.endpoint,
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.config.timeout_seconds) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            raise ModelError(f"model endpoint returned HTTP {exc.code}") from exc
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise InfrastructureError(f"model endpoint failed: {type(exc).__name__}") from exc
        try:
            message = decoded["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelError("model response has no assistant message") from exc
        return ModelResponse(message=message)
