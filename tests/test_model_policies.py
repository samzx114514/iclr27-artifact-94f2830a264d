import json
from io import BytesIO

from githarness.branch_execution import OpenAIExecutionRouter
from githarness.model import ModelConfig, ModelResponse, OpenAICompatibleClient
from githarness.requirement_resolution import Commit, OpenAIGitPolicy
from githarness.versioning import Checkpoint, WorkState


class FakeClient:
    def __init__(self, name: str, arguments: dict[str, object]) -> None:
        self.name = name
        self.arguments = arguments

    def complete(self, messages, *, tools=None, tool_choice=None):
        return ModelResponse(
            message={
                "tool_calls": [
                    {"function": {"name": self.name, "arguments": json.dumps(self.arguments)}}
                ]
            },
        )


def test_git_policy_parses_one_native_commit() -> None:
    policy = OpenAIGitPolicy(
        FakeClient(
            "Commit",
            {"resolved_requirement": "Compute 2 + 2", "base_version_id": None, "update_hint": ""},
        )
    )
    action = policy.choose({"feedback": "Compute 2 + 2"})
    assert action == Commit("Compute 2 + 2", None, "")


def test_router_cannot_return_exact_reuse() -> None:
    router = OpenAIExecutionRouter(
        FakeClient("route_execution", {"mode": "reuse_and_patch", "reason": "compatible base"})
    )
    checkpoint = Checkpoint("c0", "v0", WorkState({"answer.md": "old"}), receipt={"passed": True})
    decision = router.decide(
        requirement="new",
        base_requirement="old",
        update_hint="change one value",
        base_checkpoint=checkpoint,
    )
    assert decision.mode.value == "reuse_and_patch"


def test_stackplanner_config_contains_env_reference_not_secret(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TEST_REVIEW_KEY", "do-not-serialize")
    config = ModelConfig("model-name", "${TEST_BASE_URL}", "TEST_REVIEW_KEY")
    target = tmp_path / "stackplanner.json"
    config.write_stackplanner_config(target)
    text = target.read_text(encoding="utf-8")
    assert "do-not-serialize" not in text
    assert "$TEST_REVIEW_KEY" in text
    assert "$TEST_BASE_URL" in text


def test_model_config_supports_provider_compatible_tool_choice() -> None:
    config = ModelConfig.from_mapping({
        "model": "thinking-model",
        "base_url": "https://model.example/v1",
        "api_key_env": "TEST_KEY",
        "tool_choice": "auto",
    })
    assert config.tool_choice == "auto"


class _HTTPResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_model_client_sends_configured_auto_tool_choice(monkeypatch) -> None:
    observed = {}

    def fake_urlopen(call, timeout):
        observed["payload"] = json.loads(call.data)
        observed["timeout"] = timeout
        return _HTTPResponse(json.dumps({
            "choices": [{"message": {"tool_calls": []}}],
        }).encode())

    monkeypatch.setenv("TEST_KEY", "not-serialized")
    monkeypatch.setattr("githarness.model.request.urlopen", fake_urlopen)
    client = OpenAICompatibleClient(ModelConfig(
        "thinking-model",
        "https://model.example/v1",
        "TEST_KEY",
        tool_choice="auto",
    ))
    client.complete(
        [{"role": "user", "content": "call done"}],
        tools=[{
            "type": "function",
            "function": {
                "name": "done",
                "parameters": {"type": "object", "properties": {}},
            },
        }],
    )

    assert observed["payload"]["tool_choice"] == "auto"
    assert observed["payload"]["parallel_tool_calls"] is False
    assert "not-serialized" not in json.dumps(observed["payload"])
