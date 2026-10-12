from __future__ import annotations

import json
from pathlib import Path

import githarness.cli as cli_module
import pytest
from githarness.cli import main
from githarness.errors import ProtocolError
from githarness.integrations.stackplanner import (
    CanonicalStackPlannerRootExecutor,
    RemoteBochaSearchBackend,
)
from githarness.task_native_update import ToolLoopTaskNativeUpdateAgent


def _offline_math_config() -> dict:
    return {
        "task_id": "offline-math-versioning",
        "domain": "math",
        "executor": "offline_math",
        "requirement_resolution": {"max_steps": 4},
        "domain_runtime": {"update_max_steps": 16},
        "turns": [
            {
                "feedback": "Compute 2 + 3.",
                "actions": [{
                    "action": "Commit",
                    "resolved_requirement": "Compute: 2 + 3",
                    "base_version_id": None,
                    "update_hint": "",
                }],
            },
            {
                "feedback": "Use 5 instead of 3.",
                "actions": [
                    {"action": "Inspect", "version_id": "v0"},
                    {
                        "action": "Commit",
                        "resolved_requirement": "Compute: 2 + 5",
                        "base_version_id": "v0",
                        "update_hint": "Replace the second operand with 5.",
                    },
                ],
            },
            {
                "feedback": "Return to the original calculation.",
                "actions": [{"action": "Reuse", "version_id": "v0"}],
            },
        ],
    }


def _write_offline_math_config(path: Path, **updates) -> Path:
    config = _offline_math_config()
    config.update(updates)
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_offline_cli_end_to_end(tmp_path: Path) -> None:
    config = _write_offline_math_config(tmp_path / "config.json")
    assert main(["run-case", "--config", str(config), "--output", str(tmp_path)]) == 0
    result = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert [turn["mode"] for turn in result["turns"]] == ["fresh_solve", "reuse_and_patch", "exact_reuse"]
    assert len(result["versions"]) == 2
    assert all(turn["handoff"]["schema_version"] == "resolved-requirement-handoff-v1" for turn in result["turns"])
    assert result["turns"][1]["handoff"]["selected_base_checkpoint"]["checkpoint_id"] == "checkpoint-v0"
    assert result["turns"][0]["update_result"] is None
    assert result["turns"][1]["update_result"]["schema_version"] == "task-native-update-result-v1"
    assert result["turns"][2]["update_result"] is None
    assert result["step_budgets"] == {
        "git_agent": 4,
        "task_native_update_agent": 16,
    }


def test_online_cli_uses_independent_update_loop_after_stackplanner_root(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({
            "task_id": "math-online",
            "domain": "math",
            "executor": "stackplanner",
            "model": {
                "model": "review-model",
                "base_url": "https://model.example/v1",
                "api_key_env": "REVIEW_KEY",
            },
            "turns": [],
        }),
        encoding="utf-8",
    )
    observed = {}

    def capture(config_value, output, update, root, router, **kwargs):
        observed.update(update=update, root=root, kwargs=kwargs)
        return {"task_id": config_value["task_id"], "turns": []}

    monkeypatch.setattr(cli_module, "_execute_case", capture)
    cli_module.run_case(config, tmp_path / "run")

    assert isinstance(observed["root"], CanonicalStackPlannerRootExecutor)
    assert isinstance(observed["update"], ToolLoopTaskNativeUpdateAgent)
    assert observed["update"].max_steps == 16


def test_cli_propagates_custom_step_budgets(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({
            "task_id": "math-online",
            "domain": "math",
            "executor": "stackplanner",
            "model": {
                "model": "review-model",
                "base_url": "https://model.example/v1",
                "api_key_env": "REVIEW_KEY",
            },
            "requirement_resolution": {"max_steps": 7},
            "domain_runtime": {"update_max_steps": 19},
            "turns": [],
        }),
        encoding="utf-8",
    )
    observed = {}

    def capture(config_value, output, update, root, router, **kwargs):
        observed.update(update=update)
        return {
            "task_id": config_value["task_id"],
            "turns": [],
            "step_budgets": cli_module._step_budgets(config_value),
        }

    monkeypatch.setattr(cli_module, "_execute_case", capture)
    result = cli_module.run_case(config, tmp_path / "run")

    assert observed["update"].max_steps == 19
    assert result["step_budgets"] == {
        "git_agent": 7,
        "task_native_update_agent": 19,
    }


def test_git_agent_budget_limits_cli_resolution(tmp_path: Path) -> None:
    config = _offline_math_config()
    config["requirement_resolution"]["max_steps"] = 1
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ProtocolError, match="within 1 steps"):
        cli_module.run_case(path, tmp_path / "run")


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"requirement_resolution": {"max_steps": 0}}, "requirement_resolution.max_steps"),
        ({"requirement_resolution": {"max_steps": True}}, "requirement_resolution.max_steps"),
        ({"domain_runtime": {"update_max_steps": -1}}, "domain_runtime.update_max_steps"),
        ({"domain_runtime": {"update_max_steps": "16"}}, "domain_runtime.update_max_steps"),
        ({"requirement_resolution": []}, "requirement_resolution must be an object"),
    ],
)
def test_cli_rejects_invalid_step_budgets(config, message: str) -> None:
    with pytest.raises(SystemExit, match=message):
        cli_module._step_budgets(config)


def test_research_cli_injects_one_bocha_backend_into_root_and_update(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "research.json"
    config.write_text(
        json.dumps({
            "task_id": "research-online",
            "domain": "research",
            "executor": "stackplanner",
            "model": {
                "model": "review-model",
                "base_url": "https://model.example/v1",
                "api_key_env": "REVIEW_KEY",
            },
            "domain_runtime": {
                "backend": "remote_bocha",
                "api_key_env": "BOCHA_API_KEY",
                "top_k": 4,
                "timeout": 7,
                "max_attempts": 1,
            },
            "turns": [],
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("BOCHA_API_KEY", "private-value")
    monkeypatch.setenv("SEARCH_API", "unused")
    observed = {}

    def capture(config_value, output, update, root, router, **kwargs):
        observed.update(update=update, root=root, kwargs=kwargs)
        return {"task_id": config_value["task_id"], "turns": []}

    monkeypatch.setattr(cli_module, "_execute_case", capture)
    cli_module.run_case(config, tmp_path / "run")

    backend = observed["kwargs"]["update_runtime_assets"]["search"]
    assert isinstance(backend, RemoteBochaSearchBackend)
    assert "search_backend" not in observed["root"].configurable
    assert observed["kwargs"]["update_runtime_assets"]["capabilities"] == [
        "search(query)"
    ]
    assert "private-value" not in repr(backend)
    assert cli_module.os.environ["SEARCH_API"] == "bocha"


def test_research_cli_requires_bocha_credential(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "research.json"
    config.write_text(
        json.dumps({
            "domain": "research",
            "executor": "stackplanner",
            "model": {
                "model": "review-model",
                "base_url": "https://model.example/v1",
                "api_key_env": "REVIEW_KEY",
            },
            "domain_runtime": {
                "backend": "remote_bocha",
                "api_key_env": "BOCHA_API_KEY",
            },
            "turns": [],
        }),
        encoding="utf-8",
    )
    monkeypatch.delenv("BOCHA_API_KEY", raising=False)

    with pytest.raises(SystemExit, match="BOCHA_API_KEY is not set"):
        cli_module.run_case(config, tmp_path / "run")


def test_deferred_schedule_materializes_only_final_necessary_path(tmp_path: Path) -> None:
    source = _write_offline_math_config(tmp_path / "eager.json")
    config_path = _write_offline_math_config(
        tmp_path / "deferred.json",
        execution_schedule="deferred_necessary_path",
    )
    output = tmp_path / "run"
    eager_output = tmp_path / "eager"

    assert main(["run-case", "--config", str(config_path), "--output", str(output)]) == 0
    result = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert main(["run-case", "--config", str(source), "--output", str(eager_output)]) == 0
    eager = json.loads((eager_output / "run.json").read_text(encoding="utf-8"))

    assert result["execution_schedule"] == "deferred_necessary_path"
    assert result["deferred_target_turns"] == [2]
    assert [item["version_id"] for item in result["planned_versions"]] == ["v0", "v1"]
    assert [item["version_id"] for item in result["versions"]] == ["v0"]
    assert [turn["materialized"] for turn in result["turns"]] == [True, False, True]
    assert result["turns"][0]["mode"] == "fresh_solve"
    assert result["turns"][1]["handoff_events"][0]["observation"]["public_output"]
    assert result["turns"][1]["update_result"] is None
    assert result["turns"][2]["mode"] == "exact_reuse"
    assert result["turns"][2]["final_response"] == result["turns"][0]["final_response"]
    assert result["planned_versions"] == eager["versions"]
    assert result["turns"][2]["final_response"] == eager["turns"][2]["final_response"]
    assert eager["turns"][1]["update_result"] is not None
