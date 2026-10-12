"""GitHarness command-line interface."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .branch_execution import BranchExecutionRuntime, ConservativeExecutionRouter, OpenAIExecutionRouter
from .domains import get_domain_spec
from .execution_schedule import (
    ExecutionSchedule,
    RequirementExecutionPlan,
    parse_execution_schedule,
)
from .integrations.stackplanner import (
    CanonicalStackPlannerRootExecutor,
    CodeStackPlannerRootExecutor,
    CodeTestbedTaskNativeUpdateAgent,
    DEFAULT_BOCHA_ENDPOINT,
    JsonlSearchBackend,
    RemoteBochaSearchBackend,
    RemoteBrowseCompSearchBackend,
    image_name_for_instance,
)
from .model import ModelConfig, OpenAICompatibleClient
from .offline import (
    DeterministicArtifactRootExecutor,
    DeterministicMathRootExecutor,
    deterministic_artifact_update,
    deterministic_math_update,
)
from .requirement_resolution import (
    Commit,
    GitAgent,
    Inspect,
    OpenAIGitPolicy,
    Reuse,
    CheckpointBinding,
    handoff_to_dict,
)
from .requirement_resolution.agent import ScriptedGitPolicy
from .version_interface import FileWorkspaceVersionInterface
from .versioning import VersionGraph
from .versioning.models import digest
from .state0 import state0_to_dict
from .task_native_update import (
    CallableTaskNativeUpdateAgent,
    OpenAITaskNativePolicy,
    ToolLoopTaskNativeUpdateAgent,
    update_result_to_dict,
)


def _load(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as error:
            raise SystemExit("YAML config requires: pip install -e '.[yaml]'") from error
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit("config must contain an object")
    return value


def _action(value: dict[str, Any]):
    kind = value.get("action")
    if kind == "Inspect":
        return Inspect(str(value["version_id"]))
    if kind == "Commit":
        return Commit(
            str(value["resolved_requirement"]),
            str(value["base_version_id"]) if value.get("base_version_id") is not None else None,
            str(value.get("update_hint") or ""),
        )
    if kind == "Reuse":
        return Reuse(str(value["version_id"]))
    raise SystemExit(f"unknown scripted action: {kind}")


def _required_env_reference(value: Any, *, setting: str) -> str:
    if not isinstance(value, str) or not value.startswith("${") or not value.endswith("}"):
        raise SystemExit(f"{setting} must be an environment reference such as ${{SEARCH_URL}}")
    name = value[2:-1]
    if not name or not name.replace("_", "A").isalnum():
        raise SystemExit(f"invalid environment variable reference for {setting}")
    resolved = os.environ.get(name)
    if not resolved:
        raise SystemExit(f"required environment variable {name} is not set")
    return resolved


def _step_budgets(config: Mapping[str, Any]) -> dict[str, int]:
    resolution = config.get("requirement_resolution", {})
    runtime = config.get("domain_runtime", {})
    resolution = {} if resolution is None else resolution
    runtime = {} if runtime is None else runtime
    if not isinstance(resolution, Mapping):
        raise SystemExit("requirement_resolution must be an object")
    if not isinstance(runtime, Mapping):
        raise SystemExit("domain_runtime must be an object")

    def positive(value: Any, *, setting: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise SystemExit(f"{setting} must be a positive integer")
        return value

    return {
        "git_agent": positive(
            resolution.get("max_steps", 4),
            setting="requirement_resolution.max_steps",
        ),
        "task_native_update_agent": positive(
            runtime.get("update_max_steps", 16),
            setting="domain_runtime.update_max_steps",
        ),
    }


def _search_backend(runtime_config: dict[str, Any], config_path: Path, *, domain: str) -> Any:
    default_backend = "remote_bocha" if domain == "research" else "local_jsonl"
    backend = str(runtime_config.get("backend") or default_backend)
    if backend == "local_jsonl":
        corpus_value = runtime_config.get("corpus")
        if not isinstance(corpus_value, str):
            raise SystemExit("Search local_jsonl backend requires domain_runtime.corpus")
        corpus = (config_path.parent / corpus_value).resolve()
        return JsonlSearchBackend(corpus, top_k=int(runtime_config.get("top_k", 8)))
    if backend == "remote_browsecomp":
        if domain != "search":
            raise SystemExit("remote_browsecomp is available only for Search")
        base_url = _required_env_reference(runtime_config.get("base_url"), setting="domain_runtime.base_url")
        key_name = runtime_config.get("api_key_env")
        if not isinstance(key_name, str) or not key_name:
            raise SystemExit("Search remote_browsecomp backend requires domain_runtime.api_key_env")
        return RemoteBrowseCompSearchBackend(
            base_url,
            api_key=os.environ.get(key_name, ""),
            top_k=int(runtime_config.get("top_k", 5)),
            snippet_max_tokens=int(runtime_config.get("snippet_max_tokens", 512)),
            timeout=float(runtime_config.get("timeout", 120)),
        )
    if backend == "remote_bocha":
        if domain != "research":
            raise SystemExit("remote_bocha is available only for Research")
        key_name = runtime_config.get("api_key_env")
        if key_name != "BOCHA_API_KEY":
            raise SystemExit("Research remote_bocha requires api_key_env: BOCHA_API_KEY")
        api_key = os.environ.get(key_name, "")
        if not api_key:
            raise SystemExit("required environment variable BOCHA_API_KEY is not set")
        endpoint = str(runtime_config.get("endpoint") or DEFAULT_BOCHA_ENDPOINT)
        if endpoint != DEFAULT_BOCHA_ENDPOINT:
            raise SystemExit("Research remote_bocha endpoint must use the public pinned endpoint")
        os.environ["SEARCH_API"] = "bocha"
        return RemoteBochaSearchBackend(
            api_key=api_key,
            endpoint=endpoint,
            top_k=int(runtime_config.get("top_k", 8)),
            timeout=float(runtime_config.get("timeout", 60)),
            max_attempts=int(runtime_config.get("max_attempts", 3)),
            retry_backoff=float(runtime_config.get("retry_backoff", 3)),
        )
    raise SystemExit(
        "domain_runtime.backend must be local_jsonl, remote_browsecomp, or remote_bocha"
    )


def run_case(config_path: Path, output: Path) -> dict[str, Any]:
    config = _load(config_path)
    domain = str(config.get("domain"))
    get_domain_spec(domain)
    step_budgets = _step_budgets(config)
    output.mkdir(parents=True, exist_ok=True)
    executor = config.get("executor")
    if executor == "offline_math":
        if domain != "math":
            raise SystemExit("executor=offline_math requires domain=math")
        return _execute_case(
            config,
            output,
            deterministic_math_update,
            DeterministicMathRootExecutor(),
            ConservativeExecutionRouter(),
            scripted=True,
        )
    if executor == "offline_fixture":
        if config.get("synthetic_fixture_only") is not True:
            raise SystemExit(
                "executor=offline_fixture requires synthetic_fixture_only=true"
            )
        fixture = config.get("fixture_artifact")
        if not isinstance(fixture, dict):
            raise SystemExit("executor=offline_fixture requires fixture_artifact")
        artifact_path = fixture.get("path")
        content = fixture.get("content")
        spec = get_domain_spec(domain)
        if artifact_path != spec.artifact_path or not isinstance(content, str):
            raise SystemExit(
                "offline fixture artifact must match the domain artifact path"
            )
        update = deterministic_artifact_update(artifact_path, content)
        return _execute_case(
            config,
            output,
            update,
            DeterministicArtifactRootExecutor(artifact_path, content),
            ConservativeExecutionRouter(),
            scripted=True,
        )
    if executor != "stackplanner":
        raise SystemExit(
            "executor must be offline_math, offline_fixture, or stackplanner"
        )

    raw_model = config.get("model")
    if isinstance(raw_model, str):
        model_path = (config_path.parent / raw_model).resolve()
        raw_model = _load(model_path)
    if not isinstance(raw_model, dict):
        raise SystemExit("executor=stackplanner requires a model object or model config path")
    model_config = ModelConfig.from_mapping(raw_model)
    stackplanner_config = output / "runtime" / "stackplanner-model.json"
    stackplanner_config.parent.mkdir(parents=True, exist_ok=True)
    model_config.write_stackplanner_config(stackplanner_config)
    os.environ["STACKPLANNER_CONFIG_PATH"] = str(stackplanner_config.resolve())
    runtime_config = dict(config.get("domain_runtime") or {})
    configurable: dict[str, Any] = dict(runtime_config.get("stackplanner") or {})
    update_assets: dict[str, Any] = {
        "capabilities": list(get_domain_spec(domain).public_capabilities)
    }
    if domain in {"search", "research"}:
        search_backend = _search_backend(runtime_config, config_path, domain=domain)
        if domain == "search":
            configurable["search_backend"] = search_backend
        update_assets["search"] = search_backend
    client = OpenAICompatibleClient(model_config)
    tool_loop = ToolLoopTaskNativeUpdateAgent(
        OpenAITaskNativePolicy(client),
        max_steps=step_budgets["task_native_update_agent"],
    )
    if domain == "code":
        image = runtime_config.get("image")
        instance_id = runtime_config.get("instance_id")
        if not isinstance(image, str):
            if not isinstance(instance_id, str):
                raise SystemExit("Code requires domain_runtime.instance_id or image")
            image = image_name_for_instance(instance_id)
        update = CodeTestbedTaskNativeUpdateAgent(image=image, delegate=tool_loop)
        root_executor = CodeStackPlannerRootExecutor(image=image, configurable=configurable)
    else:
        update = tool_loop
        root_executor = CanonicalStackPlannerRootExecutor(
            domain=domain,
            configurable=configurable,
        )
    return _execute_case(
        config,
        output,
        update,
        root_executor,
        OpenAIExecutionRouter(client),
        scripted=False,
        client=client,
        update_runtime_assets=update_assets,
    )


def _execute_case(
    config: dict[str, Any],
    output: Path,
    update: Any,
    root_executor: Any,
    router: Any,
    *,
    scripted: bool,
    client: OpenAICompatibleClient | None = None,
    update_runtime_assets: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    domain = str(config["domain"])
    step_budgets = _step_budgets(config)
    spec = get_domain_spec(domain)
    graph = VersionGraph()
    interface = FileWorkspaceVersionInterface(output / "workspace", spec.finalizer)
    runtime = BranchExecutionRuntime(
        graph=graph,
        checkpoints={},
        bindings={},
        router=router,
        interface=interface,
        domain=domain,
        root_executor=root_executor,
        update_agent=(
            update if hasattr(update, "execute") else CallableTaskNativeUpdateAgent(update)
        ),
        public_runtime_assets={"capabilities": list(spec.public_capabilities)},
        update_runtime_assets=update_runtime_assets,
    )
    try:
        schedule = parse_execution_schedule(config.get("execution_schedule"))
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if schedule is ExecutionSchedule.DEFERRED_NECESSARY_PATH:
        return _execute_case_deferred(
            config=config,
            output=output,
            runtime=runtime,
            scripted=scripted,
            client=client,
            step_budgets=step_budgets,
        )
    rows = []
    version_outputs: dict[str, str] = {}
    for turn_index, turn in enumerate(config.get("turns") or []):
        if scripted:
            policy = ScriptedGitPolicy([_action(value) for value in turn.get("actions") or []])
        else:
            if client is None:
                raise RuntimeError("online execution requires a model client")
            policy = OpenAIGitPolicy(client)
        handoff = GitAgent(policy, max_steps=step_budgets["git_agent"]).resolve(
            feedback=str(turn.get("feedback") or ""),
            source_turn=turn_index,
            graph=graph,
            version_outputs=version_outputs,
            checkpoints=runtime.checkpoints,
            checkpoint_bindings=runtime.bindings,
            task_id=str(config.get("task_id") or "offline-task"),
            domain=domain,
        )
        outcome = runtime.execute(
            task_id=str(config.get("task_id") or "offline-task"),
            source_turn=turn_index,
            handoff=handoff,
        )
        if outcome.final_response:
            version_outputs[outcome.version_id] = outcome.final_response
        rows.append(
            {
                "turn": turn_index,
                "version_id": outcome.version_id,
                "checkpoint_id": outcome.checkpoint_id,
                "mode": outcome.mode.value,
                "final_response": outcome.final_response,
                "work_state_digest": outcome.work_state.digest,
                "handoff": handoff_to_dict(handoff),
                "handoff_events": list(handoff.events),
                "state0": state0_to_dict(outcome.state0) if outcome.state0 else None,
                "update_result": (
                    update_result_to_dict(outcome.update_result)
                    if outcome.update_result else None
                ),
            }
        )
    result = {
        "schema_version": 1,
        "task_id": config.get("task_id"),
        "domain": domain,
        "executor": config.get("executor"),
        "execution_schedule": schedule.value,
        "step_budgets": step_budgets,
        "versions": list(graph.view()),
        "turns": rows,
    }
    (output / "run.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def _execute_case_deferred(
    *,
    config: dict[str, Any],
    output: Path,
    runtime: BranchExecutionRuntime,
    scripted: bool,
    client: OpenAICompatibleClient | None,
    step_budgets: Mapping[str, int],
) -> dict[str, Any]:
    turns = list(config.get("turns") or [])
    raw_targets = config.get("deferred_target_turns")
    if raw_targets is None:
        targets = [len(turns) - 1] if turns else []
    elif not isinstance(raw_targets, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in raw_targets
    ):
        raise SystemExit("deferred_target_turns must be an array of turn indexes")
    else:
        targets = list(dict.fromkeys(raw_targets))
    if any(item < 0 or item >= len(turns) for item in targets):
        raise SystemExit("deferred_target_turns contains an out-of-range turn index")
    task_id = str(config.get("task_id") or "offline-task")
    domain = str(config["domain"])
    plan = RequirementExecutionPlan()
    version_outputs: dict[str, str] = {}
    outcomes: dict[str, Any] = {}
    execution_handoffs: dict[str, Any] = {}

    def materialize(version_id: str) -> str | None:
        if version_id in outcomes:
            return outcomes[version_id].final_response
        handoff = plan.handoffs_by_version.get(version_id)
        if handoff is None:
            raise RuntimeError(f"planned version has no Commit handoff: {version_id}")
        bootstrap = plan.bootstrap_version_id
        if bootstrap is not None and bootstrap != version_id and bootstrap not in outcomes:
            materialize(bootstrap)
        if handoff.selected_base_version_id is not None:
            materialize(handoff.selected_base_version_id)
        bound = _bind_planned_handoff(handoff, runtime)
        outcome = runtime.execute(
            task_id=task_id,
            source_turn=_source_turn_for(plan, version_id),
            handoff=bound,
        )
        outcomes[version_id] = outcome
        execution_handoffs[version_id] = bound
        if outcome.final_response:
            version_outputs[version_id] = outcome.final_response
        return outcome.final_response

    for turn_index, turn in enumerate(turns):
        policy = _git_policy(turn, scripted=scripted, client=client)
        handoff = GitAgent(policy, max_steps=step_budgets["git_agent"]).resolve(
            feedback=str(turn.get("feedback") or ""),
            source_turn=turn_index,
            graph=plan.graph,
            version_outputs=version_outputs,
            public_output_loader=materialize,
            task_id=task_id,
            domain=domain,
        )
        plan.record(turn_index, handoff)

    for version_id in plan.necessary_versions(targets):
        materialize(version_id)

    targeted = set(targets)
    reuse_outcomes: dict[int, Any] = {}
    reuse_execution_handoffs: dict[int, Any] = {}
    for turn_index in targets:
        planned = plan.turns[turn_index]
        if planned.handoff.reused_version_id is None:
            continue
        bound = _bind_planned_handoff(planned.handoff, runtime)
        reuse_outcomes[turn_index] = runtime.execute(
            task_id=task_id,
            source_turn=turn_index,
            handoff=bound,
        )
        reuse_execution_handoffs[turn_index] = bound

    rows = []
    for turn_index, planned in enumerate(plan.turns):
        if planned.handoff.reused_version_id is not None:
            outcome = reuse_outcomes.get(turn_index)
            bound = reuse_execution_handoffs.get(turn_index)
        else:
            outcome = outcomes.get(planned.active_version_id)
            bound = execution_handoffs.get(planned.active_version_id)
        rows.append(
            _deferred_row(
                turn_index=turn_index,
                planned=planned,
                outcome=outcome,
                execution_handoff=bound,
                targeted=turn_index in targeted,
            )
        )

    if plan.turns:
        final_active = plan.turns[-1].active_version_id
        if final_active in runtime.bindings:
            runtime.graph.activate_existing(final_active)
    result = {
        "schema_version": 1,
        "task_id": config.get("task_id"),
        "domain": domain,
        "executor": config.get("executor"),
        "execution_schedule": ExecutionSchedule.DEFERRED_NECESSARY_PATH.value,
        "step_budgets": dict(step_budgets),
        "deferred_target_turns": targets,
        "planned_versions": list(plan.graph.view()),
        "versions": list(runtime.graph.view()),
        "turns": rows,
    }
    (output / "run.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _git_policy(
    turn: Mapping[str, Any],
    *,
    scripted: bool,
    client: OpenAICompatibleClient | None,
):
    if scripted:
        return ScriptedGitPolicy([_action(value) for value in turn.get("actions") or []])
    if client is None:
        raise RuntimeError("online execution requires a model client")
    return OpenAIGitPolicy(client)


def _source_turn_for(plan: RequirementExecutionPlan, version_id: str) -> int:
    return plan.graph.get(version_id).source_turn


def _bind_planned_handoff(
    handoff,
    runtime: BranchExecutionRuntime,
):
    base_id = handoff.selected_base_version_id
    if base_id is None:
        return handoff
    checkpoint_id = runtime.bindings.get(base_id)
    checkpoint = runtime.checkpoints.get(checkpoint_id or "")
    if checkpoint is None:
        raise RuntimeError(f"planned base was not materialized: {base_id}")
    binding = CheckpointBinding.from_checkpoint(checkpoint)
    identity = None
    if handoff.reused_version_id is not None:
        version = runtime.graph.get(base_id)
        identity = digest({
            "version_id": version.version_id,
            "requirement_digest": version.requirement_state.digest,
            "checkpoint_digest": checkpoint.digest,
        })
    return replace(
        handoff,
        selected_base_checkpoint=binding,
        exact_reuse_identity=identity,
    )


def _deferred_row(
    *,
    turn_index: int,
    planned,
    outcome,
    execution_handoff,
    targeted: bool,
) -> dict[str, Any]:
    handoff = planned.handoff
    if outcome is None:
        return {
            "turn": turn_index,
            "targeted": targeted,
            "materialized": False,
            "version_id": planned.active_version_id,
            "checkpoint_id": None,
            "mode": None,
            "final_response": None,
            "work_state_digest": None,
            "handoff": handoff_to_dict(handoff),
            "execution_handoff": None,
            "handoff_events": list(handoff.events),
            "state0": None,
            "update_result": None,
        }
    return {
        "turn": turn_index,
        "targeted": targeted,
        "materialized": True,
        "version_id": outcome.version_id,
        "checkpoint_id": outcome.checkpoint_id,
        "mode": outcome.mode.value,
        "final_response": outcome.final_response,
        "work_state_digest": outcome.work_state.digest,
        "handoff": handoff_to_dict(handoff),
        "execution_handoff": handoff_to_dict(execution_handoff),
        "handoff_events": list(handoff.events),
        "state0": state0_to_dict(outcome.state0) if outcome.state0 else None,
        "update_result": (
            update_result_to_dict(outcome.update_result)
            if outcome.update_result else None
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="githarness", description="Git-style version tracking for evolving requirements")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-case", help="run one case")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, default=Path(".runs/offline"))
    inspect = sub.add_parser("inspect-version", help="print a stored run's version graph")
    inspect.add_argument("--run-dir", type=Path, required=True)
    trace = sub.add_parser("inspect-trace", help="print a stored run's turn trace")
    trace.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run-case":
        result = run_case(args.config, args.output)
        print(json.dumps({"task_id": result["task_id"], "turns": len(result["turns"]), "output": str(args.output / "run.json")}, ensure_ascii=False))
        return 0
    payload = json.loads((args.run_dir / "run.json").read_text(encoding="utf-8"))
    key = "versions" if args.command == "inspect-version" else "turns"
    print(json.dumps(payload[key], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
