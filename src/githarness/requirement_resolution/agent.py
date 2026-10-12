"""Requirement Resolution through a short Inspect/Commit/Reuse loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from githarness.errors import ProtocolError
from githarness.versioning.graph import VersionGraph
from githarness.versioning.models import Checkpoint, digest

from .actions import Commit, GitAction, Inspect, Reuse
from .context import build_git_agent_context, inspect_version_context
from .handoff import CheckpointBinding, VersionHandoff


class GitPolicy(Protocol):
    def choose(self, context: dict[str, Any]) -> GitAction: ...


@dataclass
class ScriptedGitPolicy:
    actions: list[GitAction]

    def choose(self, context: dict[str, Any]) -> GitAction:
        if not self.actions:
            raise ProtocolError("scripted Git policy has no remaining action")
        return self.actions.pop(0)


@dataclass
class GitAgent:
    policy: GitPolicy
    max_steps: int = 4

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or self.max_steps < 1
        ):
            raise ValueError("Git Agent max_steps must be a positive integer")

    def resolve(
        self,
        *,
        feedback: str,
        source_turn: int,
        graph: VersionGraph,
        version_outputs: dict[str, str] | None = None,
        checkpoints: Mapping[str, Checkpoint] | None = None,
        checkpoint_bindings: Mapping[str, str] | None = None,
        public_output_loader: Callable[[str], str | None] | None = None,
        task_id: str | None = None,
        domain: str | None = None,
    ) -> VersionHandoff:
        if not feedback.strip():
            raise ValueError("feedback must be non-empty")
        outputs = dict(version_outputs or {})
        observations: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        inspected = False
        require_binding = checkpoints is not None or checkpoint_bindings is not None
        for step in range(self.max_steps):
            context = build_git_agent_context(
                feedback=feedback,
                source_turn=source_turn,
                graph=graph,
                observations=observations,
                domain=domain,
            )
            try:
                action = self.policy.choose(context)
            except ProtocolError as error:
                events.append({
                    "step": step,
                    "action": "Invalid",
                    "status": "error",
                    "code": "invalid_action",
                    "message": str(error),
                })
                observations.append({"ok": False, "code": "invalid_action"})
                continue
            if isinstance(action, Inspect):
                if inspected:
                    observation = {"ok": False, "code": "inspect_limit"}
                    observations.append(observation)
                    events.append({
                        "step": step,
                        "action": "Inspect",
                        "version_id": action.version_id,
                        "status": "error",
                        "code": "inspect_limit",
                    })
                    continue
                try:
                    public_output = outputs.get(action.version_id)
                    if public_output is None and public_output_loader is not None:
                        public_output = public_output_loader(action.version_id)
                        if public_output is not None:
                            outputs[action.version_id] = public_output
                    observation = inspect_version_context(
                        graph,
                        action.version_id,
                        public_output=public_output,
                    )
                except KeyError:
                    observations.append({"ok": False, "code": "unknown_version"})
                    events.append({
                        "step": step,
                        "action": "Inspect",
                        "version_id": action.version_id,
                        "status": "error",
                        "code": "unknown_version",
                    })
                    continue
                inspected = True
                observations.append(observation)
                events.append({
                    "step": step,
                    "action": "Inspect",
                    "version_id": action.version_id,
                    "status": "ok",
                    "observation": observation,
                })
                continue
            if isinstance(action, Reuse):
                try:
                    version = graph.get(action.version_id)
                    binding = _binding_for(
                        version.version_id,
                        version.checkpoint_id,
                        checkpoints,
                        checkpoint_bindings,
                        required=require_binding,
                    )
                except (KeyError, ProtocolError):
                    events.append({
                        "step": step,
                        "action": "Reuse",
                        "version_id": action.version_id,
                        "status": "error",
                        "code": "invalid_base",
                    })
                    observations.append({"ok": False, "code": "invalid_base"})
                    continue
                events.append({
                    "step": step,
                    "action": "Reuse",
                    "version_id": action.version_id,
                    "status": "ok",
                })
                return VersionHandoff(
                    resolved_requirement=version.requirement_state.text,
                    selected_base_version_id=version.version_id,
                    target_version_id=version.version_id,
                    update_hint="",
                    reused_version_id=version.version_id,
                    events=tuple(events),
                    task_id=task_id,
                    domain=domain,
                    selected_base_checkpoint=binding,
                    exact_reuse_identity=(
                        digest({
                            "version_id": version.version_id,
                            "requirement_digest": version.requirement_state.digest,
                            "checkpoint_digest": binding.checkpoint_digest,
                        })
                        if binding else None
                    ),
                )
            if isinstance(action, Commit):
                requirement = action.resolved_requirement.strip()
                if not requirement:
                    raise ProtocolError("Commit requires a complete resolved requirement")
                binding = None
                if action.base_version_id is not None:
                    try:
                        base = graph.get(action.base_version_id)
                        binding = _binding_for(
                            base.version_id,
                            base.checkpoint_id,
                            checkpoints,
                            checkpoint_bindings,
                            required=require_binding,
                        )
                    except (KeyError, ProtocolError):
                        events.append({
                            "step": step,
                            "action": "Commit",
                            "base_version_id": action.base_version_id,
                            "status": "error",
                            "code": "invalid_base",
                        })
                        observations.append({"ok": False, "code": "invalid_base"})
                        continue
                target = graph.propose_version_id()
                events.append(
                    {
                        "step": step,
                        "action": "Commit",
                        "base_version_id": action.base_version_id,
                        "target_version_id": target,
                        "status": "ok",
                    }
                )
                return VersionHandoff(
                    resolved_requirement=requirement,
                    selected_base_version_id=action.base_version_id,
                    target_version_id=target,
                    update_hint=action.update_hint.strip(),
                    events=tuple(events),
                    task_id=task_id,
                    domain=domain,
                    selected_base_checkpoint=binding,
                )
            raise ProtocolError(f"unsupported Git Agent action: {action!r}")
        raise ProtocolError(f"Git Agent did not Commit or Reuse within {self.max_steps} steps")


def _binding_for(
    version_id: str,
    checkpoint_id: str,
    checkpoints: Mapping[str, Checkpoint] | None,
    bindings: Mapping[str, str] | None,
    *,
    required: bool,
) -> CheckpointBinding | None:
    if not required:
        return None
    if checkpoints is None or bindings is None:
        raise ProtocolError("accepted checkpoint maps must be supplied together")
    bound_id = bindings.get(version_id)
    checkpoint = checkpoints.get(bound_id or "")
    if bound_id != checkpoint_id or checkpoint is None or checkpoint.version_id != version_id:
        raise ProtocolError(f"version has no matching accepted checkpoint: {version_id}")
    return CheckpointBinding.from_checkpoint(checkpoint)
