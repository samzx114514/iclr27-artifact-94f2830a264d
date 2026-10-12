"""Canonical StackPlanner-backed Task-Native Update Agent."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Mapping

from githarness.domains import get_domain_spec
from githarness.state0 import assert_public_payload
from githarness.task_native_update import (
    UpdateAgentRequest,
    UpdateAgentResult,
    UpdateExecutionFailure,
)
from githarness.task_native_update.runtime import candidate_work_state

from .runtime import (
    artifact_from_state,
    compile_task_graph,
    default_vendor_root,
    root_work_state,
)


GraphFactory = Callable[[str, Path], Any]


class StackPlannerTaskNativeUpdateAgent:
    """Run one isolated candidate update through the domain's canonical graph."""

    def __init__(
        self,
        *,
        domain: str,
        vendor_root: Path | None = None,
        configurable: Mapping[str, Any] | None = None,
        graph_factory: GraphFactory = compile_task_graph,
    ) -> None:
        self.spec = get_domain_spec(domain)
        self.domain = self.spec.name
        self.vendor_root = Path(vendor_root or default_vendor_root())
        self.configurable = dict(configurable or {})
        assert_public_payload(self.configurable, location="update_configurable")
        self.graph_factory = graph_factory

    def execute(self, request: UpdateAgentRequest) -> UpdateAgentResult:
        if request.domain != self.domain:
            raise ValueError("Update Agent request domain does not match executor")
        try:
            return asyncio.run(self._run(request))
        except Exception as error:
            return UpdateAgentResult(
                False,
                self.spec.artifact_path,
                "",
                candidate_work_state(request),
                failure=UpdateExecutionFailure(
                    "infrastructure_failure",
                    f"Task-Native Update Agent failed ({type(error).__name__})",
                    retryable=True,
                ),
            )

    async def _run(self, request: UpdateAgentRequest) -> UpdateAgentResult:
        graph = self.graph_factory(self.spec.graph_format, self.vendor_root)
        prompt = request.resolved_requirement
        if request.update_hint:
            prompt += f"\n\nNon-authoritative update hint:\n{request.update_hint}"
        configurable = {
            "thread_id": f"update-{request.candidate.candidate_id}",
            "graph_format": self.spec.graph_format,
            **self.configurable,
        }
        inherited = request.candidate.metadata.get("task_native_work_state")
        if not isinstance(inherited, Mapping):
            inherited = request.candidate.metadata.get("state0_root_work_state")
        inherited = dict(inherited) if isinstance(inherited, Mapping) else {}
        assert_public_payload(inherited, location="inherited_update_work_state")
        state: dict[str, Any] = {
            **inherited,
            "messages": [{"role": "user", "content": prompt}],
            "user_query": prompt,
            "final_report": "",
            "auto_accepted_plan": True,
            "enable_background_investigation": False,
            "skip_perception": True,
        }
        result = await graph.ainvoke(
            state,
            config={
                "configurable": configurable,
                "recursion_limit": int(configurable.get("recursion_limit", 200)),
            },
        )
        if not isinstance(result, Mapping):
            raise TypeError("StackPlanner update returned a non-object state")
        artifact = artifact_from_state(self.spec, result)
        target = request.candidate.root / self.spec.artifact_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(artifact, encoding="utf-8")
        projected_work_state = dict(root_work_state(self.spec, result))
        assert_public_payload(projected_work_state, location="updated_native_work_state")
        request.candidate.metadata["task_native_work_state"] = projected_work_state
        trace = ({
            "event": "task_native_update_completed",
            "domain": self.domain,
            "graph_format": self.spec.graph_format,
            "artifact_path": self.spec.artifact_path,
        },)
        return UpdateAgentResult(
            True,
            self.spec.artifact_path,
            artifact,
            candidate_work_state(request),
            trace=trace,
        )


# Compatibility import name.  The object now implements the typed Update Agent
# protocol and is no longer a filesystem-interface callback.
StackPlannerUpdateAgent = StackPlannerTaskNativeUpdateAgent
