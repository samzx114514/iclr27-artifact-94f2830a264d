"""Canonical StackPlanner root executor that produces State 0 candidates."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Mapping

from githarness.domains import get_domain_spec
from githarness.state0 import (
    ExecutionFailure,
    StackPlannerRootResult,
    State0Request,
    assert_public_payload,
)
from githarness.version_interface.protocol import CandidateWorkState

from .runtime import (
    artifact_from_state,
    compile_task_graph,
    default_vendor_root,
    root_plan_from_state,
    root_work_state,
)


GraphFactory = Callable[[str, Path], Any]


class CanonicalStackPlannerRootExecutor:
    """Run one isolated canonical graph without reusing native graph memory."""

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
        assert_public_payload(self.configurable, location="root_configurable")
        self.graph_factory = graph_factory

    def create_state0(
        self,
        candidate: CandidateWorkState,
        request: State0Request,
    ) -> StackPlannerRootResult:
        if request.domain != self.domain:
            raise ValueError("State 0 request domain does not match root executor")
        try:
            return asyncio.run(self._run(candidate, request))
        except Exception as error:
            return StackPlannerRootResult(
                finished=False,
                artifact_path=self.spec.artifact_path,
                artifact="",
                failure=ExecutionFailure(
                    "infrastructure_failure",
                    f"canonical StackPlanner root failed ({type(error).__name__})",
                    retryable=True,
                ),
            )

    async def _run(
        self,
        candidate: CandidateWorkState,
        request: State0Request,
    ) -> StackPlannerRootResult:
        graph = self.graph_factory(self.spec.graph_format, self.vendor_root)
        configurable = {
            "thread_id": f"state0-{candidate.candidate_id}",
            "graph_format": self.spec.graph_format,
            **self.configurable,
        }
        state = {
            "messages": [{"role": "user", "content": request.resolved_requirement}],
            "user_query": request.resolved_requirement,
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
            raise TypeError("canonical StackPlanner root returned a non-object state")
        artifact = artifact_from_state(self.spec, result)
        target = candidate.root / self.spec.artifact_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(artifact, encoding="utf-8")
        trace = ({
            "event": "stackplanner_root_completed",
            "domain": self.domain,
            "graph_format": self.spec.graph_format,
            "artifact_path": self.spec.artifact_path,
        },)
        return StackPlannerRootResult(
            finished=True,
            artifact_path=self.spec.artifact_path,
            artifact=artifact,
            root_plan=root_plan_from_state(result),
            work_state=root_work_state(self.spec, result),
            trace=trace,
        )
