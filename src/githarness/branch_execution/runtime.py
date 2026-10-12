"""Atomic Branch Execution over immutable accepted versions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from githarness.errors import (
    InfrastructureError,
    ProtocolError,
    RootExecutionError,
    RuntimeFinalizationError,
)
from githarness.requirement_resolution.handoff import VersionHandoff
from githarness.state0 import (
    StackPlannerRootExecutor,
    StackPlannerRootResult,
    State0,
    State0Candidate,
    State0Request,
    accept_state0,
    assert_public_payload,
)
from githarness.task_native_update import (
    TaskNativeUpdateAgent,
    UpdateAgentRequest,
    UpdateAgentResult,
)
from githarness.version_interface.protocol import ExecutionResult, UnifiedVersionInterface
from githarness.versioning.graph import VersionGraph
from githarness.versioning.models import Checkpoint, RequirementState, Version, WorkState, digest

from .modes import ExecutionMode
from .router import ExecutionRouter


@dataclass(frozen=True)
class ExecutionOutcome:
    finished: bool
    mode: ExecutionMode
    version_id: str
    checkpoint_id: str
    work_state: WorkState
    final_response: str
    receipt: Mapping[str, Any]
    state0: State0 | None = None
    update_result: UpdateAgentResult | None = None


@dataclass
class BranchExecutionRuntime:
    graph: VersionGraph
    checkpoints: dict[str, Checkpoint]
    bindings: dict[str, str]
    router: ExecutionRouter
    interface: UnifiedVersionInterface
    domain: str
    root_executor: StackPlannerRootExecutor
    update_agent: TaskNativeUpdateAgent
    public_runtime_assets: Mapping[str, Any] = field(default_factory=dict)
    update_runtime_assets: Mapping[str, Any] | None = None
    state0s: dict[str, State0] = field(default_factory=dict)

    def execute(
        self,
        *,
        task_id: str,
        source_turn: int,
        handoff: VersionHandoff,
    ) -> ExecutionOutcome:
        if handoff.task_id is not None and handoff.task_id != task_id:
            raise ProtocolError("Git Agent handoff task identity mismatch")
        if handoff.domain is not None and handoff.domain != self.domain:
            raise ProtocolError("Git Agent handoff domain identity mismatch")
        if handoff.reused_version_id is not None:
            version = self.graph.get(handoff.reused_version_id)
            checkpoint = self._checkpoint_for(version.version_id)
            self._validate_handoff_binding(handoff, checkpoint)
            if handoff.exact_reuse_identity is not None:
                identity = digest({
                    "version_id": version.version_id,
                    "requirement_digest": version.requirement_state.digest,
                    "checkpoint_digest": checkpoint.digest,
                })
                if handoff.exact_reuse_identity != identity:
                    raise ProtocolError("Git Agent exact-reuse identity mismatch")
            self.graph.activate_existing(version.version_id)
            return ExecutionOutcome(
                True,
                ExecutionMode.EXACT_REUSE,
                version.version_id,
                checkpoint.checkpoint_id,
                self.interface.restore(checkpoint),
                checkpoint.final_response,
                checkpoint.receipt,
                self.state0s.get(checkpoint.origin_state0_id or ""),
            )

        if handoff.target_version_id in self.bindings:
            raise ProtocolError("target version is already bound")

        base_version = None
        base_checkpoint = None
        if handoff.selected_base_version_id is not None:
            base_version = self.graph.get(handoff.selected_base_version_id)
            base_checkpoint = self._checkpoint_for(base_version.version_id)
            self._validate_handoff_binding(handoff, base_checkpoint)

        if base_checkpoint is None:
            if not self.graph.versions:
                return self._execute_initial_state0(
                    task_id=task_id,
                    source_turn=source_turn,
                    handoff=handoff,
                )
            mode = ExecutionMode.FRESH_SOLVE
            candidate = self.interface.empty(task_id)
            candidate.metadata["origin_state0_id"] = self._origin_state0_id()
        else:
            decision = self.router.decide(
                requirement=handoff.resolved_requirement,
                base_requirement=base_version.requirement_state.text if base_version else "",
                update_hint=handoff.update_hint,
                base_checkpoint=base_checkpoint,
            )
            mode = decision.mode
            if mode is ExecutionMode.REUSE_AND_PATCH:
                candidate = self.interface.fork(task_id, base_checkpoint)
            elif mode is ExecutionMode.FRESH_SOLVE:
                candidate = self.interface.empty(task_id)
                candidate.metadata["origin_state0_id"] = base_checkpoint.origin_state0_id
            else:
                raise ProtocolError(f"router returned invalid mode: {mode}")

        try:
            update_result = self.update_agent.execute(
                UpdateAgentRequest(
                    task_id=task_id,
                    domain=self.domain,
                    candidate=candidate,
                    resolved_requirement=handoff.resolved_requirement,
                    selected_base_checkpoint=base_checkpoint,
                    update_hint=handoff.update_hint,
                    domain_spec=self._domain_spec(),
                    public_runtime_assets=(
                        self.update_runtime_assets
                        if self.update_runtime_assets is not None
                        else self.public_runtime_assets
                    ),
                )
            )
            result = self._update_execution_result(update_result)
            receipt = self.interface.finalize(candidate, result)
            checkpoint_id = f"checkpoint-{handoff.target_version_id}"
            version = Version(
                version_id=handoff.target_version_id,
                requirement_state=RequirementState(handoff.resolved_requirement),
                parent_version_id=handoff.selected_base_version_id,
                checkpoint_id=checkpoint_id,
                source_turn=source_turn,
                created_by="commit",
                summary=handoff.update_hint,
            )
            self.graph.validate_add(version)
            checkpoint = self.interface.commit(
                candidate,
                checkpoint_id=checkpoint_id,
                version_id=handoff.target_version_id,
                receipt=receipt,
                final_response=result.final_response,
                origin_state0_id=str(candidate.metadata["origin_state0_id"]),
            )
        except (ProtocolError, RuntimeFinalizationError):
            self.interface.discard(candidate)
            raise
        except Exception as error:
            self.interface.discard(candidate)
            raise InfrastructureError(f"branch execution failed: {type(error).__name__}: {error}") from error

        # Graph mutation is deliberately last: failures above leave history unchanged.
        self._accept_checkpoint(version, checkpoint)
        return ExecutionOutcome(
            True,
            mode,
            version.version_id,
            checkpoint.checkpoint_id,
            checkpoint.work_state,
            result.final_response,
            receipt,
            update_result=update_result,
        )

    def _execute_initial_state0(
        self,
        *,
        task_id: str,
        source_turn: int,
        handoff: VersionHandoff,
    ) -> ExecutionOutcome:
        candidate = self.interface.empty(task_id)
        state0_id = f"state0-{handoff.target_version_id}"
        request = State0Request(
            task_id=task_id,
            domain=self.domain,
            resolved_requirement=handoff.resolved_requirement,
            public_runtime_assets=self.public_runtime_assets,
        )
        try:
            root_result = self.root_executor.create_state0(candidate, request)
            state0_candidate = State0Candidate(
                state0_id=state0_id,
                candidate_id=candidate.candidate_id,
                request=request,
                root_result=root_result,
            )
            if not root_result.finished:
                assert root_result.failure is not None
                raise RootExecutionError(root_result.failure.message)
            candidate.metadata.update(
                {
                    "state0_id": state0_id,
                    "state0_root_plan": dict(root_result.root_plan),
                    "state0_root_work_state": dict(root_result.work_state),
                }
            )
            execution_result = self._root_execution_result(root_result)
            receipt = self.interface.finalize(candidate, execution_result)
            assert_public_payload(receipt, location="finalization_receipt")
            checkpoint_id = f"checkpoint-{handoff.target_version_id}"
            version = Version(
                version_id=handoff.target_version_id,
                requirement_state=RequirementState(handoff.resolved_requirement),
                parent_version_id=None,
                checkpoint_id=checkpoint_id,
                source_turn=source_turn,
                created_by="commit",
                summary=handoff.update_hint,
            )
            self.graph.validate_add(version)
            preview = Checkpoint(
                checkpoint_id=checkpoint_id,
                version_id=handoff.target_version_id,
                work_state=self.interface.inspect(candidate),
                receipt=receipt,
                final_response=root_result.artifact,
                state0_id=state0_id,
                origin_state0_id=state0_id,
            )
            accept_state0(state0_candidate, checkpoint=preview, receipt=receipt)
            checkpoint = self.interface.commit(
                candidate,
                checkpoint_id=checkpoint_id,
                version_id=handoff.target_version_id,
                receipt=receipt,
                final_response=root_result.artifact,
                state0_id=state0_id,
                origin_state0_id=state0_id,
            )
            state0 = accept_state0(state0_candidate, checkpoint=checkpoint, receipt=receipt)
        except (ProtocolError, RootExecutionError, RuntimeFinalizationError, ValueError):
            self.interface.discard(candidate)
            raise
        except Exception as error:
            self.interface.discard(candidate)
            raise InfrastructureError(
                f"State 0 execution failed: {type(error).__name__}: {error}"
            ) from error

        self._accept_checkpoint(version, checkpoint, state0=state0)
        return ExecutionOutcome(
            True,
            ExecutionMode.FRESH_SOLVE,
            version.version_id,
            checkpoint.checkpoint_id,
            checkpoint.work_state,
            root_result.artifact,
            receipt,
            state0,
        )

    @staticmethod
    def _root_execution_result(root_result: StackPlannerRootResult) -> ExecutionResult:
        return ExecutionResult(
            finished=root_result.finished,
            final_response=root_result.artifact,
            trace=root_result.trace,
            error=root_result.failure.message if root_result.failure else None,
        )

    def _origin_state0_id(self) -> str:
        origins = {
            checkpoint.origin_state0_id
            for checkpoint in self.checkpoints.values()
            if checkpoint.origin_state0_id is not None
        }
        origins.update(self.state0s)
        if len(origins) != 1:
            raise ProtocolError("ordinary FreshSolve requires one accepted origin State 0")
        return next(iter(origins))

    def _accept_checkpoint(
        self,
        version: Version,
        checkpoint: Checkpoint,
        *,
        state0: State0 | None = None,
    ) -> None:
        try:
            self.graph.add(version)
        except Exception as error:
            self.interface.rollback_commit(checkpoint)
            raise InfrastructureError(
                f"checkpoint binding failed: {type(error).__name__}: {error}"
            ) from error
        self.checkpoints[checkpoint.checkpoint_id] = checkpoint
        self.bindings[version.version_id] = checkpoint.checkpoint_id
        if state0 is not None:
            self.state0s[state0.state0_id] = state0

    def _domain_spec(self):
        from githarness.domains import get_domain_spec

        return get_domain_spec(self.domain)

    @staticmethod
    def _update_execution_result(result: UpdateAgentResult) -> ExecutionResult:
        return ExecutionResult(
            finished=result.finished,
            final_response=result.artifact,
            trace=result.trace,
            error=result.failure.message if result.failure else None,
        )

    def _checkpoint_for(self, version_id: str) -> Checkpoint:
        try:
            checkpoint_id = self.bindings[version_id]
            return self.checkpoints[checkpoint_id]
        except KeyError as error:
            raise ProtocolError(f"version has no accepted checkpoint: {version_id}") from error

    @staticmethod
    def _validate_handoff_binding(handoff: VersionHandoff, checkpoint: Checkpoint) -> None:
        binding = handoff.selected_base_checkpoint
        if binding is None:
            return
        if (
            binding.version_id != checkpoint.version_id
            or binding.checkpoint_id != checkpoint.checkpoint_id
            or binding.checkpoint_digest != checkpoint.digest
            or binding.workspace_digest != checkpoint.work_state.digest
            or binding.origin_state0_id != checkpoint.origin_state0_id
        ):
            raise ProtocolError("Git Agent handoff checkpoint binding mismatch")
