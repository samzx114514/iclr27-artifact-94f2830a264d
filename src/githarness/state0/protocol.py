"""Protocol implemented by the canonical StackPlanner root."""

from __future__ import annotations

from typing import Protocol

from githarness.version_interface.protocol import CandidateWorkState

from .contracts import StackPlannerRootResult, State0Request


class StackPlannerRootExecutor(Protocol):
    def create_state0(
        self,
        candidate: CandidateWorkState,
        request: State0Request,
    ) -> StackPlannerRootResult: ...
