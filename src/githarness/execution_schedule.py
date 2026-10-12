"""Execution schedules over one shared Git Agent and Branch Execution method."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from githarness.requirement_resolution.handoff import VersionHandoff
from githarness.versioning import RequirementState, Version, VersionGraph


class ExecutionSchedule(str, Enum):
    EAGER = "eager"
    DEFERRED_NECESSARY_PATH = "deferred_necessary_path"


@dataclass(frozen=True)
class PlannedTurn:
    source_turn: int
    handoff: VersionHandoff
    active_version_id: str


@dataclass
class RequirementExecutionPlan:
    """Requirement-only graph; entries are proposals, never accepted checkpoints."""

    graph: VersionGraph = field(default_factory=VersionGraph)
    turns: list[PlannedTurn] = field(default_factory=list)
    handoffs_by_version: dict[str, VersionHandoff] = field(default_factory=dict)

    def record(self, source_turn: int, handoff: VersionHandoff) -> PlannedTurn:
        if handoff.reused_version_id is not None:
            self.graph.activate_existing(handoff.reused_version_id)
            active = handoff.reused_version_id
        else:
            version = Version(
                version_id=handoff.target_version_id,
                requirement_state=RequirementState(handoff.resolved_requirement),
                parent_version_id=handoff.selected_base_version_id,
                checkpoint_id=f"planned-{handoff.target_version_id}",
                source_turn=source_turn,
                summary=handoff.update_hint,
            )
            self.graph.add(version)
            self.handoffs_by_version[version.version_id] = handoff
            active = version.version_id
        planned = PlannedTurn(source_turn, handoff, active)
        self.turns.append(planned)
        return planned

    @property
    def bootstrap_version_id(self) -> str | None:
        return next(iter(self.handoffs_by_version), None)

    def necessary_versions(self, target_turns: Iterable[int]) -> tuple[str, ...]:
        targets = tuple(target_turns)
        if not targets:
            return ()
        required: set[str] = set()
        if self.bootstrap_version_id is not None:
            required.add(self.bootstrap_version_id)
        for turn_index in targets:
            if turn_index < 0 or turn_index >= len(self.turns):
                raise ValueError(f"deferred target turn is out of range: {turn_index}")
            active = self.turns[turn_index].active_version_id
            required.update(item.version_id for item in self.graph.lineage(active))
        return tuple(
            version.version_id
            for version in self.graph.versions
            if version.version_id in required
        )


def parse_execution_schedule(value: object) -> ExecutionSchedule:
    try:
        return ExecutionSchedule(str(value or ExecutionSchedule.EAGER.value))
    except ValueError as error:
        choices = ", ".join(item.value for item in ExecutionSchedule)
        raise ValueError(f"execution_schedule must be one of: {choices}") from error
