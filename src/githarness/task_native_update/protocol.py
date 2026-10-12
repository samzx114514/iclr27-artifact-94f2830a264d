"""Task-Native Update Agent protocol."""

from typing import Protocol

from .contracts import UpdateAgentRequest, UpdateAgentResult


class TaskNativeUpdateAgent(Protocol):
    def execute(self, request: UpdateAgentRequest) -> UpdateAgentResult: ...
