"""Deterministic offline agents used by examples and tests."""

from __future__ import annotations

import ast
from decimal import Decimal
from pathlib import Path
import re
from typing import Callable

from .state0 import ExecutionFailure, StackPlannerRootResult, State0Request
from .version_interface.protocol import CandidateWorkState
from .version_interface.protocol import ExecutionResult


_BINARY = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
}
_UNARY = {ast.UAdd: lambda a: a, ast.USub: lambda a: -a}


def safe_arithmetic(expression: str) -> Decimal:
    def visit(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return Decimal(str(node.value))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return _BINARY[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](visit(node.operand))
        raise ValueError("offline calculator accepts numbers and + - * / only")
    return visit(ast.parse(expression, mode="eval"))


def deterministic_math_update(workspace: Path, requirement: str, update_hint: str) -> ExecutionResult:
    match = re.search(r"(?i)(?:expression|compute|calculate)\s*[:=]?\s*([-+*/().\d\s]+)", requirement)
    if match is None:
        return ExecutionResult(False, "", error="requirement has no arithmetic expression")
    expression = match.group(1).strip().rstrip(".")
    try:
        value = safe_arithmetic(expression)
    except Exception as error:
        return ExecutionResult(False, "", error=str(error))
    normalized = format(value.normalize(), "f")
    content = f"Expression: `{expression}`\n\nFinal answer: \\boxed{{{normalized}}}\n"
    (workspace / "answer.md").write_text(content, encoding="utf-8")
    return ExecutionResult(
        True,
        content,
        trace=({"tool": "calculate", "expression": expression, "value": normalized},),
    )


class DeterministicMathRootExecutor:
    """Offline-only State 0 root used by deterministic examples and tests."""

    domain = "math"

    def create_state0(
        self,
        candidate: CandidateWorkState,
        request: State0Request,
    ) -> StackPlannerRootResult:
        result = deterministic_math_update(
            candidate.root,
            request.resolved_requirement,
            "",
        )
        if not result.finished:
            return StackPlannerRootResult(
                False,
                "answer.md",
                "",
                failure=ExecutionFailure("root_failure", result.error or "offline root failed"),
            )
        return StackPlannerRootResult(
            True,
            "answer.md",
            result.final_response,
            root_plan={"kind": "deterministic_arithmetic"},
            work_state={"calculator": "decimal"},
            trace=result.trace,
        )


def deterministic_artifact_update(
    artifact_path: str, content: str,
) -> Callable[[Path, str, str], ExecutionResult]:
    """Return a deterministic fixture update agent for offline boundary tests."""
    relative = Path(artifact_path)
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
        raise ValueError("offline fixture artifact must be one safe relative filename")
    if not content.strip() and artifact_path != "patch.diff":
        raise ValueError("offline fixture artifact content must be non-empty")

    def update(workspace: Path, requirement: str, update_hint: str) -> ExecutionResult:
        del update_hint
        target = workspace / relative
        target.write_text(content, encoding="utf-8")
        return ExecutionResult(
            True,
            content,
            trace=({
                "tool": "offline_fixture",
                "artifact": artifact_path,
                "requirement_chars": len(requirement),
            },),
        )

    return update


class DeterministicArtifactRootExecutor:
    """Synthetic-only State 0 root for five-domain boundary smokes."""

    def __init__(self, artifact_path: str, content: str) -> None:
        self.artifact_path = artifact_path
        self.content = content

    def create_state0(
        self,
        candidate: CandidateWorkState,
        request: State0Request,
    ) -> StackPlannerRootResult:
        result = deterministic_artifact_update(
            self.artifact_path, self.content,
        )(candidate.root, request.resolved_requirement, "")
        return StackPlannerRootResult(
            finished=result.finished,
            artifact_path=self.artifact_path,
            artifact=result.final_response,
            root_plan={"kind": "synthetic_fixture"},
            work_state={"artifact_path": self.artifact_path},
            trace=result.trace,
            failure=(
                None
                if result.finished
                else ExecutionFailure(
                    "root_failure", result.error or "offline fixture root failed"
                )
            ),
        )
