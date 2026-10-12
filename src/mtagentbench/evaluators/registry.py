"""Closed evaluator registry for the five task-native domains."""

from __future__ import annotations

from githarness.errors import TaskEvaluationError

from ..contracts import Domain
from .base import EvaluatorAdapter
from .code import CodeEvaluator
from .math import MathEvaluator
from .research import ResearchEvaluator
from .search import SearchEvaluator
from .sql import SQLEvaluator


EVALUATORS: dict[Domain, EvaluatorAdapter] = {
    "math": MathEvaluator(),
    "sql": SQLEvaluator(),
    "search": SearchEvaluator(),
    "research": ResearchEvaluator(),
    "code": CodeEvaluator(),
}


def get_evaluator(domain: str) -> EvaluatorAdapter:
    try:
        return EVALUATORS[domain]  # type: ignore[index]
    except KeyError as exc:
        raise TaskEvaluationError(f"unsupported domain: {domain}") from exc
