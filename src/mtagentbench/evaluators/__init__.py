"""Task-native evaluator adapters behind one stable dispatch boundary."""

from .base import EvaluationContext, EvaluatorAdapter
from .registry import get_evaluator

__all__ = ["EvaluationContext", "EvaluatorAdapter", "get_evaluator"]
