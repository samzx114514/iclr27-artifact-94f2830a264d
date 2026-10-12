"""Normalized exact-match evaluator for Math artifacts."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re

from ..contracts import EvaluationResult, PrivateEvaluationRecord
from .base import Artifact, EvaluationContext, require_record


def _boxed(text: str) -> list[str]:
    marker = r"\boxed{"
    values: list[str] = []
    cursor = 0
    while True:
        start = text.find(marker, cursor)
        if start < 0:
            return values
        index = start + len(marker)
        depth = 1
        begin = index
        while index < len(text) and depth:
            depth += (text[index] == "{") - (text[index] == "}")
            index += 1
        if depth:
            return values
        values.append(text[begin:index - 1])
        cursor = index


def extract_math_answer(text: str) -> str | None:
    values = _boxed(text)
    if values:
        return values[-1].strip()
    gsm = re.findall(r"####\s*([^\n]+)", text)
    if gsm:
        return gsm[-1].strip()
    marked = re.findall(r"(?i)(?:final answer|answer)\s*(?:is|:|=)\s*([^\n]+)", text)
    if marked:
        return marked[-1].strip().rstrip(".")
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return numbers[-1] if numbers else None


def _numeric_math(value: str) -> Fraction | None:
    """Parse the leading boxed number without treating units as answer content."""

    cleaned = value.replace("−", "-").replace("{,}", "").replace(",", "")
    fraction = re.match(
        r"\s*\\(?:d?frac)\{\s*([-+]?\d+(?:\.\d+)?)\s*\}"
        r"\{\s*([-+]?\d+(?:\.\d+)?)\s*\}",
        cleaned,
    )
    if fraction:
        denominator = Decimal(fraction.group(2))
        if denominator == 0:
            return None
        return Fraction(Decimal(fraction.group(1))) / Fraction(denominator)

    # Currency markers and simple LaTeX wrappers may precede the number. Match
    # only the leading numeric expression so units or a boxed breakdown do not
    # become part of the answer.
    cleaned = re.sub(
        r"\\(?:text|mathrm)\{\s*([-+]?\d+(?:\.\d+)?)\s*\}", r"\1", cleaned,
    )
    cleaned = re.sub(r"^\s*(?:\\?[$£€¥]\s*)+", "", cleaned)
    number = re.match(
        r"\s*([-+]?\d+(?:\.\d+)?)(?:\s*/\s*([-+]?\d+(?:\.\d+)?))?",
        cleaned,
    )
    if not number:
        return None
    numerator = Fraction(Decimal(number.group(1)))
    if number.group(2) is None:
        return numerator
    denominator = Decimal(number.group(2))
    if denominator == 0:
        return None
    return numerator / Fraction(denominator)


def _normalized_math(value: str) -> str:
    numeric = _numeric_math(value)
    if numeric is not None:
        return str(numeric.numerator) if numeric.denominator == 1 else str(numeric)
    cleaned = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", value)
    cleaned = cleaned.replace(",", "").replace("$", "").replace("{", "").replace("}", "").strip()
    try:
        return format(Decimal(cleaned).normalize(), "f")
    except InvalidOperation:
        return " ".join(cleaned.lower().split())


def evaluate_math(
    task_id: str, artifact: str, record: PrivateEvaluationRecord,
) -> EvaluationResult:
    predicted = extract_math_answer(artifact)
    expected = str(record.reference.get("answer") or "")
    correct = (
        predicted is not None
        and _normalized_math(predicted)
        == _normalized_math(expected)
    )
    return EvaluationResult(
        task_id, "math", "correct" if correct else "incorrect", True,
        correct, float(correct),
    )


class MathEvaluator:
    domain = "math"
    verifier = "normalized_exact_match"

    def evaluate(
        self, artifact: Artifact, record: PrivateEvaluationRecord,
        context: EvaluationContext,
    ) -> EvaluationResult:
        del context
        require_record(record, domain=self.domain, verifier=self.verifier)
        return evaluate_math(record.task_id, str(artifact), record)
