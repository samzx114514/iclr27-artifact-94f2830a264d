"""BrowseComp-style normalized exact-match evaluator."""

from __future__ import annotations

from collections import Counter
import json
import re
import unicodedata
from typing import Any, Mapping

from ..contracts import EvaluationResult, PrivateEvaluationRecord
from .base import Artifact, EvaluationContext, require_record


def _search_normalize(value: str) -> str:
    value = "".join(
        char for char in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(char)
    ).lower()
    value = re.sub(r"\s*\([^)]*\)\s*", " ", value)
    value = re.sub(r"[\u2018\u2019\u201A\u201B\u2032\"'`*#]+", "", value)
    value = re.sub(r"[:–—\-_/.,;!()?]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def search_exact_match(label: str, prediction: str) -> bool:
    ignored = {"a", "an", "the", "of", "on", "in", "and", "&", "for", "to", "by", "with"}

    def strip(value: str) -> str:
        return re.sub(r"\s+", "", _search_normalize(value))

    if strip(label) == strip(prediction):
        return True

    def tokens(value: str) -> list[str]:
        return [
            token for token in _search_normalize(value).split()
            if token not in ignored and not re.fullmatch(r"\d{4}", token)
        ]

    return bool(tokens(label)) and Counter(tokens(label)) == Counter(tokens(prediction))


def evaluate_search(
    task_id: str, artifact: str | Mapping[str, Any],
    record: PrivateEvaluationRecord,
) -> EvaluationResult:
    if isinstance(artifact, str):
        try:
            artifact = json.loads(artifact)
        except json.JSONDecodeError:
            artifact = {}
    predicted = artifact.get("exact_answer") if isinstance(artifact, Mapping) else None
    expected = record.reference.get("answer")
    correct = (
        isinstance(predicted, str)
        and isinstance(expected, str)
        and search_exact_match(expected, predicted)
    )
    return EvaluationResult(
        task_id, "search", "correct" if correct else "incorrect", True,
        correct, float(correct),
    )


class SearchEvaluator:
    domain = "search"
    verifier = "browsecomp_normalized_em"

    def evaluate(
        self, artifact: Artifact, record: PrivateEvaluationRecord,
        context: EvaluationContext,
    ) -> EvaluationResult:
        del context
        require_record(record, domain=self.domain, verifier=self.verifier)
        return evaluate_search(record.task_id, artifact, record)
