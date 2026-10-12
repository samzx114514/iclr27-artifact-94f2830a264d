"""Common primitives for deterministic dataset packaging and validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from githarness.errors import DataLeakageError
from mtagentbench.batch_evaluation import read_json_or_jsonl
from mtagentbench.contracts import to_plain
from mtagentbench.leakage import assert_public_safe


@dataclass(frozen=True)
class FrozenSplit:
    """Repository locations for one canonical public/private split."""

    domain: str
    split: str
    public_directory: str
    private_directory: str

    @property
    def key(self) -> tuple[str, str]:
        return self.domain, self.split

    @property
    def public_path(self) -> str:
        return f"{self.public_directory}/public_cases.jsonl"

    @property
    def private_path(self) -> str:
        return f"{self.private_directory}/private_eval.jsonl"


CANONICAL_SPLITS = (
    FrozenSplit("math", "train", "datasets/math/train", ".local/datasets/math/train"),
    FrozenSplit("math", "validation", "datasets/math/validation", ".local/datasets/math/validation"),
    FrozenSplit("math", "test", "datasets/math/test", ".local/datasets/math/test"),
    FrozenSplit("sql", "train", "datasets/sql/train", ".local/datasets/sql/train"),
    FrozenSplit("sql", "validation", "datasets/sql/validation", ".local/datasets/sql/validation"),
    FrozenSplit("sql", "test", "datasets/sql/test", ".local/datasets/sql/test"),
    FrozenSplit("search", "test", "datasets/search/test", ".local/datasets/search/test"),
    FrozenSplit("research", "test", "datasets/research/test", ".local/datasets/research"),
    FrozenSplit("code", "test", "datasets/code/test", ".local/datasets/code/test"),
)

FORBIDDEN_PUBLIC_KEYS = {
    "answer", "label", "reference", "gold", "gold_sql", "grader",
    "grader_feedback", "per_turn_gold", "operator_validation", "change_plan",
    "construction_state", "audit_receipt",
}
SENSITIVE_TEXT = re.compile(
    r"/(?:data\d*|home)/[^/\s]+/|sk-[A-Za-z0-9_-]{16,}|"
    r"AKIA[0-9A-Z]{16}|BEGIN[^\n]{0,20}PRIVATE KEY",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_json(path: Path, value: object) -> None:
    _atomic_write(
        path,
        json.dumps(to_plain(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def write_jsonl(path: Path, values: Iterable[object]) -> None:
    _atomic_write(
        path,
        "".join(
            json.dumps(to_plain(value), ensure_ascii=False, sort_keys=True) + "\n"
            for value in values
        ),
    )


def records_by_task_id(
    rows: Sequence[Mapping[str, Any]], *, label: str,
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"{label}: task IDs must be non-empty strings")
        if task_id in result:
            raise ValueError(f"{label}: duplicate task ID: {task_id}")
        result[task_id] = row
    return result


def _hidden_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        keys = {str(key).casefold() for key in value} & FORBIDDEN_PUBLIC_KEYS
        return keys | set().union(*(_hidden_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_hidden_keys(item) for item in value), set())
    return set()


def _safe_repository_path(root: Path, value: object, *, label: str) -> Path:
    relative = Path(str(value))
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or relative.parts[0] != "datasets"
    ):
        raise ValueError(f"{label}: registry path escapes the datasets directory")
    return root / relative


def validate_public_split(
    root: Path, spec: FrozenSplit, entry: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], list[str]]:
    """Validate common schema, SHA-256, isolation, and path rules for one split."""
    errors: list[str] = []
    label = f"{spec.domain}/{spec.split}"
    try:
        public_path = _safe_repository_path(root, entry["public_path"], label=label)
    except (KeyError, ValueError) as exc:
        return [], [str(exc)]
    rows = read_json_or_jsonl(public_path)
    try:
        records_by_task_id(rows, label=label)
    except ValueError as exc:
        errors.append(str(exc))
    if len(rows) != entry.get("rows"):
        errors.append(f"{label}: row count differs from registry")
    for row in rows:
        if row.get("domain") != spec.domain or len(row.get("turns") or []) != 7:
            errors.append(f"{label}: domain or seven-turn contract failed")
            break
        if any(turn.get("role") != "user" for turn in row.get("turns") or []):
            errors.append(f"{label}: every public turn must have the user role")
            break
        try:
            assert_public_safe(row)
        except (DataLeakageError, ValueError) as exc:
            errors.append(f"{label}: public leakage check failed: {exc}")
            break
        hidden = _hidden_keys(row)
        if hidden:
            errors.append(f"{label}: hidden public keys found: {sorted(hidden)}")
            break
    if SENSITIVE_TEXT.search(public_path.read_text(encoding="utf-8")):
        errors.append(f"{label}: secret, username, or internal path pattern found")
    if entry.get("public_sha256") != sha256_file(public_path):
        errors.append(f"{label}: registry public SHA-256 differs from file")
    forbidden_files = {"private_eval.jsonl", "validation.json"}
    present = {path.name for path in public_path.parent.iterdir() if path.is_file()}
    if present & forbidden_files:
        errors.append(f"{label}: private-only file present in public directory")
    return rows, errors
