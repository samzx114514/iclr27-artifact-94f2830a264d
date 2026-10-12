"""Domain artifact schemas and non-semantic finalization checks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from githarness.version_interface.protocol import ExecutionResult


Finalizer = Callable[[Path, ExecutionResult], Mapping[str, Any]]

_SQL_WRITE_KEYWORDS = re.compile(
    r"\b(?:insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|replace|reindex)\b",
    re.IGNORECASE,
)
_SQL_QUOTED = re.compile(
    r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`[^`]*`|\[[^]]*]",
    re.DOTALL,
)


def validate_read_only_sql(query: str) -> str:
    value = query.strip()
    if re.match(r"^(?:select|with)\b", value, re.IGNORECASE) is None:
        raise ValueError("only SELECT/WITH SQL is permitted")
    if "--" in value or "/*" in value or "*/" in value:
        raise ValueError("SQL comments are not permitted")
    statement = value[:-1].rstrip() if value.endswith(";") else value
    if ";" in statement:
        raise ValueError("exactly one SQL statement is permitted")
    if _SQL_WRITE_KEYWORDS.search(_SQL_QUOTED.sub("", statement)):
        raise ValueError("SQL write or control statements are not permitted")
    return statement


def _text_artifact(path: str, *, prefix: tuple[str, ...] = ()) -> Finalizer:
    def finalize(root: Path, result: ExecutionResult) -> Mapping[str, Any]:
        artifact = root / path
        if not artifact.is_file():
            return {"passed": False, "error": f"missing artifact: {path}"}
        content = artifact.read_text(encoding="utf-8").strip()
        if not content:
            return {"passed": False, "error": f"empty artifact: {path}"}
        if prefix and not content.lower().startswith(prefix):
            return {"passed": False, "error": f"invalid artifact format: {path}"}
        return {"passed": True, "artifact": path, "bytes": len(content.encode("utf-8"))}
    return finalize


def _search_finalizer(root: Path, result: ExecutionResult) -> Mapping[str, Any]:
    path = root / "answer.json"
    if not path.is_file():
        return {"passed": False, "error": "missing artifact: answer.json"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return {"passed": False, "error": f"invalid answer.json: {error}"}
    allowed = {"schema_version", "exact_answer", "explanation", "citations", "confidence"}
    if not isinstance(payload, dict) or set(payload) - allowed:
        return {"passed": False, "error": "answer.json has unsupported fields"}
    answer = payload.get("exact_answer")
    if not isinstance(answer, str) or not answer.strip():
        return {"passed": False, "error": "answer.json requires exact_answer"}
    citations = payload.get("citations", [])
    if (
        not isinstance(citations, list)
        or any(not isinstance(item, str) or not item for item in citations)
        or len(citations) != len(set(citations))
    ):
        return {"passed": False, "error": "answer.json citations must be unique docids"}
    evidence = root / "evidence.jsonl"
    if citations:
        if not evidence.is_file():
            return {"passed": False, "error": "cited answer requires runtime evidence"}
        try:
            docids = {
                str(json.loads(line)["docid"])
                for line in evidence.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            return {"passed": False, "error": "invalid runtime evidence ledger"}
        if not set(citations) <= docids:
            return {"passed": False, "error": "answer.json cites unknown docids"}
    return {
        "passed": True,
        "artifact": "answer.json",
        "exact_answer_chars": len(answer),
        "citations": len(citations),
    }


def _sql_finalizer(root: Path, result: ExecutionResult) -> Mapping[str, Any]:
    path = root / "answer.sql"
    if not path.is_file():
        return {"passed": False, "error": "missing artifact: answer.sql"}
    content = path.read_text(encoding="utf-8")
    try:
        statement = validate_read_only_sql(content)
    except ValueError as error:
        return {"passed": False, "error": str(error)}
    return {
        "passed": True,
        "artifact": "answer.sql",
        "bytes": len(statement.encode("utf-8")),
    }


def _research_finalizer(root: Path, result: ExecutionResult) -> Mapping[str, Any]:
    base = dict(_text_artifact("report.md")(root, result))
    if base.get("passed") is not True:
        return base
    report = (root / "report.md").read_text(encoding="utf-8")
    labels = {int(item) for item in re.findall(r"【(\d+)】", report)}
    references_path = root / "references.json"
    if references_path.is_file():
        try:
            references = json.loads(references_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"passed": False, "error": "invalid references.json"}
        if not isinstance(references, dict) or any(
            not str(key).isdigit() or not isinstance(value, dict)
            for key, value in references.items()
        ):
            return {"passed": False, "error": "invalid references.json schema"}
        known = {int(key) for key in references}
        if not labels <= known:
            return {"passed": False, "error": "report cites unknown runtime sources"}
        base["references"] = len(references)
        base["citations"] = len(labels)
    return base


def _code_finalizer(root: Path, result: ExecutionResult) -> Mapping[str, Any]:
    path = root / "patch.diff"
    if not path.is_file():
        return {"passed": False, "error": "missing artifact: patch.diff"}
    patch = path.read_text(encoding="utf-8")
    if patch.strip() and "diff --git " not in patch:
        return {"passed": False, "error": "patch.diff is not a git diff"}
    receipt_path = root / "test_receipt.json"
    test_runs = 0
    if receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"passed": False, "error": "invalid test_receipt.json"}
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema_version") != 1
            or receipt.get("workspace") != "/testbed"
            or not isinstance(receipt.get("test_runs"), list)
        ):
            return {"passed": False, "error": "invalid test_receipt.json schema"}
        expected = hashlib.sha256(patch.encode("utf-8")).hexdigest()
        if receipt.get("patch_sha256") != expected:
            return {"passed": False, "error": "test receipt patch binding mismatch"}
        test_runs = len(receipt["test_runs"])
    return {
        "passed": True,
        "artifact": "patch.diff",
        "bytes": len(patch.encode("utf-8")),
        "files": len(re.findall(r"(?m)^diff --git ", patch)),
        "noop": not patch.strip(),
        "test_runs": test_runs,
    }


@dataclass(frozen=True)
class DomainSpec:
    name: str
    graph_format: str
    artifact_path: str
    finalizer: Finalizer
    public_capabilities: tuple[str, ...]


DOMAIN_SPECS: dict[str, DomainSpec] = {
    "math": DomainSpec("math", "sp_math", "answer.md", _text_artifact("answer.md"), ("calculator",)),
    "sql": DomainSpec(
        "sql",
        "sp_sql",
        "answer.sql",
        _sql_finalizer,
        (),
    ),
    "search": DomainSpec("search", "sp_search", "answer.json", _search_finalizer, ("search(query)",)),
    "research": DomainSpec(
        "research",
        "sp_deep_research",
        "report.md",
        _research_finalizer,
        ("search(query)",),
    ),
    "code": DomainSpec("code", "sp_swe", "patch.diff", _code_finalizer, ("/testbed",)),
}


def get_domain_spec(domain: str) -> DomainSpec:
    try:
        return DOMAIN_SPECS[domain]
    except KeyError as error:
        raise ValueError(f"unsupported domain: {domain}") from error
