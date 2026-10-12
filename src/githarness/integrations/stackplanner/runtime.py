"""Shared lazy access to the vendored canonical StackPlanner graphs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
import sys
from typing import Any, Mapping

from githarness.domains.specs import DomainSpec
from githarness.errors import InfrastructureError


def default_vendor_root() -> Path:
    repository_root = Path(__file__).resolve().parents[4]
    source_vendor = repository_root / "vendor" / "stackplanner"
    installed_vendor = Path(__file__).resolve().parents[2] / "_vendor" / "stackplanner"
    return source_vendor if source_vendor.is_dir() else installed_vendor


def compile_task_graph(graph_format: str, vendor_root: Path) -> Any:
    if not (vendor_root / "src").is_dir():
        raise InfrastructureError(f"vendored StackPlanner runtime is missing: {vendor_root}")
    root_text = str(vendor_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    try:
        from langgraph.checkpoint.memory import MemorySaver
        from src.graph.sp_nodes import init_agents
        from src.graph.task_profiles import get_task_graph_profile

        if graph_format == "sp_math":
            from src.graph.math_graph import build_math_graph as factory
        elif graph_format == "sp_sql":
            from src.graph.sql_graph import build_sql_graph as factory
        elif graph_format == "sp_swe":
            from src.graph.swe_graph import build_swe_graph as factory
        elif graph_format == "sp_search":
            from src.graph.search_graph import build_search_graph as factory
        elif graph_format == "sp_deep_research":
            from src.graph.deep_research_graph import build_deep_research_graph as factory
        else:
            raise ValueError(f"unsupported StackPlanner graph: {graph_format}")

        if graph_format in {"sp_math", "sp_sql", "sp_deep_research"}:
            init_agents(graph_format)
        builder = factory()
        profile = get_task_graph_profile(graph_format)
        checkpointer = MemorySaver() if profile and profile.supports_checkpoint_memory else None
        return builder.compile(checkpointer=checkpointer)
    except Exception as error:
        raise InfrastructureError(f"StackPlanner imports failed: {error}") from error


def artifact_from_state(spec: DomainSpec, state: Mapping[str, Any]) -> str:
    if spec.name == "search":
        answer = state.get("search_exact_answer") or state.get("exact_answer") or state.get("final_report")
        import json

        return json.dumps({"exact_answer": str(answer or "").strip()}, ensure_ascii=False)
    if spec.name == "code":
        return str(state.get("swe_patch") or state.get("model_patch") or "")
    if spec.name == "sql":
        import re

        raw = str(state.get("final_report") or state.get("answer") or "").strip()
        block = re.search(r"```(?:sql)?\s*(.*?)```", raw, re.IGNORECASE | re.DOTALL)
        return (block.group(1) if block else raw).strip().rstrip(";")
    return str(state.get("final_report") or state.get("answer") or "").strip()


def primitive(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [primitive(item) for item in value]
    if is_dataclass(value):
        return primitive(asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return primitive(model_dump())
    return str(value)


def root_plan_from_state(state: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("current_plan", "plan", "task_plan", "search_plan", "swe_plan", "report_outline"):
        value = state.get(key)
        if value not in (None, "", [], {}):
            return {"source_field": key, "plan": primitive(value)}
    return {}


_WORK_STATE_FIELDS = {
    "math": ("memory_stack", "final_report", "answer"),
    "sql": ("memory_stack", "final_report", "answer"),
    "search": (
        "search_plan",
        "search_exact_answer",
        "search_citations",
        "search_confidence",
        "search_retrieval_trace",
        "search_retrieved_docids",
    ),
    "research": (
        "memory_stack",
        "final_report",
        "report_outline",
        "deep_research_brief",
        "deep_research_submitted",
    ),
    "code": ("swe_plan", "swe_coder_summary", "swe_patch", "swe_steps", "swe_tool_calls"),
}


def root_work_state(spec: DomainSpec, state: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        key: primitive(state[key])
        for key in _WORK_STATE_FIELDS[spec.name]
        if key in state and state[key] is not None
    }
