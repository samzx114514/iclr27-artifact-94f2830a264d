"""Bounded StackPlanner graph for single- and multi-turn DeepResearch reports."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from src.config.agents import AGENT_LLM_MAP
from src.llms.llm import get_llm_by_type
from src.prompts.sops import load_task_sop
from src.utils.logger import logger

from . import sp_nodes
from .types import State


def _public_user_turns(state: State) -> list[str]:
    """Return only user-authored turns, excluding named internal messages."""
    turns = []
    for message in state.get("messages", []):
        if (
            isinstance(message, HumanMessage)
            and not getattr(message, "name", None)
            and isinstance(message.content, str)
            and message.content.strip()
        ):
            turns.append(message.content.strip())
    return turns


def _locale(state: State, turns: list[str]) -> str:
    configured = str(state.get("locale") or "").strip()
    if configured:
        return configured
    combined = "\n".join(turns)
    return "zh-CN" if re.search(r"[\u3400-\u9fff]", combined) else "en-US"


def _restore_memory(state: State) -> None:
    sp_nodes._check_agents_initialized()
    checkpointed = state.get("memory_stack")
    if checkpointed:
        sp_nodes.global_central_agent.memory_stack.load_from_dict(checkpointed)


def _update(command: Any) -> dict[str, Any]:
    update = getattr(command, "update", None)
    if not isinstance(update, dict):
        raise RuntimeError("DeepResearch stage returned no state update")
    return dict(update)


def _qwen_report_compat(config: RunnableConfig) -> bool:
    return bool((config or {}).get("configurable", {}).get("qwen_report_compat"))


def _bounded_qwen_source_evidence(state: State, max_chars: int = 30000) -> str:
    """Expose real Qwen tool outputs to Reporter without overflowing context."""
    collections = [
        str(item) for item in state.get("data_collections", []) if str(item).strip()
    ]
    evidence = "\n\n".join(collections)
    # The shared search tool labels URLs as 【链接N】 while the legacy reporter
    # and deterministic reference processor both require bare 【N】 citations.
    evidence = re.sub(r"【链接(\d+)】", r"【\1】", evidence)
    if len(evidence) > max_chars:
        return evidence[:max_chars] + " [truncated]"
    return evidence


async def deep_research_central_brief_node(
    state: State, config: RunnableConfig
) -> dict[str, Any]:
    """Create a plain-text active brief; graph routing is not model-controlled."""
    _restore_memory(state)
    turns = _public_user_turns(state)
    if not turns:
        raise RuntimeError("DeepResearch requires at least one public user turn")
    locale = _locale(state, turns)
    transcript = "\n\n".join(
        f"User turn {index}:\n{content}"
        for index, content in enumerate(turns, start=1)
    )
    messages = [
        SystemMessage(content=load_task_sop("sp_deep_research") or ""),
        HumanMessage(
            content=(
                "Create the Central Brief for the active request after the latest "
                "turn. Preserve all non-conflicting earlier requirements and let "
                "later corrections replace earlier ones. State the research goal, "
                "scope, constraints, desired evidence, output language, and expected "
                "deliverable. Output plain text only; do not write an outline or the "
                f"report. Required locale: {locale}.\n\n{transcript}"
            )
        ),
    ]
    response = await get_llm_by_type(
        AGENT_LLM_MAP.get("central_agent", "basic")
    ).ainvoke(messages, config=config)
    brief = str(response.content or "").strip()
    if not brief:
        raise RuntimeError("Central Brief returned empty content")
    logger.info("DeepResearch Central Brief completed")
    return {
        "messages": [AIMessage(content=brief, name="deep_research_central_brief")],
        "deep_research_brief": brief,
        "user_query": brief,
        "locale": locale,
        "final_report": "",
        # These are per-turn scratch fields.  The public MessagesState remains
        # append-only, but prior outlines and retrieved snippets must not leak
        # into the next report generation pass.
        "report_outline": None,
        "observations": [],
        "data_collections": [],
        "delegation_context": None,
        "user_dst": "",
        "deep_research_submitted": False,
        "deep_research_error": None,
        "current_node": "central_brief",
    }


async def deep_research_outline_node(
    state: State, config: RunnableConfig
) -> dict[str, Any]:
    working = dict(state)
    # Legacy outline prompts include state["messages"].  Feed only the active
    # brief explicitly so checkpointed internal messages cannot steer output.
    working["messages"] = []
    working["user_query"] = state.get("deep_research_brief") or state.get(
        "user_query", ""
    )
    if _qwen_report_compat(config):
        # Match the existing async BoCha per-result bound and leave ample room
        # under the deployed Qwen3-32B 61,440-token context window.
        working["outline_background_per_result_chars"] = 1500
        working["outline_background_total_chars"] = 10000
    update = _update(await sp_nodes.sub_agent_manager.execute_outline(working, config))
    update["current_node"] = "outline"
    return update


async def deep_research_researcher_node(
    state: State, config: RunnableConfig
) -> dict[str, Any]:
    brief = str(state.get("deep_research_brief") or state.get("user_query") or "")
    outline = str(state.get("report_outline") or "")
    working = dict(state)
    working["messages"] = []
    working["user_query"] = brief
    working["delegation_context"] = {
        "agent_type": "researcher",
        "task_description": (
            "Research authoritative and practical evidence for the active brief, "
            "following the confirmed outline. Preserve retrieved document IDs for "
            "inline citations and do not draft the final report.\n\n"
            f"Active brief:\n{brief}\n\nConfirmed outline:\n{outline}"
        ),
        "original_query": brief,
    }
    update = _update(
        await sp_nodes.sub_agent_manager.execute_web_researcher(working, config)
    )
    messages = update.get("messages", [])
    if any("失败" in str(getattr(message, "content", message)) for message in messages):
        raise RuntimeError("DeepResearch web research failed")
    update["current_node"] = "researcher"
    return update


async def deep_research_reporter_node(
    state: State, config: RunnableConfig
) -> dict[str, Any]:
    brief = str(state.get("deep_research_brief") or state.get("user_query") or "")
    working = dict(state)
    qwen_compat = _qwen_report_compat(config)
    # reporter_xxqg receives the brief, outline and current-turn observations
    # through explicit fields below; full checkpoint history is both redundant
    # and capable of making the model repeat a Central Brief instead of a report.
    working["messages"] = []
    working["user_query"] = brief
    if qwen_compat:
        raw_evidence = _bounded_qwen_source_evidence(state)
        if raw_evidence:
            working["observations"] = list(state.get("observations", [])) + [
                "Real web-search tool outputs with stable source labels:\n"
                + raw_evidence,
                (
                    "Qwen report citation contract: the evidence labels 【1】, 【2】, "
                    "and so on are valid source IDs. Cite factual claims inline with "
                    "one or more of those exact labels. Do not output a References "
                    "section; deterministic post-processing will build it."
                ),
            ]
    citation_instruction = (
        " Real search evidence below has normalized source labels such as 【1】. "
        "Cite supported claims using those exact numeric labels; deterministic "
        "post-processing will generate the reference list."
        if qwen_compat
        else ""
    )
    working["delegation_context"] = {
        "agent_type": "reporter",
        "task_description": (
            "Write exactly one complete report for the active brief, following the "
            "confirmed outline and using only the collected evidence. Use the "
            "configured locale. Cite sources inline only as 【docid】. Do not create "
            "a References or 参考文献 section; deterministic post-processing owns "
            "the reference list."
            + citation_instruction
        ),
        "original_query": brief,
        "report_type": "final_report",
    }
    update = _update(
        sp_nodes.sub_agent_manager.execute_xxqg_reporter(working, config)
    )
    report = str(update.get("final_report") or "").strip()
    if not report or report.startswith("报告生成失败"):
        raise RuntimeError(report or "DeepResearch Reporter returned empty content")
    update["current_node"] = "reporter"
    return update


def _deduplicate_reference_sections(report: str) -> str:
    """Keep only the deterministic final 参考文献 section."""
    marker = re.compile(r"(?im)^##\s*参考文献\s*$")
    matches = list(marker.finditer(report))
    if not matches:
        return report.strip()
    canonical = matches[-1]
    body = report[: canonical.start()].rstrip()
    body = re.split(
        r"(?im)^##\s*(?:references|参考文献)\s*$", body, maxsplit=1
    )[0].rstrip()
    return f"{body}\n\n{report[canonical.start():].strip()}".strip()


async def deep_research_submit_node(
    state: State, config: RunnableConfig
) -> dict[str, Any]:
    report = _deduplicate_reference_sections(str(state.get("final_report") or ""))
    if not report:
        raise RuntimeError("DeepResearch Submit received an empty report")
    logger.info("DeepResearch report submitted; fixed pipeline terminating")
    return {
        "messages": [
            AIMessage(content="DeepResearch report submitted", name="submit")
        ],
        "final_report": report,
        "deep_research_submitted": True,
        "current_node": "submit",
    }


def build_deep_research_graph() -> StateGraph:
    """Build Central Brief -> Outline -> Researcher -> Reporter -> Submit."""
    builder = StateGraph(State)
    builder.add_node("central_brief", deep_research_central_brief_node)
    builder.add_node("outline", deep_research_outline_node)
    builder.add_node("researcher", deep_research_researcher_node)
    builder.add_node("reporter", deep_research_reporter_node)
    builder.add_node("submit", deep_research_submit_node)
    builder.add_edge(START, "central_brief")
    builder.add_edge("central_brief", "outline")
    builder.add_edge("outline", "researcher")
    builder.add_edge("researcher", "reporter")
    builder.add_edge("reporter", "submit")
    builder.add_edge("submit", END)
    return builder
