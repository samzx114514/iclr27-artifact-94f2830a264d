"""Dedicated single-turn StackPlanner graph for BrowseComp+ search."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph

from src.llms.llm import get_llm_by_type
from src.search.backend import SearchBackend

from .types import State


CENTRAL_PROMPT = """You are the Central Agent for a single-turn BrowseComp+ task.
Read only the public question and produce a concise search plan. Break identifying
clues into useful retrieval queries and explain how evidence should be joined.
Do not invent an answer or claim to have searched.

Public question:
{question}
"""

SEARCH_PROMPT = """You are the StackPlanner Search Agent for BrowseComp+.
Use the Search tool iteratively to identify the exact answer from the supplied
read-only corpus. Search returns top passages with docid, score, and snippet.
Ground the explanation in retrieved passages and cite supporting docids inline as
[docid]. When evidence is sufficient, call Finish with a succinct exact answer,
an evidence-based explanation, the cited docids, and confidence from 0 to 100.

The exact_answer field is graded independently by normalized exact match. Put only
the shortest canonical answer requested by the question in exact_answer. Do not
add explanations, affiliations, locations, translations, parenthetical aliases,
titles, or alternative names unless they are part of the answer itself. Put all
supporting detail in explanation instead.

Never access a host shell, arbitrary files, the internet, benchmark metadata, the
gold answer, or grader feedback. Do not invent observations or citations. Every
citation must name a docid returned by Search. Keep queries targeted and stop when
the clues are jointly supported.

Public question:
{question}

Central Agent plan:
{plan}
"""

MULTITURN_CENTRAL_PROMPT = """You are the Central Agent for a multi-turn
BrowseComp+ task. Maintain the investigation across turns. The newest user turn
may add clues, ask for the answer, or explicitly correct an earlier detail; the
newest correction takes precedence. Produce a concise plan for this turn that
reuses relevant prior evidence and identifies what still needs verification. Do
not invent an answer or claim to have searched."""

MULTITURN_SEARCH_PROMPT = """You are the StackPlanner Search Agent for a
multi-turn BrowseComp+ task. Your message stream is append-only and contains the
previous user turns, plans, Search calls, returned passages, and answers. Reuse
that evidence, but treat an explicit newer user correction as authoritative over
an older conflicting clue. Search again when the accumulated evidence is not
sufficient for the current request.

Use only the injected read-only Search tool. Never access a shell, arbitrary
files, the internet, benchmark metadata, the gold answer, or grader feedback.
When ready, call Finish. Put only the shortest canonical answer in exact_answer;
put supporting details in explanation. Citations and confidence are auxiliary,
and every cited docid must have been returned by Search."""


def _configurable(config: RunnableConfig) -> dict[str, Any]:
    return dict(config.get("configurable", {}))


def _backend(config: RunnableConfig) -> SearchBackend:
    backend = _configurable(config).get("search_backend")
    if backend is None:
        raise RuntimeError("sp_search requires configurable.search_backend")
    return backend


def _llm(config: RunnableConfig):
    return _configurable(config).get("search_llm") or get_llm_by_type("basic")


def _normalize_results(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("search backend must return a list")
    rows = []
    for item in value:
        if not isinstance(item, dict) or "docid" not in item:
            raise ValueError("each search result must contain docid")
        rows.append(
            {
                "docid": str(item["docid"]),
                "score": float(item.get("score", 0.0)),
                "snippet": str(item.get("snippet", "")),
            }
        )
    return rows


async def search_central_node(state: State, config: RunnableConfig) -> dict[str, Any]:
    question = state.get("user_query", "")
    if not question:
        raise ValueError("sp_search requires a non-empty public question")
    response = await _llm(config).ainvoke(
        [HumanMessage(content=CENTRAL_PROMPT.format(question=question))]
    )
    plan = str(response.content).strip()
    if not plan:
        raise RuntimeError("Search Central Agent returned an empty plan")
    return {
        "search_plan": plan,
        "current_node": "search_central",
    }


async def search_multiturn_central_node(
    state: State, config: RunnableConfig
) -> dict[str, Any]:
    user_turn = state.get("user_query", "")
    if not user_turn:
        raise ValueError("sp_search_multiturn requires a non-empty public user turn")
    messages = list(state.get("search_central_messages", []))
    if not messages:
        messages.append(SystemMessage(content=MULTITURN_CENTRAL_PROMPT))
    messages.append(HumanMessage(content=user_turn))
    response = await _llm(config).ainvoke(messages)
    messages.append(response)
    plan = str(response.content).strip()
    if not plan:
        raise RuntimeError("Search Central Agent returned an empty plan")
    return {
        "search_plan": plan,
        "search_central_messages": messages,
        "current_node": "search_central",
    }


def _tools(backend: SearchBackend) -> list[StructuredTool]:
    def search(query: str) -> str:
        """Search the read-only BrowseComp+ corpus for relevant passages."""
        if not query.strip():
            raise ValueError("query must be non-empty")
        return json.dumps(_normalize_results(backend.search(query)), ensure_ascii=False)

    def finish(
        exact_answer: str,
        explanation: str = "",
        citations: list[str] | None = None,
        confidence: float = 0,
    ) -> str:
        """Submit the shortest exact answer; evidence fields are optional metadata."""
        return json.dumps(
            {
                "explanation": explanation,
                "exact_answer": exact_answer,
                "citations": citations or [],
                "confidence": confidence,
            },
            ensure_ascii=False,
        )

    return [
        StructuredTool.from_function(search, name="Search"),
        StructuredTool.from_function(finish, name="Finish"),
    ]


def _validate_finish(
    args: dict[str, Any], retrieved_docids: set[str]
) -> tuple[dict[str, Any] | None, str | None]:
    exact_answer = args.get("exact_answer")
    if not isinstance(exact_answer, str) or not exact_answer.strip():
        return None, "Finish exact_answer must be non-empty"
    explanation = args.get("explanation")
    if not isinstance(explanation, str):
        explanation = ""
    citations = args.get("citations")
    if not isinstance(citations, list):
        citations = []
    citations = [str(item) for item in citations]
    confidence = args.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= float(confidence) <= 100
    ):
        confidence = 0
    return {
        "explanation": explanation.strip(),
        "exact_answer": exact_answer.strip(),
        "citations": citations,
        "confidence": float(confidence),
    }, None


async def search_agent_node(state: State, config: RunnableConfig) -> dict[str, Any]:
    max_steps = int(_configurable(config).get("search_max_steps", 15))
    backend = _backend(config)
    tools = _tools(backend)
    tools_by_name = {tool.name: tool for tool in tools}
    model = _llm(config).bind_tools(tools)
    is_multiturn = _configurable(config).get("graph_format") == "sp_search_multiturn"
    if is_multiturn:
        messages = list(state.get("search_agent_messages", []))
        if not messages:
            messages.append(SystemMessage(content=MULTITURN_SEARCH_PROMPT))
        messages.append(
            HumanMessage(
                content=(
                    f"Newest user turn:\n{state.get('user_query', '')}\n\n"
                    f"Central Agent plan for this turn:\n{state.get('search_plan', '')}"
                )
            )
        )
    else:
        messages = [
            SystemMessage(
                content=SEARCH_PROMPT.format(
                    question=state.get("user_query", ""),
                    plan=state.get("search_plan", ""),
                )
            ),
            HumanMessage(
                content="Search the corpus and submit the grounded exact answer."
            ),
        ]
    trace: list[dict[str, Any]] = (
        list(state.get("search_retrieval_trace", [])) if is_multiturn else []
    )
    retrieved_docids: set[str] = (
        set(state.get("search_retrieved_docids", [])) if is_multiturn else set()
    )
    tool_calls = 0
    search_calls = 0
    error = None
    steps_run = 0

    for step in range(1, max_steps + 1):
        steps_run = step
        response = await model.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            error = "Search Agent response requires Search or Finish"
            messages.append(HumanMessage(content=error))
            continue
        should_finish = False
        final: dict[str, Any] | None = None
        for call in response.tool_calls:
            tool_calls += 1
            name = call["name"]
            args = dict(call.get("args", {}))
            if name == "Finish":
                final, validation_error = _validate_finish(args, retrieved_docids)
                if validation_error:
                    content = json.dumps(
                        {"accepted": False, "error": validation_error},
                        ensure_ascii=False,
                    )
                    error = validation_error
                else:
                    content = await tools_by_name[name].ainvoke(args)
                    should_finish = True
            elif name == "Search":
                if search_calls >= max_steps:
                    content = json.dumps(
                        {"error": "Search budget exhausted; call Finish"},
                        ensure_ascii=False,
                    )
                    error = "Search budget exhausted; call Finish"
                else:
                    search_calls += 1
                    try:
                        content = await tools_by_name[name].ainvoke(args)
                        results = _normalize_results(content)
                        prior_docids = set(retrieved_docids)
                        retrieved_docids.update(item["docid"] for item in results)
                        trace.append(
                            {
                                "query": str(args.get("query", "")),
                                "results": [
                                    {"docid": item["docid"], "score": item["score"]}
                                    for item in results
                                ],
                            }
                        )
                        # Preserve the real append-only trajectory while avoiding
                        # repeated copies of passages already present earlier in
                        # that trajectory. Scores/docids remain visible and full
                        # text stays in its first observation.
                        observation = []
                        for item in results:
                            row = dict(item)
                            if item["docid"] in prior_docids:
                                row["snippet"] = (
                                    "[Passage omitted: this docid was returned by "
                                    "an earlier Search call.]"
                                )
                            observation.append(row)
                        content = json.dumps(observation, ensure_ascii=False)
                    except Exception as exc:
                        content = json.dumps(
                            {"error": f"{type(exc).__name__}: {exc}"},
                            ensure_ascii=False,
                        )
                        error = f"{type(exc).__name__}: {exc}"
            else:
                content = json.dumps({"error": f"Unknown tool: {name}"})
                error = f"Unknown tool: {name}"
            messages.append(
                ToolMessage(content=str(content), tool_call_id=call["id"], name=name)
            )
        if should_finish and final is not None:
            result = {
                "search_explanation": final["explanation"],
                "search_exact_answer": final["exact_answer"],
                "search_citations": final["citations"],
                "search_confidence": final["confidence"],
                "search_retrieval_trace": trace,
                "search_retrieved_docids": sorted(retrieved_docids),
                "search_steps": step,
                "search_tool_calls": tool_calls,
                "search_error": None,
                "current_node": "search_agent",
            }
            if is_multiturn:
                result["search_agent_messages"] = messages
            return result
        if search_calls >= max_steps:
            break

    messages.append(
        HumanMessage(
            content=(
                "The search budget is exhausted. Use the evidence already retrieved "
                "and call Finish now. Do not search again. The exact_answer must "
                "contain only the shortest canonical answer, with supporting detail "
                "kept in explanation."
            )
        )
    )
    finish_tool = tools_by_name["Finish"]
    # Some OpenAI-compatible thinking endpoints reject forced tool_choice.
    # Supplying only Finish preserves a deterministic tool surface without
    # sending provider-specific forcing parameters.
    forced_model = _llm(config).bind_tools([finish_tool])
    response = await forced_model.ainvoke(messages)
    messages.append(response)
    final = None
    if response.tool_calls and response.tool_calls[0]["name"] == "Finish":
        tool_calls += 1
        args = dict(response.tool_calls[0].get("args", {}))
        final, error = _validate_finish(args, retrieved_docids)
        messages.append(
            ToolMessage(
                content=json.dumps(
                    {"accepted": final is not None, "error": error},
                    ensure_ascii=False,
                ),
                tool_call_id=response.tool_calls[0]["id"],
                name="Finish",
            )
        )
    else:
        error = "Forced final response did not call Finish"
    if final is not None:
        result = {
            "search_explanation": final["explanation"],
            "search_exact_answer": final["exact_answer"],
            "search_citations": final["citations"],
            "search_confidence": final["confidence"],
            "search_retrieval_trace": trace,
            "search_retrieved_docids": sorted(retrieved_docids),
            "search_steps": steps_run + 1,
            "search_tool_calls": tool_calls,
            "search_error": None,
            "current_node": "search_agent",
        }
        if is_multiturn:
            result["search_agent_messages"] = messages
        return result

    result = {
        "search_retrieval_trace": trace,
        "search_retrieved_docids": sorted(retrieved_docids),
        "search_steps": steps_run + 1,
        "search_tool_calls": tool_calls,
        "search_error": error or "Search Agent finalization failed",
        "current_node": "search_agent",
    }
    if is_multiturn:
        result["search_agent_messages"] = messages
    return result


def search_submit_node(state: State) -> dict[str, Any]:
    answer = state.get("search_exact_answer", "")
    if not answer:
        return {"final_report": "", "current_node": "search_submit"}
    confidence = float(state.get("search_confidence", 0.0))
    report = (
        f"Explanation: {state.get('search_explanation', '')}\n"
        f"Exact Answer: {answer}\n"
        f"Confidence: {confidence:g}%"
    )
    return {"final_report": report, "current_node": "search_submit"}


def build_search_graph() -> StateGraph:
    """Build Central -> Search Agent -> Submit without changing other profiles."""
    builder = StateGraph(State)
    builder.add_node("search_central", search_central_node)
    builder.add_node("search_agent", search_agent_node)
    builder.add_node("search_submit", search_submit_node)
    builder.add_edge(START, "search_central")
    builder.add_edge("search_central", "search_agent")
    builder.add_edge("search_agent", "search_submit")
    builder.add_edge("search_submit", END)
    return builder


def build_search_multiturn_graph() -> StateGraph:
    """Build the checkpointed append-only Search graph."""
    builder = StateGraph(State)
    builder.add_node("search_central", search_multiturn_central_node)
    builder.add_node("search_agent", search_agent_node)
    builder.add_node("search_submit", search_submit_node)
    builder.add_edge(START, "search_central")
    builder.add_edge("search_central", "search_agent")
    builder.add_edge("search_agent", "search_submit")
    builder.add_edge("search_submit", END)
    return builder
