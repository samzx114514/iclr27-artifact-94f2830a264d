"""Dedicated single-turn SWE-bench graph."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph

from src.llms.llm import get_llm_by_type
from src.swe.workspace import SWEWorkspace

from .types import State


CENTRAL_PROMPT = """You are the Central Agent in a single-turn SWE-bench workflow.
Read only the public problem statement below. Produce a concise investigation and
implementation plan for the SWE Coder. Do not claim to have inspected files, do not
invent test results, and do not write a patch.

Public task or conversation transcript:
{problem_statement}
"""

CODER_PROMPT = """You are the SWE Coder Agent working on a SWE-bench Verified task.
The repository is already checked out at the correct base commit in /testbed.

Use the bash tool to inspect the real repository, edit production code, and run
focused tests. Work only in /testbed. Do not look for benchmark metadata, hidden
tests, a gold patch, or grader feedback. Do not modify tests merely to make them
pass. Do not install packages or download external source; network access is
disabled and the checked-out repository is the source of truth. Keep the fix
minimal. When the workspace is ready, call finish. The runtime will export git
diff, so never rewrite the full patch in your natural-language answer.

Public task or conversation transcript:
{problem_statement}

Central Agent plan:
{plan}
"""

MULTITURN_CENTRAL_PROMPT = """You are the Central Agent in a multi-turn SWE workflow.
User turns arrive chronologically and later requests may change priority. Maintain
continuity with your complete prior Central Agent message history. For the newest
turn, produce a concise plan for the SWE Coder. Respect read-only requests and do
not claim to have used tools yourself.
"""

MULTITURN_CODER_PROMPT = """You are the SWE Coder Agent in a multi-turn software
engineering conversation. The same repository workspace persists at /testbed and
your complete prior SWE Coder message and tool history is appended across turns.

Follow each newest user turn exactly. A turn may request investigation only, in
which case inspect as needed but do not edit files. A later turn may change
priority. Use only the bash tool in /testbed. Never seek benchmark metadata, hidden
tests, a gold patch, or grader feedback. Do not modify tests merely to make them
pass, install packages, or download source. When the current turn is complete,
call finish. The runtime exports git diff deterministically.
"""


def _configurable(config: RunnableConfig) -> dict[str, Any]:
    return dict(config.get("configurable", {}))


def _workspace(config: RunnableConfig) -> SWEWorkspace:
    workspace = _configurable(config).get("swe_workspace")
    if workspace is None:
        raise RuntimeError("sp_swe requires configurable.swe_workspace")
    return workspace


def _llm(config: RunnableConfig):
    return _configurable(config).get("swe_llm") or get_llm_by_type("basic")


async def swe_central_node(state: State, config: RunnableConfig) -> dict[str, Any]:
    problem = state.get("user_query", "")
    if not problem:
        raise ValueError("sp_swe requires a non-empty problem_statement")
    response = await _llm(config).ainvoke(
        [HumanMessage(content=CENTRAL_PROMPT.format(problem_statement=problem))]
    )
    plan = str(response.content).strip()
    if not plan:
        raise RuntimeError("SWE Central Agent returned an empty plan")
    return {
        "swe_plan": plan,
        "current_node": "swe_central",
    }


async def swe_multiturn_central_node(
    state: State, config: RunnableConfig
) -> dict[str, Any]:
    """Append the newest public user turn to the Central Agent's own thread."""
    user_turn = state.get("user_query", "")
    if not user_turn:
        raise ValueError("sp_swe_multiturn requires a non-empty user turn")
    history = list(state.get("swe_central_messages", []))
    if not history:
        history.append(SystemMessage(content=MULTITURN_CENTRAL_PROMPT))
    history.append(HumanMessage(content=user_turn))
    response = await _llm(config).ainvoke(history)
    history.append(response)
    plan = str(response.content).strip()
    if not plan:
        raise RuntimeError("SWE Central Agent returned an empty plan")
    return {
        "swe_central_messages": history,
        "swe_plan": plan,
        "current_node": "swe_central",
    }


def _tools(workspace: SWEWorkspace) -> list[StructuredTool]:
    def bash(command: str, timeout: int = 120) -> str:
        """Run one bash command in the isolated /testbed workspace."""
        result = workspace.execute(command, timeout=timeout)
        return json.dumps(result, ensure_ascii=False)

    def finish(summary: str = "") -> str:
        """Finish workspace work; the runtime will export the actual git diff."""
        return summary or "ready for deterministic patch export"

    return [
        StructuredTool.from_function(bash, name="bash"),
        StructuredTool.from_function(finish, name="finish"),
    ]


async def swe_coder_node(state: State, config: RunnableConfig) -> dict[str, Any]:
    configurable = _configurable(config)
    max_steps = int(configurable.get("swe_max_steps", 40))
    workspace = _workspace(config)
    tools = _tools(workspace)
    tools_by_name = {tool.name: tool for tool in tools}
    model = _llm(config).bind_tools(tools)
    messages = [
        SystemMessage(
            content=CODER_PROMPT.format(
                problem_statement=state.get("user_query", ""),
                plan=state.get("swe_plan", ""),
            )
        ),
        HumanMessage(content="Inspect the workspace and implement the requested fix."),
    ]
    tool_calls = 0
    error = None
    latest_summary = ""

    for step in range(1, max_steps + 1):
        response = await model.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            latest_summary = str(response.content).strip()
            return {
                "swe_steps": step,
                "swe_tool_calls": tool_calls,
                "swe_coder_summary": latest_summary,
                "current_node": "swe_coder",
            }
        should_finish = False
        for call in response.tool_calls:
            tool_calls += 1
            name = call["name"]
            if name not in tools_by_name:
                content = f"Unknown tool: {name}"
            else:
                try:
                    content = await tools_by_name[name].ainvoke(call.get("args", {}))
                except Exception as exc:
                    content = f"{type(exc).__name__}: {exc}"
                    error = content
            messages.append(
                ToolMessage(
                    content=str(content),
                    tool_call_id=call["id"],
                    name=name,
                )
            )
            if name == "finish":
                latest_summary = str(call.get("args", {}).get("summary", "")).strip()
                should_finish = True
        if should_finish:
            return {
                "swe_steps": step,
                "swe_tool_calls": tool_calls,
                "swe_coder_summary": latest_summary,
                "swe_error": error,
                "current_node": "swe_coder",
            }

    return {
        "swe_steps": max_steps,
        "swe_tool_calls": tool_calls,
        "swe_coder_summary": latest_summary,
        "swe_error": "SWE Coder reached the step limit",
        "current_node": "swe_coder",
    }


async def swe_multiturn_coder_node(
    state: State, config: RunnableConfig
) -> dict[str, Any]:
    """Continue the Coder's raw append-only assistant/tool message stream."""
    configurable = _configurable(config)
    max_steps = int(configurable.get("swe_max_steps", 40))
    workspace = _workspace(config)
    tools = _tools(workspace)
    tools_by_name = {tool.name: tool for tool in tools}
    model = _llm(config).bind_tools(tools)
    messages = list(state.get("swe_coder_messages", []))
    if not messages:
        messages.append(SystemMessage(content=MULTITURN_CODER_PROMPT))
    messages.append(
        HumanMessage(
            content=(
                "Newest public user turn:\n"
                f"{state.get('user_query', '')}\n\n"
                "Central Agent plan for this turn:\n"
                f"{state.get('swe_plan', '')}"
            )
        )
    )
    tool_calls = 0
    error = None
    latest_response = ""

    for step in range(1, max_steps + 1):
        response = await model.ainvoke(messages)
        messages.append(response)
        response_text = str(response.content).strip()
        if response_text:
            latest_response = response_text
        if not response.tool_calls:
            return {
                "swe_coder_messages": messages,
                "swe_coder_response": latest_response,
                "swe_steps": step,
                "swe_tool_calls": tool_calls,
                "swe_error": error,
                "current_node": "swe_coder",
            }
        should_finish = False
        for call in response.tool_calls:
            tool_calls += 1
            name = call["name"]
            if name not in tools_by_name:
                content = f"Unknown tool: {name}"
            else:
                try:
                    content = await tools_by_name[name].ainvoke(call.get("args", {}))
                except Exception as exc:
                    content = f"{type(exc).__name__}: {exc}"
                    error = content
            messages.append(
                ToolMessage(content=str(content), tool_call_id=call["id"], name=name)
            )
            if name == "finish":
                finish_text = str(call.get("args", {}).get("summary", "")).strip()
                if finish_text:
                    latest_response = finish_text
                should_finish = True
        if should_finish:
            return {
                "swe_coder_messages": messages,
                "swe_coder_response": latest_response,
                "swe_steps": step,
                "swe_tool_calls": tool_calls,
                "swe_error": error,
                "current_node": "swe_coder",
            }

    return {
        "swe_coder_messages": messages,
        "swe_coder_response": latest_response,
        "swe_steps": max_steps,
        "swe_tool_calls": tool_calls,
        "swe_error": "SWE Coder reached the step limit",
        "current_node": "swe_coder",
    }


def swe_submit_node(state: State, config: RunnableConfig) -> dict[str, Any]:
    patch = _workspace(config).git_diff()
    return {
        "swe_patch": patch,
        "final_report": patch,
        "current_node": "submit",
    }


def build_swe_graph() -> StateGraph:
    """Build Central Agent -> SWE Coder -> Submit orchestration."""
    builder = StateGraph(State)
    builder.add_node("swe_central", swe_central_node)
    builder.add_node("swe_coder", swe_coder_node)
    builder.add_node("submit", swe_submit_node)
    builder.add_edge(START, "swe_central")
    builder.add_edge("swe_central", "swe_coder")
    builder.add_edge("swe_coder", "submit")
    builder.add_edge("submit", END)
    return builder


def build_swe_multiturn_graph() -> StateGraph:
    """Build a checkpointed Central -> Coder -> Submit turn transition."""
    builder = StateGraph(State)
    builder.add_node("swe_central", swe_multiturn_central_node)
    builder.add_node("swe_coder", swe_multiturn_coder_node)
    builder.add_node("submit", swe_submit_node)
    builder.add_edge(START, "swe_central")
    builder.add_edge("swe_central", "swe_coder")
    builder.add_edge("swe_coder", "submit")
    builder.add_edge("submit", END)
    return builder
