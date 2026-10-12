"""Task-specific StackPlanner profiles.

Each task family owns its graph, SOP, allowed sub-agents, terminal agent, and
memory policy. Task families use separate profiles rather than branching inside
another task's graph.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskGraphProfile:
    graph_format: str
    task_family: str
    sop_name: str
    sub_agents: tuple[tuple[str, str], ...]
    terminal_agent: str
    supports_checkpoint_memory: bool = True


MATH_PROFILE = TaskGraphProfile(
    graph_format="sp_math",
    task_family="math",
    sop_name="math",
    sub_agents=(
        ("coder", "Verify non-trivial arithmetic or symbolic calculations"),
        ("conclusion", "Produce a concise, verified mathematical conclusion"),
    ),
    terminal_agent="conclusion",
)

SQL_PROFILE = TaskGraphProfile(
    graph_format="sp_sql",
    task_family="sql",
    sop_name="sql",
    sub_agents=(
        ("sql_agent", "Draft a SQLite query from the active conversational intent"),
        ("sql_conclusion", "Return the final query in one SQL code block"),
    ),
    terminal_agent="sql_conclusion",
)

SWE_PROFILE = TaskGraphProfile(
    graph_format="sp_swe",
    task_family="swe",
    sop_name="swe",
    sub_agents=(
        ("swe_coder", "Inspect, modify, and test the isolated code workspace"),
        ("submit", "Export the final workspace diff"),
    ),
    terminal_agent="submit",
    supports_checkpoint_memory=False,
)

SWE_MULTITURN_PROFILE = TaskGraphProfile(
    graph_format="sp_swe_multiturn",
    task_family="swe",
    sop_name="swe",
    sub_agents=(
        ("swe_coder", "Continue the append-only isolated code workspace thread"),
        ("submit", "Export the current workspace diff"),
    ),
    terminal_agent="submit",
    supports_checkpoint_memory=True,
)

SEARCH_PROFILE = TaskGraphProfile(
    graph_format="sp_search",
    task_family="search",
    sop_name="search",
    sub_agents=(
        ("search_agent", "Search the read-only corpus and join retrieved evidence"),
        ("search_submit", "Return the exact answer with verified citations"),
    ),
    terminal_agent="search_submit",
    supports_checkpoint_memory=False,
)

SEARCH_MULTITURN_PROFILE = TaskGraphProfile(
    graph_format="sp_search_multiturn",
    task_family="search",
    sop_name="search_multiturn",
    sub_agents=(
        ("search_agent", "Continue the append-only evidence search thread"),
        ("search_submit", "Return the current turn's exact answer"),
    ),
    terminal_agent="search_submit",
    supports_checkpoint_memory=True,
)

DEEP_RESEARCH_PROFILE = TaskGraphProfile(
    graph_format="sp_deep_research",
    task_family="deep_research",
    sop_name="deep_research",
    sub_agents=(
        ("outline", "Generate the report outline from the active public brief"),
        ("researcher", "Collect web evidence for the confirmed outline"),
        ("reporter", "Write one report using inline document-id citations"),
    ),
    terminal_agent="submit",
    supports_checkpoint_memory=True,
)


TASK_GRAPH_PROFILES = {
    MATH_PROFILE.graph_format: MATH_PROFILE,
    SQL_PROFILE.graph_format: SQL_PROFILE,
    SWE_PROFILE.graph_format: SWE_PROFILE,
    SWE_MULTITURN_PROFILE.graph_format: SWE_MULTITURN_PROFILE,
    SEARCH_PROFILE.graph_format: SEARCH_PROFILE,
    SEARCH_MULTITURN_PROFILE.graph_format: SEARCH_MULTITURN_PROFILE,
    DEEP_RESEARCH_PROFILE.graph_format: DEEP_RESEARCH_PROFILE,
}


def get_task_graph_profile(graph_format: str) -> TaskGraphProfile | None:
    return TASK_GRAPH_PROFILES.get(graph_format)
