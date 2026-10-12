from typing import Literal

# Define available LLM types
LLMType = Literal["basic", "reasoning", "vision", "report"]

# Define agent-LLM mapping
AGENT_LLM_MAP: dict[str, LLMType] = {
    "coordinator": "basic",
    "planner": "basic",
    "researcher": "basic",
    "coder": "basic",
    "reporter": "basic",
    "speech": "basic",
    "central_agent": "basic",
    "conclusion": "basic",
    "sql_agent": "basic",
    "sql_conclusion": "basic",
    "swe_central": "basic",
    "swe_coder": "basic",
    "researcher_xxqg": "basic",
    "researcher_web": "basic",
    "replanner": "basic",
    "perception": "basic",
    "outline": "basic",
    "reporter_xxqg": "report",
}
