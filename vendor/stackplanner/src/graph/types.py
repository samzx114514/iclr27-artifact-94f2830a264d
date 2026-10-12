from langgraph.graph import MessagesState

from src.prompts.planner_model import Plan
from src.rag import Resource
from typing import Any


class State(MessagesState):
    """State for the agent system, extends MessagesState with next field."""

    # Runtime Variables
    locale: str = "en-US"  # "zh-CN"
    observations: list[str] = []
    data_collections: list[Any] = []
    resources: list[Resource] = []
    plan_iterations: int = 0
    current_plan: Plan | str = None
    user_query: str = ""
    final_report: str = ""
    replan_result: str = ""
    auto_accepted_plan: bool = False
    enable_background_investigation: bool = True
    background_investigation_results: str = None
    user_dst: str = None
    wait_stage: str = ""
    dst_question: str = None
    hitl_feedback: str = ""
    report_outline: str = None
    skip_perception: bool = False

    delegation_context: dict = None
    current_node: str = None
    memory_stack: str = None

    # FactStruct Stage 1 相关字段
    factstruct_outline: Any = None  # OutlineNode 根节点
    factstruct_memory: Any = None  # Memory 实例（需要序列化时转换为 dict）
    total_word_limit: int = 5000  # 用户指定的总字数限制
    ref_map: dict = None
    outline_feedback: str = None
    # ZB V1.1相关字段
    sop: str = None

    # Single-turn SWE-bench runtime fields. The workspace itself is supplied
    # through RunnableConfig and is never checkpointed into graph state.
    swe_plan: str = ""
    swe_coder_summary: str = ""
    # Native multi-turn SWE histories. Each agent owns an append-only message
    # stream; LangGraph checkpoint memory carries these streams across turns.
    swe_central_messages: list[Any] = []
    swe_coder_messages: list[Any] = []
    swe_coder_response: str = ""
    swe_patch: str = ""
    swe_steps: int = 0
    swe_tool_calls: int = 0
    swe_error: str = None

    # Single-turn BrowseComp+ search fields. The read-only retriever is injected
    # through RunnableConfig and never serialized into graph state.
    search_plan: str = ""
    # Native multi-turn Search histories are checkpointed while the retriever
    # itself remains runtime-only. Full tool observations stay in the Search
    # Agent stream so later corrections can reuse earlier evidence.
    search_central_messages: list[Any] = []
    search_agent_messages: list[Any] = []
    search_explanation: str = ""
    search_exact_answer: str = ""
    search_citations: list[str] = []
    search_confidence: float = 0.0
    search_retrieval_trace: list[dict[str, Any]] = []
    search_retrieved_docids: list[str] = []
    search_steps: int = 0
    search_tool_calls: int = 0
    search_error: str = None

    # Deterministic DeepResearch pipeline fields. Public conversation history
    # is carried by MessagesState; only derived, non-evaluation state lives here.
    deep_research_brief: str = ""
    deep_research_submitted: bool = False
    deep_research_error: str = None
