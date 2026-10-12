from datetime import datetime
import time
import asyncio
import hashlib
import json
import re
from pydantic import BaseModel, Field, model_validator
from langchain_core.messages import AnyMessage, ToolMessage
from langchain.tools import BaseTool
from langgraph.types import Command
from abc import ABCMeta, abstractmethod
from langgraph.prebuilt import create_react_agent
from src.utils.logger import logger
from src.prompts import apply_prompt_template
from src.llms.llm import get_llm_by_type
from src.config.agents import AGENT_LLM_MAP
from src.agents.ReActHandler import ReactAgentCallbackHandler, ToolResultCallbackHandler

from typing import Literal, Optional, List


_TEXT_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE
)


def parse_text_tool_calls(content: object) -> tuple[list[dict], str]:
    """Convert Qwen-style textual tool blocks into LangChain tool calls."""
    if not isinstance(content, str) or "<tool_call" not in content.lower():
        return [], str(content or "")
    calls = []
    for index, match in enumerate(_TEXT_TOOL_CALL_PATTERN.finditer(content)):
        try:
            payload = json.loads(match.group(1))
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        function = payload.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            arguments = function.get("arguments", {})
        else:
            name = payload.get("name")
            arguments = payload.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        if not isinstance(name, str) or not name or not isinstance(arguments, dict):
            continue
        digest = hashlib.sha256(
            f"{index}:{name}:{match.group(1)}".encode("utf-8")
        ).hexdigest()[:16]
        calls.append(
            {
                "name": name,
                "args": arguments,
                "id": f"call_sp_qwen_{digest}",
                "type": "tool_call",
            }
        )
    remaining = _TEXT_TOOL_CALL_PATTERN.sub("", content).strip() if calls else content
    return calls, remaining


class CommonReactAgent(BaseModel, metaclass=ABCMeta):

    agent_name: str = Field(..., description="Unique name of the agent")
    description: Optional[str] = Field(None, description="Optional agent description")
    system_prompt: Optional[str] = Field(None, description="System instruction prompt")
    llm_config_name: str = Field(
        default="gpt-4o-mini", description="model wrapper name"
    )
    tools: List[BaseTool] = Field(default=[])

    tool_names: List[str] = Field(default=[])
    tool_results: List[str] = Field(default=[])

    # Execution control
    max_steps: int = Field(default=10, description="Maximum steps before termination")
    current_step: int = Field(default=0, description="Current step in execution")
    react_version: Literal["v1", "v2"] = Field(
        default="v2",
        description="LangGraph tool execution mode",
    )
    parallel_tool_calls: Optional[bool] = Field(
        default=None,
        description="Whether the model may emit parallel tool calls",
    )
    strict_tool_message_loop: bool = Field(
        default=False,
        description="Execute and append every tool result before the next model call",
    )
    strict_tool_call_limit: Optional[int] = Field(
        default=None,
        description="Maximum real tool executions in the strict loop",
    )
    text_tool_call_compat: bool = Field(
        default=False,
        description=(
            "Parse Qwen textual <tool_call> blocks in the strict research loop. "
            "Disabled by default so all existing task profiles retain native behavior."
        ),
    )
    strict_tool_result_char_limit: int | None = Field(
        default=None,
        description="Maximum characters returned to the model for one tool result",
    )
    strict_tool_history_char_limit: int | None = Field(
        default=None,
        description="Maximum cumulative tool-result characters in the strict loop",
    )

    def model_post_init(self, __context) -> None:
        model = get_llm_by_type(AGENT_LLM_MAP[self.agent_name])
        if self.parallel_tool_calls is not None and self.tools:
            model = model.bind_tools(
                self.tools,
                parallel_tool_calls=self.parallel_tool_calls,
            )
        self._model = model
        self._agent = create_react_agent(
            name=self.agent_name,
            model=model,
            tools=self.tools,
            prompt=lambda state: apply_prompt_template(self.system_prompt, state),
            version=self.react_version,
        )
        self._handler = ToolResultCallbackHandler(self)

    async def ainvoke(self, *args, **kwargs):
        from copy import deepcopy

        # 获取已有的 config 或创建一个新的 dict
        config = deepcopy(kwargs.get("config", {}))

        # 合并 callbacks
        if "callbacks" in config:
            config["callbacks"] = config["callbacks"] + [self._handler]
        else:
            config["callbacks"] = [self._handler]

        # 替换 kwargs 中的 config
        kwargs["config"] = config

        if self.strict_tool_message_loop:
            return await self._strict_ainvoke(*args, **kwargs)

        # 再调用 ainvoke
        return await self._agent.ainvoke(*args, **kwargs)

    async def _strict_ainvoke(self, *args, **kwargs):
        """Run ReAct while preserving the OpenAI tool-call pairing invariant.

        Some OpenAI-compatible providers ignore ``parallel_tool_calls=False``.
        Their response can therefore contain several calls even when sequential
        execution was requested.  Execute every call and append all matching
        ToolMessages before invoking the model again.
        """
        state = kwargs.get("input") or (args[0] if args else {})
        config = kwargs.get("config", {})
        messages = list(apply_prompt_template(self.system_prompt, state))
        tools_by_name = {tool.name: tool for tool in self.tools}
        recursion_limit = int(config.get("recursion_limit", 25))
        max_model_calls = max(1, (recursion_limit + 1) // 2)
        executed_tool_calls = 0
        tool_history_chars = 0

        for _ in range(max_model_calls):
            response = await self._model.ainvoke(messages, config=config)
            messages.append(response)
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls and self.text_tool_call_compat:
                tool_calls, remaining = parse_text_tool_calls(response.content)
                if tool_calls:
                    response.tool_calls = tool_calls
                    response.content = remaining
            if not tool_calls:
                return {"messages": messages}

            for call in tool_calls:
                tool_name = call.get("name", "")
                tool = tools_by_name.get(tool_name)
                if (
                    self.strict_tool_call_limit is not None
                    and executed_tool_calls >= self.strict_tool_call_limit
                ):
                    content = (
                        "Tool budget exhausted. Synthesize the final research "
                        "result from evidence already collected."
                    )
                elif tool is None:
                    content = f"Error: unknown tool {tool_name!r}"
                else:
                    executed_tool_calls += 1
                    try:
                        # LangChain injects runtime-only arguments into the
                        # mapping passed to a tool.  Never let those objects
                        # mutate the AIMessage tool-call payload that will be
                        # serialized on the next model request.
                        tool_input = dict(call.get("args") or {})
                        content = await tool.ainvoke(tool_input, config=config)
                    except Exception as exc:
                        content = f"Error: {exc!r}"
                content = str(content)
                if self.strict_tool_result_char_limit is not None:
                    content = content[: self.strict_tool_result_char_limit]
                if self.strict_tool_history_char_limit is not None:
                    remaining = max(
                        0, self.strict_tool_history_char_limit - tool_history_chars
                    )
                    content = content[:remaining]
                    if not content:
                        content = "Tool result omitted: research context budget exhausted."
                    tool_history_chars += min(len(content), remaining)
                messages.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=call["id"],
                        name=tool_name or None,
                    )
                )

        raise RuntimeError(
            f"Agent exceeded strict tool-loop limit ({max_model_calls} model calls)"
        )

    @abstractmethod
    def execute_agent_step(self, state) -> Command:
        """
        Execute a single step of the agent's logic.
        This method should be implemented by subclasses to define the agent's behavior.
        """
