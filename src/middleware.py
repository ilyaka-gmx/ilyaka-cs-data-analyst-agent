"""
Middleware harness for the Customer Service Data Analyst Agent.

Uses LangChain's AgentMiddleware system (langchain>=1.3.0) with composable hooks.
Middleware instances are passed to ``create_agent(middleware=[...])`` which wires
them into the agent's ReAct loop automatically.

Stack (outermost → innermost):
  1. TokenTrackingMiddleware  — wrap_model_call: accumulates per-query token usage
  2. ToolTimingMiddleware     — wrap_tool_call: records execution time per tool call
  3. SummarizationMiddleware  — before_model (prebuilt): compresses old messages

ToolBounds note:
  Tool input bounds (e.g. get_examples n ∈ [1, 20]) are enforced by Pydantic
  ``Field(ge=..., le=...)`` on each tool's input schema.  A separate middleware
  is unnecessary because Pydantic validation runs before the tool function body,
  rejecting out-of-range values with a clear error message.
"""

import logging
import time
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token Tracking
# ---------------------------------------------------------------------------


class TokenTrackingMiddleware(AgentMiddleware):
    """Accumulates prompt / completion token counts across LLM calls.

    Reads ``usage_metadata`` from the AIMessage returned by the model.
    Call ``reset_query()`` at the start of each user turn and ``summary()``
    at the end to get a human-readable per-query report.
    """

    def __init__(self) -> None:
        self.query_prompt_tokens: int = 0
        self.query_completion_tokens: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.calls: list[dict[str, int]] = []

    def reset_query(self) -> None:
        self.query_prompt_tokens = 0
        self.query_completion_tokens = 0
        self.calls = []

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler,
    ) -> ModelResponse:
        response = handler(request)
        ai_msg = response.result[0] if response.result else None
        if ai_msg is not None and hasattr(ai_msg, "usage_metadata") and ai_msg.usage_metadata:
            usage = ai_msg.usage_metadata
            prompt = usage.get("input_tokens", 0)
            completion = usage.get("output_tokens", 0)
            self.query_prompt_tokens += prompt
            self.query_completion_tokens += completion
            self.total_prompt_tokens += prompt
            self.total_completion_tokens += completion
            self.calls.append({"prompt": prompt, "completion": completion})
        return response

    def summary(self) -> str:
        return (
            f"Tokens: {self.query_prompt_tokens:,} in + "
            f"{self.query_completion_tokens:,} out = "
            f"{self.query_prompt_tokens + self.query_completion_tokens:,} total"
        )

    def session_summary(self) -> str:
        return (
            f"Session totals: {self.total_prompt_tokens:,} in + "
            f"{self.total_completion_tokens:,} out = "
            f"{self.total_prompt_tokens + self.total_completion_tokens:,} total"
        )


# ---------------------------------------------------------------------------
# Tool Timing
# ---------------------------------------------------------------------------


class ToolTimingMiddleware(AgentMiddleware):
    """Records wall-clock execution time for every tool call."""

    def __init__(self) -> None:
        self.log: list[dict[str, Any]] = []

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> "ToolMessage":
        tool_name = request.tool_call.get("name", "unknown")
        start = time.perf_counter()
        result = handler(request)
        duration_ms = int((time.perf_counter() - start) * 1000)
        self.log.append({"tool": tool_name, "duration_ms": duration_ms})
        log.debug("Tool %s completed in %dms", tool_name, duration_ms)
        return result

    def reset_query(self) -> None:
        self.log = []
