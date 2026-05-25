"""
Agent graph: outer router + inner manual ReAct loop with middleware hooks.

Architecture:
  Outer StateGraph with four nodes: router, agent_step, tool_step, decline.
  The agent_step <-> tool_step pair forms a manual ReAct loop that replaces
  create_agent's internal black-box graph.

  Middleware (TokenTrackingMiddleware, ToolTimingMiddleware) are real
  AgentMiddleware subclasses whose hook methods (wrap_model_call,
  wrap_tool_call) are invoked explicitly:
    - wrap_model_call: called directly in agent_step with proper
      ModelRequest/ModelResponse objects
    - wrap_tool_call: passed to ToolNode(wrap_tool_call=...) which
      handles ToolCallRequest construction internally

Why manual loop instead of create_agent:
  1. create_agent causes infinite tool-calling loops with Llama-3.3-70B
  2. create_agent cannot parse DeepSeek-V3.2's DSML XML tool-call format
  3. Manual loop allows loop detection, iteration control, and DSML repair
  See "Rejected Models" in assignment_master_plan.md for details.
"""

import json
import logging
import re
import uuid as _uuid
from typing import Literal, Optional

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from src.config import AGENT_MODEL, MAX_ITERATIONS, get_llm
from src.middleware import TokenTrackingMiddleware, ToolTimingMiddleware
from src.prompts import AGENT_SYSTEM_PROMPT, DECLINE_MESSAGE
from src.router import classify_query
from src.tools import get_all_tools, get_tools_for_query_type, set_current_user_id


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(MessagesState):
    """Extended state with routing, user context, and iteration tracking."""

    query_type: Optional[Literal["structured", "unstructured", "recommend", "out_of_scope"]] = None
    user_id: str = "default"
    iteration_count: int = 0
    use_past_sessions: bool = False
    thread_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Middleware instances (shared across invocations for accumulation)
# ---------------------------------------------------------------------------

token_tracker = TokenTrackingMiddleware()
tool_timer = ToolTimingMiddleware()


# ---------------------------------------------------------------------------
# Tools and ToolNode (lazy-initialized once)
# ---------------------------------------------------------------------------

_tool_node: ToolNode | None = None


def _get_tool_node() -> ToolNode:
    """Lazy-build the ToolNode with wrap_tool_call middleware wired in."""
    global _tool_node
    if _tool_node is None:
        _tool_node = ToolNode(
            get_all_tools(),
            wrap_tool_call=tool_timer.wrap_tool_call,
        )
    return _tool_node


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def router_node(state: AgentState) -> dict:
    """Classify the latest user message via the router model."""
    last_message = state["messages"][-1].content
    result = classify_query(last_message)
    return {"query_type": result.classification, "iteration_count": 0}


def agent_step(state: AgentState) -> dict:
    """One step of the ReAct loop: invoke LLM with tools bound.

    Includes iteration limits, loop detection, middleware token tracking,
    and DSML XML repair for DeepSeek backend issues.
    """
    messages = state["messages"]
    iteration = state.get("iteration_count", 0)

    if iteration >= MAX_ITERATIONS:
        fallback = AIMessage(content=(
            "I wasn't able to complete the analysis within the allowed steps. "
            "Could you try rephrasing your question?"
        ))
        return {"messages": [fallback], "iteration_count": iteration}

    force_text = _is_loop_detected(messages)

    set_current_user_id(state.get("user_id", "default"))
    llm = get_llm(AGENT_MODEL, temperature=0)

    query_type = state.get("query_type", "structured")
    exposed_tools = get_tools_for_query_type(query_type)

    if force_text:
        bound_llm = llm
        inject = [HumanMessage(content=(
            "You already called tools with these arguments. "
            "Please provide your final answer based on the results you have."
        ))]
    else:
        bound_llm = llm.bind_tools(exposed_tools)
        inject = []

    sys_prompt = AGENT_SYSTEM_PROMPT
    if query_type == "recommend" and iteration == 0:
        log.info("Recommendation mode: user=%s", state.get("user_id", "default"))

        sys_prompt += (
            "\n\nYou are in RECOMMENDATION MODE.\n\n"
            "MANDATORY FIRST STEPS — do these BEFORE generating any recommendations:\n"
            "1. Call recall_past_sessions(query_type_filter='structured') to retrieve "
            "the user's actual past business questions.\n"
            "2. Call recall_profile() to retrieve the user's profile facts.\n\n"
            "AFTER you have the tool results:\n"
            "- Base your recommendations on the ACTUAL past queries returned by the tool.\n"
            "- Quote the user's real past questions verbatim — NEVER make up questions.\n"
            "- Suggest 2-3 follow-up queries that build on what they actually explored.\n"
            "- Explain briefly why each suggestion is interesting.\n"
            "- Ask which one they'd like to try, or if they want something different.\n"
            "- Do NOT execute any data queries — only suggest and wait for confirmation.\n"
            "- Do NOT say you have no access to past sessions — use recall_past_sessions."
        )

    sys_msg = SystemMessage(content=sys_prompt)
    trimmed = _trim_messages(list(messages), max_messages=30)
    invoke_messages = [sys_msg] + trimmed + inject

    request = ModelRequest(
        model=bound_llm,
        messages=invoke_messages,
        system_message=None,
        tools=[] if force_text else exposed_tools,
        tool_choice=None,
    )

    def _handler(req: ModelRequest) -> ModelResponse:
        response = req.model.invoke(req.messages)
        return ModelResponse(result=[response])

    model_response = token_tracker.wrap_model_call(request, _handler)
    ai_msg = model_response.result[0]

    ai_msg = _repair_dsml_xml(ai_msg)

    return {
        "messages": [ai_msg],
        "iteration_count": iteration + 1,
    }


def tool_step(state: AgentState) -> dict:
    """Execute tool calls via ToolNode (with wrap_tool_call middleware)."""
    set_current_user_id(state.get("user_id", "default"))
    return _get_tool_node().invoke(state)


def decline_node(state: AgentState) -> dict:
    """Return a polite out-of-scope decline (no LLM call)."""
    return {"messages": [AIMessage(content=DECLINE_MESSAGE)]}


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------


def route_after_router(state: AgentState) -> str:
    if state.get("query_type") == "out_of_scope":
        return "decline"
    return "agent_step"


def should_continue(state: AgentState) -> str:
    """After agent_step: if tool_calls present -> execute tools; else -> done."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_step"
    return END


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph(checkpointer=None):
    """Build and compile the agent graph: router -> manual ReAct loop -> END.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. SqliteSaver)
                      for conversation persistence. Pass None for tests.

    Returns:
        Compiled StateGraph ready for ``.invoke()``.
    """
    if isinstance(checkpointer, dict):
        checkpointer = True

    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("decline", decline_node)
    graph.add_node("agent_step", agent_step)
    graph.add_node("tool_step", tool_step)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_after_router, {
        "agent_step": "agent_step",
        "decline": "decline",
    })
    graph.add_conditional_edges("agent_step", should_continue, {
        "tool_step": "tool_step",
        END: END,
    })
    graph.add_edge("tool_step", "agent_step")
    graph.add_edge("decline", END)

    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trim_messages(messages: list, max_messages: int = 30) -> list:
    """Keep last N messages, preserving AI/ToolMessage pairs at the boundary.

    Replaces SummarizationMiddleware with simple truncation.
    Trade-off: no LLM call overhead, but long conversations lose early context
    instead of getting a summary. Acceptable for short interactive sessions.
    """
    if len(messages) <= max_messages:
        return messages
    cut = len(messages) - max_messages
    while cut < len(messages) and isinstance(messages[cut], ToolMessage):
        cut += 1
    return messages[cut:]


def _is_loop_detected(messages: list) -> bool:
    """Check if the last two AI tool-call sets have identical signatures.

    Mirrors the loop detection in Rotem Levi's reference implementation:
    if the last two tool calls used the same tool with the same args,
    force the model to respond with text instead of calling tools again.
    """
    recent_sigs: list[tuple] = []
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            sig = tuple(
                (tc["name"], str(tc.get("args", {})))
                for tc in msg.tool_calls
            )
            recent_sigs.append(sig)
            if len(recent_sigs) >= 2:
                break
        elif isinstance(msg, HumanMessage):
            break
    return len(recent_sigs) >= 2 and recent_sigs[0] == recent_sigs[1]


def _repair_dsml_xml(response: AIMessage) -> AIMessage:
    """Parse DSML XML tool calls leaked into response content.

    When Nebius vLLM/SGLang fails to parse DeepSeek's tool-call format,
    raw XML appears in content with empty tool_calls. This extracts
    tool calls from two known parseable patterns:
      1. <tool_call>{"name":...,"arguments":...}</tool_call>
      2. <functioncall>{"name":...,"arguments":...}</functioncall>

    For the raw DSML Unicode tag format, we log a warning and fall back
    to the model's text response. That pattern indicates a severe backend
    failure that cannot be reliably recovered from.
    """
    if response.tool_calls or not response.content:
        return response

    content = response.content

    for tag in ("tool_call", "functioncall"):
        match = re.search(
            rf"<{tag}>\s*(\{{.*?\}})\s*</{tag}>", content, re.DOTALL
        )
        if match:
            try:
                parsed = json.loads(match.group(1))
                tool_name = parsed.get("name", "")
                tool_args = parsed.get("arguments", {})
                if isinstance(tool_args, str):
                    tool_args = json.loads(tool_args)
                if tool_name:
                    repaired = AIMessage(
                        content="",
                        tool_calls=[{
                            "name": tool_name,
                            "args": tool_args,
                            "id": f"repaired_{_uuid.uuid4().hex[:8]}",
                        }],
                    )
                    log.warning(
                        "DSML repair: parsed %s from XML: %s(%s)",
                        tag, tool_name, tool_args,
                    )
                    return repaired
            except (json.JSONDecodeError, KeyError):
                pass

    if "DSML" in content or "\uff5c" in content:
        log.warning("DSML XML detected but not parseable: %s", content[:200])

    return response
