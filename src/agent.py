"""
LangGraph agent graph for the Customer Service Data Analyst.

Architecture:
  Outer StateGraph with three nodes (router, agent, decline) provides
  query classification and routing.  The agent node delegates to an inner
  ``create_agent`` graph that runs the ReAct tool-calling loop with a
  composable middleware stack (token tracking, tool timing, conversation
  summarization).

  The outer graph owns the checkpointer (SqliteSaver) for conversation
  persistence.  The inner agent is stateless across turns — all history
  flows through the outer graph's state.
"""

from typing import Literal, Optional

from langchain.agents import create_agent
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from src.config import AGENT_MODEL, get_llm, get_summarizer_model
from src.middleware import TokenTrackingMiddleware, ToolTimingMiddleware
from src.prompts import AGENT_SYSTEM_PROMPT, DECLINE_MESSAGE
from src.router import classify_query
from src.tools import get_all_tools, set_current_user_id


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(MessagesState):
    """Extended state with routing classification and user context."""

    query_type: Optional[Literal["structured", "unstructured", "out_of_scope"]] = None
    user_id: str = "default"


# ---------------------------------------------------------------------------
# Middleware instances (shared across invocations for accumulation)
# ---------------------------------------------------------------------------

token_tracker = TokenTrackingMiddleware()
tool_timer = ToolTimingMiddleware()


def _build_middleware_stack() -> list:
    """Build the middleware list for create_agent."""
    summarizer = SummarizationMiddleware(
        model=get_llm(get_summarizer_model(), temperature=0),
        trigger=("messages", 40),
        keep=("messages", 20),
    )
    return [token_tracker, tool_timer, summarizer]


# ---------------------------------------------------------------------------
# Inner ReAct agent (built once, invoked per turn)
# ---------------------------------------------------------------------------

_inner_agent = None


def _get_inner_agent():
    """Lazy-build the inner ReAct agent with middleware."""
    global _inner_agent
    if _inner_agent is not None:
        return _inner_agent

    llm = get_llm(AGENT_MODEL, temperature=0)
    _inner_agent = create_agent(
        model=llm,
        tools=get_all_tools(),
        system_prompt=AGENT_SYSTEM_PROMPT,
        middleware=_build_middleware_stack(),
    )
    return _inner_agent


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def router_node(state: AgentState) -> dict:
    """Classify the latest user message via the router model."""
    last_message = state["messages"][-1].content
    result = classify_query(last_message)
    return {"query_type": result.classification}


def agent_node(state: AgentState) -> dict:
    """Run the ReAct agent with tools and middleware."""
    set_current_user_id(state.get("user_id", "default"))
    inner = _get_inner_agent()
    result = inner.invoke(
        {"messages": state["messages"]},
        {"recursion_limit": 25},
    )
    return {"messages": result["messages"]}


def decline_node(state: AgentState) -> dict:
    """Return a polite out-of-scope decline (no LLM call)."""
    return {"messages": [AIMessage(content=DECLINE_MESSAGE)]}


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------


def route_after_router(state: AgentState) -> str:
    if state.get("query_type") == "out_of_scope":
        return "decline"
    return "agent"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph(checkpointer=None):
    """Build and compile the outer agent graph.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. SqliteSaver)
                      for conversation persistence. Pass None for tests.

    Returns:
        Compiled StateGraph ready for ``.invoke()``.
    """
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("agent", agent_node)
    graph.add_node("decline", decline_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"agent": "agent", "decline": "decline"},
    )
    graph.add_edge("agent", END)
    graph.add_edge("decline", END)

    if isinstance(checkpointer, dict):
        checkpointer = True

    return graph.compile(checkpointer=checkpointer)
