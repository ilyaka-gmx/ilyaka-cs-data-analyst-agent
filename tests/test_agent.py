"""Integration tests for the full agent graph.

Slow tests make real LLM calls.
Run: uv run pytest tests/test_agent.py -v
"""

import sqlite3

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agent import build_graph


# --- Fast tests (no LLM calls) ---


def test_build_graph_with_dict_checkpointer():
    """langgraph-api 0.8.7 passes a dict config as checkpointer; build_graph must handle it."""
    graph = build_graph(checkpointer={"some": "config"})
    assert graph is not None


@pytest.fixture
def graph():
    """Build graph without checkpointer for stateless tests."""
    return build_graph(checkpointer=None)


@pytest.fixture
def config():
    return {"configurable": {"thread_id": "test"}, "recursion_limit": 12}


def _invoke(graph, query: str, config: dict, user_id: str = "test") -> str:
    """Invoke the graph and return the final AI message content."""
    result = graph.invoke(
        {"messages": [HumanMessage(content=query)], "user_id": user_id},
        config,
    )
    return result["messages"][-1].content


# --- Structured queries ---


@pytest.mark.slow
def test_structured_categories(graph, config):
    final = _invoke(graph, "What categories exist?", config)
    assert "ORDER" in final.upper()
    assert "REFUND" in final.upper()


@pytest.mark.slow
def test_structured_count(graph, config):
    final = _invoke(graph, "How many refund requests did we get?", config)
    assert any(char.isdigit() for char in final)


@pytest.mark.slow
def test_structured_examples(graph, config):
    final = _invoke(graph, "Show me 3 examples from SHIPPING.", config)
    assert "shipping" in final.lower() or "SHIPPING" in final


@pytest.mark.slow
def test_structured_distribution(graph, config):
    final = _invoke(
        graph,
        "What is the distribution of intents in the ACCOUNT category?",
        config,
    )
    assert any(word in final.lower() for word in ("create_account", "edit_account", "account"))


# --- Unstructured queries ---


@pytest.mark.slow
def test_unstructured_summarize(graph, config):
    final = _invoke(graph, "Summarize the FEEDBACK category.", config)
    assert len(final) > 100


# --- Out-of-scope ---


@pytest.mark.slow
def test_out_of_scope(graph, config):
    final = _invoke(graph, "Who won the Champions League?", config)
    assert "dataset" in final.lower() or "customer service" in final.lower()
    assert "real madrid" not in final.lower()


# --- Persistence (SqliteSaver with :memory:) ---


@pytest.mark.slow
def test_persistence_across_turns():
    """Verify that conversation context survives across separate invocations."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = build_graph(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "persist_test"}, "recursion_limit": 12}

    _invoke(graph, "How many refund requests did we get?", config)
    final = _invoke(graph, "What about complaints?", config)

    assert any(char.isdigit() for char in final)
    conn.close()


# --- Multi-step reasoning ---


@pytest.mark.slow
def test_multi_step_shows_tool_calls(graph, config):
    """Verify the agent actually calls tools (not just answering from memory)."""
    result = graph.invoke(
        {"messages": [HumanMessage(content="How many refund requests?")], "user_id": "test"},
        config,
    )
    messages = result["messages"]
    from langchain_core.messages import AIMessage

    tool_calls_found = any(
        isinstance(m, AIMessage) and m.tool_calls for m in messages
    )
    assert tool_calls_found, "Agent should use tool calls for data questions"
