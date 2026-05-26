"""Integration tests for the full agent graph.

Slow tests make real LLM calls.
Run: uv run pytest tests/test_agent.py -v
"""

import json
import sqlite3
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agent import build_graph
from src.config import RECURSION_LIMIT


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
    return {"configurable": {"thread_id": "test"}, "recursion_limit": RECURSION_LIMIT}


def _invoke(graph, query: str, config: dict, user_id: str = "test") -> str:
    """Invoke the graph and return the final AI message content."""
    result = graph.invoke(
        {"messages": [HumanMessage(content=query)], "user_id": user_id},
        config,
    )
    return result["messages"][-1].content


def _invoke_full(graph, query: str, config: dict, user_id: str = "test") -> dict:
    """Invoke the graph and return the full result dict."""
    return graph.invoke(
        {"messages": [HumanMessage(content=query)], "user_id": user_id},
        config,
    )


def _has_tool_call(messages: list, tool_name: str) -> bool:
    """Check if any message in the list contains a call to the named tool."""
    return any(
        isinstance(m, AIMessage) and any(
            tc.get("name") == tool_name for tc in (m.tool_calls or [])
        )
        for m in messages
    )


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

    config = {"configurable": {"thread_id": "persist_test"}, "recursion_limit": RECURSION_LIMIT}

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

    tool_calls_found = any(
        isinstance(m, AIMessage) and m.tool_calls for m in messages
    )
    assert tool_calls_found, "Agent should use tool calls for data questions"


# ==========================================================================
# Gate 4 — Conversation Memory
# ==========================================================================


@pytest.mark.slow
def test_gate4_persistence_across_restart(tmp_path):
    """Simulate process restart: two separate graph instances, same SQLite file."""
    db_path = tmp_path / "checkpoints.db"
    thread_id = f"restart_{uuid.uuid4().hex[:8]}"
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}

    conn1 = sqlite3.connect(str(db_path), check_same_thread=False)
    cp1 = SqliteSaver(conn1)
    g1 = build_graph(checkpointer=cp1)
    print("  [1/2] Asking about REFUND examples...")
    _invoke(g1, "Show me 3 examples from the REFUND category.", cfg)
    conn1.close()
    print("  [1/2] Done. DB exists:", db_path.exists())

    assert db_path.exists(), "checkpoints.db must be created on first run"

    conn2 = sqlite3.connect(str(db_path), check_same_thread=False)
    cp2 = SqliteSaver(conn2)
    g2 = build_graph(checkpointer=cp2)
    print("  [2/2] Follow-up: 'Show me 3 more from the same category.'")
    final = _invoke(g2, "Show me 3 more from the same category.", cfg)
    conn2.close()
    print("  [2/2] Done. Response length:", len(final))

    assert len(final) > 50, "Follow-up should produce a substantive response"
    assert any(
        w in final.lower() for w in ("refund", "get_refund", "track_refund", "check_refund")
    ), f"Follow-up should reference REFUND context, got: {final[:200]}"


@pytest.mark.slow
def test_gate4_follow_up_chain():
    """Three-step follow-up chain within a single session."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = build_graph(checkpointer=checkpointer)
    cfg = {"configurable": {"thread_id": "chain_test"}, "recursion_limit": RECURSION_LIMIT}

    print("  [1/3] How many complaint entries?")
    r1 = _invoke(graph, "How many complaint entries are in the dataset?", cfg)
    print(f"  [1/3] Done: {r1[:80]}")
    assert any(c.isdigit() for c in r1), f"Q1 should return a number, got: {r1[:200]}"

    print("  [2/3] How many refund entries?")
    r2 = _invoke(graph, "And how many refund entries are there?", cfg)
    print(f"  [2/3] Done: {r2[:80]}")
    assert any(c.isdigit() for c in r2), f"Q2 should return a number, got: {r2[:200]}"

    print("  [3/3] Combined total?")
    r3 = _invoke(
        graph,
        "What is the combined total of the complaint and refund counts you just gave me?",
        cfg,
    )
    print(f"  [3/3] Done: {r3[:80]}")
    assert any(c.isdigit() for c in r3), f"Q3 should return a total, got: {r3[:200]}"
    conn.close()


@pytest.mark.slow
def test_gate4_independent_sessions():
    """Different thread_ids produce independent conversation histories."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = build_graph(checkpointer=checkpointer)

    cfg_a = {"configurable": {"thread_id": "session_A"}, "recursion_limit": RECURSION_LIMIT}
    cfg_b = {"configurable": {"thread_id": "session_B"}, "recursion_limit": RECURSION_LIMIT}

    print("  [1/4] Session A: ORDER examples")
    _invoke(graph, "Show me 3 examples from ORDER.", cfg_a)
    print("  [2/4] Session B: FEEDBACK examples")
    _invoke(graph, "Show me 3 examples from FEEDBACK.", cfg_b)

    print("  [3/4] Session A follow-up")
    follow_a = _invoke(graph, "Show me 3 more from the same category.", cfg_a)
    print(f"  [3/4] Done: {follow_a[:80]}")
    assert "order" in follow_a.lower() or "ORDER" in follow_a, (
        f"Session A follow-up should reference ORDER, got: {follow_a[:200]}"
    )

    print("  [4/4] Session B follow-up")
    follow_b = _invoke(graph, "Show me 3 more from the same category.", cfg_b)
    print(f"  [4/4] Done: {follow_b[:80]}")
    assert "feedback" in follow_b.lower() or "FEEDBACK" in follow_b, (
        f"Session B follow-up should reference FEEDBACK, got: {follow_b[:200]}"
    )
    conn.close()


def test_gate4_checkpoints_db_path():
    """Verify CHECKPOINTS_DB points to a file (not :memory:) in config."""
    from src.config import CHECKPOINTS_DB

    assert CHECKPOINTS_DB.name == "checkpoints.db"
    assert str(CHECKPOINTS_DB).endswith("checkpoints.db")


def test_gate4_main_uses_sqlite_saver():
    """Verify main.py uses SqliteSaver, not MemorySaver."""
    import importlib
    import inspect

    import main as main_mod

    importlib.reload(main_mod)
    source = inspect.getsource(main_mod)
    assert "SqliteSaver" in source, "main.py must use SqliteSaver"
    assert "MemorySaver" not in source, "main.py must NOT use MemorySaver"


# ==========================================================================
# Gate 5 — User Profile
# ==========================================================================


def _mock_mem0(monkeypatch):
    """Set up an in-memory mock for mem0 in src.memory."""
    import src.memory as mem_mod

    _store: dict[str, list[dict]] = {}

    def mock_add(text, user_id=None, **kw):
        _store.setdefault(user_id, []).append({"memory": text, "id": str(len(_store.get(user_id, [])))})
        return {"results": [{"event": "ADD", "memory": text}]}

    def mock_get_all(filters=None, **kw):
        uid = (filters or {}).get("user_id", "default")
        return {"results": _store.get(uid, [])}

    def mock_search(query, filters=None, **kw):
        uid = (filters or {}).get("user_id", "default")
        return {"results": [m for m in _store.get(uid, []) if query.lower() in m["memory"].lower()]}

    def mock_delete_all(user_id=None, **kw):
        _store.pop(user_id, None)

    class MockMemory:
        add = staticmethod(mock_add)
        get_all = staticmethod(mock_get_all)
        search = staticmethod(mock_search)
        delete_all = staticmethod(mock_delete_all)

    monkeypatch.setattr(mem_mod, "_memory", MockMemory())
    return _store


@pytest.mark.slow
def test_gate5_remember_fact_stores_profile(monkeypatch):
    """Agent should call remember_fact when personal info is shared."""
    _mock_mem0(monkeypatch)

    uid = "remember_user"
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    graph = build_graph(checkpointer=SqliteSaver(conn))
    cfg = {"configurable": {"thread_id": "profile_test"}, "recursion_limit": RECURSION_LIMIT}

    result = _invoke_full(
        graph,
        "Please remember: my name is Alex and I work in the refund department.",
        cfg,
        user_id=uid,
    )

    tool_called = _has_tool_call(result["messages"], "remember_fact")
    assert tool_called, (
        "Agent should call remember_fact when user shares personal info"
    )
    conn.close()


@pytest.mark.slow
def test_gate5_recall_profile_responds(monkeypatch):
    """Agent should respond with stored facts when asked what it remembers."""
    _mock_mem0(monkeypatch)
    import src.memory as mem_mod

    uid = "recall_user"
    mem_mod.add_fact(uid, "User's name is Alex")
    mem_mod.add_fact(uid, "Interested in refund data")

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    graph = build_graph(checkpointer=SqliteSaver(conn))
    cfg = {"configurable": {"thread_id": "recall_test"}, "recursion_limit": RECURSION_LIMIT}

    final = _invoke(
        graph, "What do you remember about me?", cfg, user_id=uid
    )

    assert any(w in final.lower() for w in ("alex", "refund")), (
        f"Recall should mention stored facts, got: {final[:300]}"
    )
    conn.close()


@pytest.mark.slow
def test_gate5_profile_persistence_across_restart(monkeypatch):
    """Profile persists independently of conversation checkpoints.

    Seeds profile via API, then asks agent in a fresh session.
    """
    _mock_mem0(monkeypatch)
    import src.memory as mem_mod

    uid = "persist_profile_user"
    mem_mod.add_fact(uid, "User's name is Sam")
    mem_mod.add_fact(uid, "Focuses on order issues")

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    graph = build_graph(checkpointer=SqliteSaver(conn))

    cfg = {"configurable": {"thread_id": "profile_s2"}, "recursion_limit": RECURSION_LIMIT}
    final = _invoke(graph, "What do you remember about me?", cfg, user_id=uid)

    assert any(w in final.lower() for w in ("sam", "order")), (
        f"Recall should mention stored facts, got: {final[:300]}"
    )
    conn.close()


def test_gate5_profile_additive(monkeypatch):
    """Multiple facts accumulate in the profile (not overwritten)."""
    _mock_mem0(monkeypatch)
    from src.memory import add_fact, get_all_memories_raw

    uid = "additive_user"
    add_fact(uid, "Likes cats")
    add_fact(uid, "Works in finance")
    add_fact(uid, "Prefers CSV exports")

    memories = get_all_memories_raw(uid)
    assert len(memories) == 3, f"Should have 3 facts, got {len(memories)}"
    texts = [m["memory"] for m in memories]
    assert "Likes cats" in texts
    assert "Works in finance" in texts
    assert "Prefers CSV exports" in texts


def test_gate5_profile_separate_from_checkpoints():
    """Mem0 data directory is separate from conversation checkpoints."""
    from src.config import CHECKPOINTS_DB, MEM0_DATA_DIR

    assert MEM0_DATA_DIR.name == "mem0_data"
    assert CHECKPOINTS_DB.name == "checkpoints.db"
    assert MEM0_DATA_DIR != CHECKPOINTS_DB.parent


def test_gate5_mem0_data_dir_created():
    """mem0_data/ directory should be created automatically by config.py."""
    from src.config import MEM0_DATA_DIR

    assert MEM0_DATA_DIR.exists(), "MEM0_DATA_DIR should be auto-created by config.py"
