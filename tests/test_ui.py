"""
Tests for shared UI helpers (src/ui_helpers.py) and session store (src/session_store.py).
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.ui_helpers import (
    DEFAULT_SUGGESTIONS,
    export_conversation_markdown,
    extract_reasoning_steps,
    format_per_query_tokens,
    format_response_footer,
    get_final_response,
    get_suggestions_after_response,
    suggest_tags,
)


# --- Suggestion chips ---


def test_suggestion_chips_defined():
    assert len(DEFAULT_SUGGESTIONS) >= 6
    assert all("label" in c and "query" in c for c in DEFAULT_SUGGESTIONS)


def test_contextual_suggestions_oos():
    suggestions = get_suggestions_after_response("out_of_scope", "I can't help")
    assert len(suggestions) >= 2


def test_contextual_suggestions_refund():
    suggestions = get_suggestions_after_response(
        "structured", "There are 2,992 REFUND entries."
    )
    assert any("refund" in s["query"].lower() for s in suggestions)


# --- Reasoning steps ---


def test_extract_reasoning_steps():
    msgs = [
        HumanMessage(content="How many refund requests?"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "count_rows", "args": {"category": "REFUND"}, "id": "1"}
            ],
        ),
        ToolMessage(content="Found 2,992 rows", name="count_rows", tool_call_id="1"),
        AIMessage(content="There are 2,992 refund requests."),
    ]
    steps = extract_reasoning_steps(msgs, start_index=1)
    assert len(steps) == 2
    assert steps[0]["type"] == "tool_call"
    assert steps[0]["name"] == "count_rows"
    assert steps[1]["type"] == "tool_result"
    assert "2,992" in steps[1]["content"]


def test_extract_reasoning_truncates_long_results():
    long_content = "x" * 500
    msgs = [
        ToolMessage(content=long_content, name="get_examples", tool_call_id="1"),
    ]
    steps = extract_reasoning_steps(msgs)
    assert steps[0]["content"].endswith("...")
    assert len(steps[0]["content"]) == 303


def test_get_final_response():
    msgs = [
        AIMessage(
            content="",
            tool_calls=[{"name": "count_rows", "args": {}, "id": "1"}],
        ),
        ToolMessage(content="result", name="count_rows", tool_call_id="1"),
        AIMessage(content="The answer is 42."),
    ]
    assert get_final_response(msgs) == "The answer is 42."


def test_get_final_response_no_response():
    msgs = [HumanMessage(content="hello")]
    assert get_final_response(msgs) == "No response generated."


# --- Token formatting ---


def test_format_per_query_tokens():
    result = format_per_query_tokens({"prompt": 1000, "completion": 200, "total": 1200})
    assert "1,200" in result
    assert "$" in result


def test_format_response_footer():
    footer = format_response_footer(
        "structured", {"prompt": 1000, "completion": 200, "total": 1200}, 2.3, 1
    )
    assert "📊" in footer
    assert "1,200" in footer
    assert "2.3s" in footer
    assert "1 calls" in footer


def test_format_response_footer_oos():
    footer = format_response_footer(
        "out_of_scope", {"prompt": 100, "completion": 50, "total": 150}, 0.5, 0
    )
    assert "🚫" in footer


# --- Auto-tagging ---


def test_auto_tags():
    msgs = [
        HumanMessage(content="Tell me about REFUND"),
        AIMessage(content="The REFUND category has..."),
    ]
    tags = suggest_tags(msgs)
    assert "refund" in tags


def test_auto_tags_multiple():
    msgs = [
        HumanMessage(content="Compare REFUND and SHIPPING categories"),
    ]
    tags = suggest_tags(msgs)
    assert "refund" in tags
    assert "shipping" in tags


def test_auto_tags_empty():
    tags = suggest_tags([HumanMessage(content="hello")])
    assert tags == []


# --- Export ---


def test_export_conversation():
    msgs = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there"),
    ]
    md = export_conversation_markdown(msgs)
    assert "**User**: Hello" in md
    assert "**Agent**: Hi there" in md


def test_export_excludes_tool_calls():
    msgs = [
        HumanMessage(content="Query"),
        AIMessage(
            content="calling tool",
            tool_calls=[{"name": "count_rows", "args": {}, "id": "1"}],
        ),
        ToolMessage(content="result", name="count_rows", tool_call_id="1"),
        AIMessage(content="Final answer"),
    ]
    md = export_conversation_markdown(msgs)
    assert "**User**: Query" in md
    assert "**Agent**: Final answer" in md
    assert "calling tool" not in md


# --- Session Store ---


def test_session_store_crud(tmp_path):
    from src.session_store import SessionStore

    s = SessionStore(path=tmp_path / "test_store.json")
    chat = s.get_or_create_chat("t1", "user1")
    assert chat.thread_id == "t1"
    assert chat.user_id == "user1"
    s.add_tag("t1", "refund")
    assert "refund" in s.chats["t1"].tags
    s.remove_tag("t1", "refund")
    assert "refund" not in s.chats["t1"].tags


def test_session_store_list_filter(tmp_path):
    from src.session_store import SessionStore

    s = SessionStore(path=tmp_path / "test_store.json")
    s.get_or_create_chat("t1")
    s.add_tag("t1", "refund")
    s.get_or_create_chat("t2")
    s.add_tag("t2", "shipping")
    assert len(s.list_chats(tag_filter=["refund"])) == 1
    assert len(s.list_chats(tag_filter=["shipping"])) == 1
    assert len(s.list_chats()) == 2


def test_session_store_search(tmp_path):
    from src.session_store import SessionStore

    s = SessionStore(path=tmp_path / "test_store.json")
    s.get_or_create_chat("t1")
    s.update_chat_title("t1", "How many refund requests?")
    s.get_or_create_chat("t2")
    s.update_chat_title("t2", "Show me shipping examples")
    assert len(s.list_chats(search="refund")) == 1
    assert len(s.list_chats(search="xyz")) == 0


def test_session_store_log_query(tmp_path):
    from src.session_store import QueryTrace, SessionStore

    s = SessionStore(path=tmp_path / "test_store.json")
    s.get_or_create_chat("t1")
    trace = QueryTrace(
        query_index=0,
        timestamp="2026-05-22T09:00:00",
        user_message="How many refunds?",
        query_type="structured",
        tokens={"prompt": 500, "completion": 100, "total": 600},
        total_duration_ms=2000,
    )
    s.log_query("t1", trace)
    assert s.chats["t1"].query_count == 1
    assert s.chats["t1"].total_tokens == 600
    assert s.chats["t1"].message_count == 2


def test_session_store_persistence(tmp_path):
    from src.session_store import SessionStore

    path = tmp_path / "test_store.json"
    s1 = SessionStore(path=path)
    s1.get_or_create_chat("t1")
    s1.add_tag("t1", "test")

    s2 = SessionStore(path=path)
    assert "t1" in s2.chats
    assert "test" in s2.chats["t1"].tags


def test_session_store_get_all_tags(tmp_path):
    from src.session_store import SessionStore

    s = SessionStore(path=tmp_path / "test_store.json")
    s.get_or_create_chat("t1")
    s.add_tag("t1", "beta")
    s.add_tag("t1", "alpha")
    s.get_or_create_chat("t2")
    s.add_tag("t2", "alpha")
    tags = s.get_all_tags()
    assert tags == ["alpha", "beta"]
