"""
Tests for shared UI helpers (src/ui_helpers.py).
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.ui_helpers import (
    SUGGESTION_CHIPS,
    SessionStats,
    export_conversation_markdown,
    extract_reasoning_steps,
    format_token_summary,
    get_final_response,
)


def test_suggestion_chips_defined():
    assert len(SUGGESTION_CHIPS) >= 6
    assert all("label" in c and "query" in c for c in SUGGESTION_CHIPS)


def test_extract_reasoning_steps():
    msgs = [
        HumanMessage(content="How many refund requests?"),
        AIMessage(
            content="",
            tool_calls=[{"name": "count_rows", "args": {"category": "REFUND"}, "id": "1"}],
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
    assert len(steps[0]["content"]) == 303  # 300 + "..."


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


def test_session_stats():
    stats = SessionStats()
    steps = [
        {"type": "tool_call", "name": "count_rows"},
        {"type": "tool_result", "content": "result"},
    ]
    stats.record_query(steps, 2.0)
    assert stats.query_count == 1
    assert stats.tool_call_count == 1
    assert stats.avg_response_time == 2.0

    stats.record_query(steps, 4.0)
    assert stats.query_count == 2
    assert stats.tool_call_count == 2
    assert stats.avg_response_time == 3.0


def test_session_stats_no_queries():
    stats = SessionStats()
    assert stats.avg_response_time == 0.0


def test_format_token_summary():
    class FakeTracker:
        total_prompt_tokens = 1000
        total_completion_tokens = 500

    result = format_token_summary(FakeTracker())
    assert result["prompt_tokens"] == 1000
    assert result["completion_tokens"] == 500
    assert result["total_tokens"] == 1500
    assert result["estimated_cost"] > 0
