"""
Streamlit UI for the Customer Service Data Analyst Agent (fallback).

Run: uv run streamlit run streamlit_app.py
"""

import sqlite3
import time

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError

from src.agent import build_graph, token_tracker, tool_timer
from src.config import CHECKPOINTS_DB
from src.data import metadata
from src.health import run_startup_checks
from src.ui_helpers import (
    SUGGESTION_CHIPS,
    SessionStats,
    export_conversation_markdown,
    extract_reasoning_steps,
    format_token_line,
    format_token_summary,
    get_final_response,
)

st.set_page_config(page_title="CS Data Analyst", page_icon="📊", layout="wide")

# --- Sidebar: Analyst Dashboard ---

with st.sidebar:
    st.title("📊 Analyst Dashboard")

    st.subheader("Session")
    session_id = st.text_input("Session ID", value="default")
    user_id = st.text_input("User ID", value="default")

    st.subheader("Dataset")
    st.metric("Records", f"{metadata.row_count:,}")
    col1, col2 = st.columns(2)
    col1.metric("Categories", metadata.num_categories)
    col2.metric("Intents", metadata.num_intents)
    with st.expander("Category → Intent Map"):
        for cat, intents in metadata.category_intent_map.items():
            st.markdown(f"**{cat}**: {', '.join(intents)}")

    st.subheader("Token Budget")
    token_data = format_token_summary(token_tracker)
    st.metric("Session Tokens", f"{token_data['total_tokens']:,}")
    st.caption(f"~${token_data['estimated_cost']:.4f} estimated cost")

    if "stats" in st.session_state:
        stats: SessionStats = st.session_state.stats
        st.subheader("Session Stats")
        col1, col2 = st.columns(2)
        col1.metric("Queries", stats.query_count)
        col2.metric("Tool Calls", stats.tool_call_count)
        if stats.query_count > 0:
            st.metric("Avg Response", f"{stats.avg_response_time:.1f}s")

    st.subheader("System Health")
    if st.button("Run Health Check"):
        report = run_startup_checks()
        if report.has_failures:
            st.error(report.summary())
        elif report.has_warnings:
            st.warning(report.summary())
        else:
            st.success("All systems operational ✓")

    st.subheader("Export")
    if st.session_state.get("chat_history"):
        md = export_conversation_markdown(
            [m["langchain_msg"] for m in st.session_state.chat_history if "langchain_msg" in m]
        )
        st.download_button(
            "📥 Download Conversation", md, "conversation.md", "text/markdown"
        )

# --- Main Chat Area ---

st.title("🔍 Customer Service Data Analyst")
st.caption(
    f"Analyzing {metadata.row_count:,} records · "
    f"{metadata.num_categories} categories · {metadata.num_intents} intents"
)

# Initialize state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.stats = SessionStats()

if "graph" not in st.session_state:
    conn = sqlite3.connect(str(CHECKPOINTS_DB), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    st.session_state.graph = build_graph(checkpointer=checkpointer)

# Display chat history
for msg_data in st.session_state.chat_history:
    with st.chat_message(msg_data["role"]):
        st.markdown(msg_data["content"])
        if "reasoning" in msg_data:
            with st.expander("🔧 Reasoning Steps"):
                for step in msg_data["reasoning"]:
                    if step["type"] == "tool_call":
                        st.code(f"→ {step['name']}({step['args']})", language="")
                    elif step["type"] == "tool_result":
                        st.text(step["content"])
        if "token_line" in msg_data:
            st.caption(msg_data["token_line"])

# Suggestion chips — only when chat is empty
if not st.session_state.chat_history:
    st.markdown("**Try one of these queries:**")
    cols = st.columns(3)
    for i, chip in enumerate(SUGGESTION_CHIPS[:6]):
        if cols[i % 3].button(chip["label"], key=f"chip_{i}"):
            st.session_state.pending_query = chip["query"]
            st.rerun()

# Handle chip click
pending = st.session_state.pop("pending_query", None)

# Chat input
user_input = st.chat_input("Ask about the customer service dataset...") or pending

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 12,
    }

    graph = st.session_state.graph
    current_state = graph.get_state(config)
    existing_count = (
        len(current_state.values.get("messages", []))
        if current_state.values
        else 0
    )

    token_tracker.reset_query()
    tool_timer.reset_query()
    start_time = time.time()

    with st.chat_message("assistant"):
        with st.status("Analyzing...", expanded=True) as status:
            try:
                result = graph.invoke(
                    {
                        "messages": [HumanMessage(content=user_input)],
                        "user_id": user_id,
                    },
                    config=config,
                )
            except GraphRecursionError:
                st.error(
                    "Could not complete the analysis within allowed steps. "
                    "Try rephrasing your question."
                )
                st.stop()

            response_time = time.time() - start_time
            steps = extract_reasoning_steps(
                result["messages"], start_index=existing_count + 1
            )

            for step in steps:
                if step["type"] == "tool_call":
                    st.write(f"🔧 **{step['name']}**({step['args']})")
                elif step["type"] == "tool_result":
                    st.text(step["content"][:200])

            status.update(
                label=f"Done in {response_time:.1f}s",
                state="complete",
                expanded=False,
            )

        final = get_final_response(result["messages"])
        st.markdown(final)

        token_line = format_token_line(token_tracker)
        st.caption(token_line)

    st.session_state.stats.record_query(steps, response_time)
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": final,
            "reasoning": steps,
            "token_line": f"{token_line} · ⏱ {response_time:.1f}s",
            "langchain_msg": result["messages"][-1],
        }
    )

    st.rerun()
