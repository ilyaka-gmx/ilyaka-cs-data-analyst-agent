"""
Streamlit UI for the Customer Service Data Analyst Agent.

Layout:
  - Left sidebar: conversation list (search, tags, new chat, settings)
  - Main area: Chat tab + Admin tab
  - Bottom status bar: health, dataset, tokens, model

Run: uv run streamlit run streamlit_app.py
"""

import sqlite3
import time
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError

from src.agent import build_graph, token_tracker, tool_timer
from src.config import AGENT_MODEL, CHECKPOINTS_DB, HEALTH_CHECK_INTERVAL_SECONDS
from src.data import metadata
from src.health import run_startup_checks
from src.session_store import QueryTrace, store
from src.ui_helpers import (
    DEFAULT_SUGGESTIONS,
    export_conversation_markdown,
    extract_reasoning_steps,
    format_response_footer,
    get_final_response,
    get_suggestions_after_response,
    suggest_tags,
)

st.set_page_config(page_title="CS Data Analyst", page_icon="🔍", layout="wide")

# ---------------------------------------------------------------------------
# Graph initialization (once per Streamlit worker)
# ---------------------------------------------------------------------------

if "graph" not in st.session_state:
    conn = sqlite3.connect(str(CHECKPOINTS_DB), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    st.session_state.graph = build_graph(checkpointer=checkpointer)

if "active_thread" not in st.session_state:
    st.session_state.active_thread = str(uuid.uuid4())[:8]
if "chat_display" not in st.session_state:
    st.session_state.chat_display = []
if "user_id" not in st.session_state:
    st.session_state.user_id = "default"


# ---------------------------------------------------------------------------
# Health check (interval-based auto-refresh)
# ---------------------------------------------------------------------------


def _check_health_if_due():
    last_check = st.session_state.get("last_health_time", 0)
    now = time.time()
    if now - last_check > HEALTH_CHECK_INTERVAL_SECONDS:
        report = run_startup_checks()
        status = (
            "error"
            if report.has_failures
            else ("warning" if report.has_warnings else "healthy")
        )
        st.session_state.last_health = {
            "status": status,
            "time": time.strftime("%H:%M:%S"),
            "age": "just now",
            "report": report,
        }
        st.session_state.last_health_time = now
    else:
        elapsed = int(now - last_check)
        age = f"{elapsed}s ago" if elapsed < 60 else f"{elapsed // 60}m ago"
        if "last_health" in st.session_state:
            st.session_state.last_health["age"] = age


_check_health_if_due()

# ---------------------------------------------------------------------------
# Sidebar — Conversations
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🔍 CS Data Analyst")

    if st.button("+ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())[:8]
        st.session_state.active_thread = new_id
        st.session_state.chat_display = []
        st.rerun()

    search = st.text_input("🔍 Search chats", key="chat_search")

    all_tags = store.get_all_tags()
    active_tags = (
        st.multiselect("Filter by tags", all_tags, key="tag_filter")
        if all_tags
        else []
    )

    chats = store.list_chats(
        tag_filter=active_tags or None, search=search or None
    )
    for chat in chats:
        is_active = st.session_state.active_thread == chat.thread_id
        label = f"{'▸ ' if is_active else ''}{chat.title}"
        if st.button(
            label, key=f"chat_{chat.thread_id}", use_container_width=True
        ):
            st.session_state.active_thread = chat.thread_id
            st.session_state.chat_display = _load_display_from_store(
                chat.thread_id
            )
            st.rerun()

        if chat.tags:
            st.caption(" ".join(f"🏷 {t}" for t in chat.tags))
        st.caption(f"{chat.updated_at[:16]} · {chat.message_count} msg")

    with st.expander("⚙️ Settings"):
        user_id = st.text_input(
            "User ID",
            value=st.session_state.get("user_id", "default"),
        )
        st.session_state.user_id = user_id

        if st.session_state.get("chat_display"):
            md = export_conversation_markdown(
                [
                    m["langchain_msg"]
                    for m in st.session_state.chat_display
                    if "langchain_msg" in m
                ]
            )
            st.download_button(
                "📥 Export Conversation",
                md,
                "conversation.md",
                "text/markdown",
            )


def _load_display_from_store(thread_id: str) -> list[dict]:
    """Rebuild chat_display from stored query traces."""
    chat = store.chats.get(thread_id)
    if not chat:
        return []
    display: list[dict] = []
    for q in chat.queries:
        display.append({"role": "user", "content": q.user_message})
        display.append({
            "role": "assistant",
            "content": q.final_response_preview,
            "reasoning": q.tool_calls,
            "footer": f"{q.query_type} · {q.tokens.get('total', 0)} tok · {q.total_duration_ms}ms",
        })
    return display


# ---------------------------------------------------------------------------
# Main Area — Tabs
# ---------------------------------------------------------------------------

chat_tab, admin_tab = st.tabs(["💬 Chat", "🔧 Admin"])

# ---------------------------------------------------------------------------
# Chat Tab
# ---------------------------------------------------------------------------

with chat_tab:
    for msg_data in st.session_state.get("chat_display", []):
        with st.chat_message(msg_data["role"]):
            st.markdown(msg_data["content"])
            if msg_data.get("reasoning"):
                with st.expander(
                    f"🔧 Reasoning ({len(msg_data['reasoning'])} steps)"
                ):
                    for step in msg_data["reasoning"]:
                        if step.get("type") == "tool_call":
                            st.code(
                                f"→ {step['name']}({step.get('args', {})})",
                                language="",
                            )
                        elif step.get("type") == "tool_result":
                            st.text(step.get("content", "")[:200])
                        elif step.get("name"):
                            st.code(
                                f"→ {step['name']}({step.get('args', {})})",
                                language="",
                            )
            if msg_data.get("footer"):
                st.caption(msg_data["footer"])

    # Suggestion chips
    show_suggestions = not st.session_state.get(
        "chat_display"
    ) or st.session_state.get("show_recovery_chips")
    if show_suggestions:
        suggestions = st.session_state.get(
            "current_suggestions", DEFAULT_SUGGESTIONS
        )
        cols = st.columns(3)
        for i, chip in enumerate(suggestions[:6]):
            if cols[i % 3].button(chip["label"], key=f"sug_{i}"):
                st.session_state.pending_query = chip["query"]
                st.session_state.show_recovery_chips = False
                st.rerun()

    # Chat input
    pending = st.session_state.pop("pending_query", None)
    user_input = (
        st.chat_input("Ask about the customer service dataset...") or pending
    )

    if user_input:
        thread_id = st.session_state.active_thread
        store.get_or_create_chat(thread_id, st.session_state.user_id)
        store.update_chat_title(thread_id, user_input)

        st.session_state.setdefault("chat_display", []).append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        graph = st.session_state.graph
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 12,
        }

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
            with st.status("🤔 Thinking...", expanded=True) as status:
                st.write("→ Routing query...")

                try:
                    result = graph.invoke(
                        {
                            "messages": [HumanMessage(content=user_input)],
                            "user_id": st.session_state.user_id,
                        },
                        config=config,
                    )
                except GraphRecursionError:
                    status.update(
                        label="⚠️ Max steps reached",
                        state="error",
                        expanded=True,
                    )
                    st.warning(
                        "Could not complete the analysis within allowed "
                        "steps. Try rephrasing your question."
                    )
                    st.stop()
                except Exception as e:
                    status.update(
                        label="❌ Error", state="error", expanded=True
                    )
                    st.error(str(e))
                    st.stop()

                duration = time.time() - start_time
                steps = extract_reasoning_steps(
                    result["messages"], start_index=existing_count + 1
                )

                for step in steps:
                    if step["type"] == "tool_call":
                        st.write(f"🔧 **{step['name']}**({step['args']})")
                    elif step["type"] == "tool_result":
                        st.text(step["content"][:150])

                status.update(
                    label=f"✅ Done in {duration:.1f}s",
                    state="complete",
                    expanded=False,
                )

            final = get_final_response(result["messages"])
            st.markdown(final)

            query_type = result.get("query_type", "structured")
            query_tokens = {
                "prompt": token_tracker.query_prompt_tokens,
                "completion": token_tracker.query_completion_tokens,
                "total": (
                    token_tracker.query_prompt_tokens
                    + token_tracker.query_completion_tokens
                ),
            }
            tool_count = sum(1 for s in steps if s["type"] == "tool_call")
            footer = format_response_footer(
                query_type, query_tokens, duration, tool_count
            )
            st.caption(footer)

            suggestions = get_suggestions_after_response(query_type, final)
            if query_type == "out_of_scope":
                st.markdown("**Try instead:**")
                st.session_state.show_recovery_chips = True
            else:
                st.session_state.show_recovery_chips = False
            st.session_state.current_suggestions = suggestions

            auto_tags = suggest_tags(result["messages"])
            for tag in auto_tags:
                store.add_tag(thread_id, tag)

        trace = QueryTrace(
            query_index=len(store.get_or_create_chat(thread_id).queries),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            user_message=user_input,
            query_type=query_type,
            steps=[],
            tool_calls=[s for s in steps if s["type"] == "tool_call"],
            tokens=query_tokens,
            total_duration_ms=int(duration * 1000),
            hit_fallback="wasn't able to complete" in final.lower(),
            final_response_preview=final[:200],
        )
        store.log_query(thread_id, trace)

        st.session_state.chat_display.append({
            "role": "assistant",
            "content": final,
            "reasoning": steps,
            "footer": footer,
            "langchain_msg": result["messages"][-1],
        })

        st.rerun()

# ---------------------------------------------------------------------------
# Admin Tab
# ---------------------------------------------------------------------------

with admin_tab:
    st.subheader("Session Traces")

    all_chats = store.list_chats()

    if not all_chats:
        st.info(
            "No sessions recorded yet. Start a conversation in the Chat tab."
        )
    else:
        for chat in all_chats:
            has_fallback = any(q.hit_fallback for q in chat.queries)
            status_icon = "⚠️" if has_fallback else "✅"

            with st.expander(
                f"{status_icon} {chat.title} — {chat.updated_at[:16]} · "
                f"{chat.query_count} queries · {chat.total_tokens:,} tok"
            ):
                st.caption(
                    f"Session: `{chat.thread_id}` · User: {chat.user_id}"
                )
                if chat.tags:
                    st.caption(
                        "Tags: " + " ".join(f"🏷 {t}" for t in chat.tags)
                    )

                for q in chat.queries:
                    fallback_icon = "⚠️" if q.hit_fallback else "✅"
                    msg_preview = (
                        q.user_message[:60] + "..."
                        if len(q.user_message) > 60
                        else q.user_message
                    )
                    st.markdown(
                        f"**{fallback_icon} Q{q.query_index + 1}**: "
                        f'"{msg_preview}" '
                        f"({q.query_type}) — {q.total_duration_ms}ms · "
                        f"{q.tokens.get('total', 0)} tok"
                    )

                    if q.tool_calls:
                        for tc in q.tool_calls:
                            st.code(
                                f"  → {tc.get('name', '?')}"
                                f"({tc.get('args', {})})",
                                language="",
                            )

                    if q.final_response_preview:
                        st.caption(f"Response: {q.final_response_preview}")

                    st.divider()

# ---------------------------------------------------------------------------
# Bottom Status Bar (fixed CSS footer)
# ---------------------------------------------------------------------------

health_data = st.session_state.get(
    "last_health", {"status": "unknown", "age": "checking..."}
)
dataset_info = (
    f"📊 {metadata.row_count:,} · "
    f"{metadata.num_categories} cat · {metadata.num_intents} int"
)
session_tokens = (
    token_tracker.total_prompt_tokens + token_tracker.total_completion_tokens
)
session_cost = (
    token_tracker.total_prompt_tokens * 0.20
    + token_tracker.total_completion_tokens * 0.60
) / 1_000_000
model_short = AGENT_MODEL.split("/")[-1]

health_dot = {"healthy": "🟢", "warning": "🟡", "error": "🔴"}.get(
    health_data.get("status", ""), "⚪"
)

status_bar_html = f"""
<div style="
    position: fixed; bottom: 0; left: 0; right: 0; z-index: 999;
    background: var(--background-color, #0e1117);
    border-top: 1px solid #333;
    padding: 6px 20px;
    font-size: 12px;
    color: #888;
    display: flex;
    gap: 16px;
    align-items: center;
">
    <span>{health_dot} {health_data.get('status', 'unknown').title()} · {health_data.get('age', '?')}</span>
    <span>│</span>
    <span>{dataset_info}</span>
    <span>│</span>
    <span>💰 {session_tokens:,} tok · ~${session_cost:.4f}</span>
    <span>│</span>
    <span>🤖 {model_short}</span>
</div>
"""
st.markdown(status_bar_html, unsafe_allow_html=True)
st.markdown('<div style="height: 40px;"></div>', unsafe_allow_html=True)
