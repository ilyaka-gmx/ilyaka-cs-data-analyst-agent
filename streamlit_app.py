"""
Streamlit UI for the Customer Service Data Analyst Agent.

Layout:
  - Left sidebar: conversation list (search, tags, new chat, tag mgmt, settings)
  - Main area: Chat tab + Admin tab
  - Bottom of sidebar: status indicator (health, dataset, tokens, model)

Run: uv run streamlit run streamlit_app.py
"""

import sqlite3
import time
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
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
# Helper: rebuild display from session store (defined before sidebar uses it)
# ---------------------------------------------------------------------------


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
            "footer": (
                f"{q.query_type} · {q.tokens.get('total', 0)} tok "
                f"· {q.total_duration_ms}ms"
            ),
        })
    return display


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
# Sidebar — Conversations, Tags, Settings, Status
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
    active_tags = st.multiselect(
        "Filter by tags",
        all_tags if all_tags else ["(no tags yet)"],
        default=[],
        key="tag_filter",
        disabled=not all_tags,
    )
    if not all_tags:
        active_tags = []

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

    # --- Tag management for active chat ---
    active_thread = st.session_state.active_thread
    active_chat = store.chats.get(active_thread)
    if active_chat:
        with st.expander("🏷 Manage Tags"):
            if active_chat.tags:
                st.caption("Current tags:")
                for tag in list(active_chat.tags):
                    col_tag, col_del = st.columns([4, 1])
                    col_tag.write(f"🏷 {tag}")
                    if col_del.button(
                        "✕", key=f"rmtag_{active_thread}_{tag}"
                    ):
                        store.remove_tag(active_thread, tag)
                        st.rerun()
            else:
                st.caption("No tags yet")

            new_tag = st.text_input(
                "Add tag", key="new_tag_input", placeholder="e.g. refund"
            )
            if st.button("Add", key="add_tag_btn"):
                tag_clean = new_tag.strip().lower()
                if tag_clean:
                    store.add_tag(active_thread, tag_clean)
                    st.rerun()

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

    # --- Status bar (native Streamlit, theme-aware, at sidebar bottom) ---
    st.divider()
    health_data = st.session_state.get(
        "last_health", {"status": "unknown", "age": "checking..."}
    )
    health_dot = {"healthy": "🟢", "warning": "🟡", "error": "🔴"}.get(
        health_data.get("status", ""), "⚪"
    )
    health_status = health_data.get("status", "unknown").title()
    health_age = health_data.get("age", "?")

    with st.expander(f"{health_dot} {health_status} · {health_age}"):
        report = health_data.get("report")
        if report:
            for check in report.checks:
                icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}[check.status]
                st.caption(f"{icon} {check.name}: {check.message}")
        if st.button("🔄 Refresh Health", key="refresh_health"):
            st.session_state.last_health_time = 0
            st.rerun()

    session_tokens = (
        token_tracker.total_prompt_tokens
        + token_tracker.total_completion_tokens
    )
    session_cost = (
        token_tracker.total_prompt_tokens * 0.20
        + token_tracker.total_completion_tokens * 0.60
    ) / 1_000_000
    model_short = AGENT_MODEL.split("/")[-1]

    st.caption(
        f"📊 {metadata.row_count:,} rows · "
        f"{metadata.num_categories} cat · {metadata.num_intents} int"
    )
    st.caption(f"💰 {session_tokens:,} tok · ~${session_cost:.4f}")
    st.caption(f"🤖 {model_short}")


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

    # Suggestion chips (shown on empty chat or after out-of-scope)
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

    # --- Detect pending query (set by chips or previous rerun) ---
    pending = st.session_state.pop("pending_query", None)
    user_input = st.chat_input("Ask about the customer service dataset...")

    if user_input:
        st.session_state.setdefault("chat_display", []).append(
            {"role": "user", "content": user_input}
        )
        st.session_state.pending_process = user_input
        thread_id = st.session_state.active_thread
        store.get_or_create_chat(thread_id, st.session_state.user_id)
        store.update_chat_title(thread_id, user_input)
        st.rerun()

    if pending:
        st.session_state.setdefault("chat_display", []).append(
            {"role": "user", "content": pending}
        )
        st.session_state.pending_process = pending
        thread_id = st.session_state.active_thread
        store.get_or_create_chat(thread_id, st.session_state.user_id)
        store.update_chat_title(thread_id, pending)
        st.rerun()

    # --- Process pending query (on the rerun after user message was added) ---
    query_to_process = st.session_state.pop("pending_process", None)

    if query_to_process:
        thread_id = st.session_state.active_thread
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

                error_occurred = False
                result = None

                try:
                    result = graph.invoke(
                        {
                            "messages": [
                                HumanMessage(content=query_to_process)
                            ],
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
                    error_occurred = True
                except Exception as e:
                    status.update(
                        label="❌ Error", state="error", expanded=True
                    )
                    st.error(str(e))
                    error_occurred = True

                if error_occurred:
                    st.caption(
                        "💡 Tip: refresh the page to clear the error state."
                    )
                    st.stop()

                duration = time.time() - start_time
                steps = extract_reasoning_steps(
                    result["messages"], start_index=existing_count + 1
                )

                for step in steps:
                    if step["type"] == "tool_call":
                        st.write(
                            f"🔧 **{step['name']}**({step['args']})"
                        )
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
            user_message=query_to_process,
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

                # --- Tag management in admin ---
                if chat.tags:
                    tag_cols = st.columns(len(chat.tags) + 1)
                    for j, tag in enumerate(list(chat.tags)):
                        if tag_cols[j].button(
                            f"🏷 {tag} ✕",
                            key=f"adm_rmtag_{chat.thread_id}_{tag}",
                        ):
                            store.remove_tag(chat.thread_id, tag)
                            st.rerun()
                else:
                    st.caption("No tags")

                adm_tag_col1, adm_tag_col2 = st.columns([3, 1])
                adm_new_tag = adm_tag_col1.text_input(
                    "Add tag",
                    key=f"adm_newtag_{chat.thread_id}",
                    label_visibility="collapsed",
                    placeholder="Add tag...",
                )
                if adm_tag_col2.button(
                    "+", key=f"adm_addtag_{chat.thread_id}"
                ):
                    tag_clean = adm_new_tag.strip().lower()
                    if tag_clean:
                        store.add_tag(chat.thread_id, tag_clean)
                        st.rerun()

                st.divider()

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
                        st.caption(
                            f"Response: {q.final_response_preview}"
                        )

                    st.divider()
