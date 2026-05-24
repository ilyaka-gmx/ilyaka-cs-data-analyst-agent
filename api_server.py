"""
FastAPI backend for the custom HTML frontend.

Serves the SPA from frontend/ and exposes REST+SSE endpoints that wrap
the same agent graph used by the CLI and Streamlit.

Run: uv run python api_server.py
"""

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agent import build_graph, token_tracker, tool_timer
from src.config import AGENT_MODEL, CHECKPOINTS_DB, PROFILES_DIR
from src.data import metadata
from src.health import run_startup_checks
from src.memory import load_profile
from src.recommender import get_recommendations
from src.session_store import QueryTrace, store
from src.ui_helpers import suggest_tags

log = logging.getLogger(__name__)

app = FastAPI(title="CS Data Analyst API")

_conn = sqlite3.connect(str(CHECKPOINTS_DB), check_same_thread=False)
_checkpointer = SqliteSaver(_conn)
_graph = build_graph(checkpointer=_checkpointer)
_graph_lock = threading.Lock()


def _get_state_safe(config: dict, retries: int = 3, delay: float = 1.0):
    """get_state with retry on SQLite contention."""
    for attempt in range(retries):
        try:
            with _graph_lock:
                return _graph.get_state(config)
        except Exception as e:
            if "Already borrowed" in str(e) and attempt < retries - 1:
                time.sleep(delay)
                continue
            raise


# ---------------------------------------------------------------------------
# Cached health check (runs in background so /api/health responds instantly)
# ---------------------------------------------------------------------------

_cached_health: dict | None = None
_health_lock = threading.Lock()


def _run_health_check():
    global _cached_health
    try:
        report = run_startup_checks()
        result = {
            "status": (
                "error"
                if report.has_failures
                else ("warning" if report.has_warnings else "healthy")
            ),
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "message": c.message,
                    "duration_ms": c.duration_ms,
                }
                for c in report.checks
            ],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    except Exception as e:
        result = {
            "status": "error",
            "checks": [
                {"name": "Health", "status": "fail", "message": str(e), "duration_ms": 0}
            ],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    with _health_lock:
        _cached_health = result


threading.Thread(target=_run_health_check, daemon=True).start()


# ---------------------------------------------------------------------------
# Startup: sync SqliteSaver sessions into session store
# ---------------------------------------------------------------------------


def _sync_sessions_from_checkpointer():
    """Discover sessions in checkpoints.db not yet in session_store.json."""
    try:
        cursor = _conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints"
        )
        db_threads = {row[0] for row in cursor.fetchall()}
    except Exception:
        return

    for tid in db_threads:
        if tid in store.chats:
            continue
        try:
            config = {"configurable": {"thread_id": tid}}
            state = _graph.get_state(config)
            if not state or not state.values:
                continue
            messages = state.values.get("messages", [])
            user_id = state.values.get("user_id", "default")
            first_msg = next(
                (m.content for m in messages if isinstance(m, HumanMessage)),
                "CLI session",
            )
            store.get_or_create_chat(tid, user_id)
            store.update_chat_title(tid, first_msg)
            store.add_tag(tid, "cli")
        except Exception:
            continue


_sync_sessions_from_checkpointer()


# ---------------------------------------------------------------------------
# API: Chat (SSE streaming)
# ---------------------------------------------------------------------------


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    query = body.get("query", "")
    session_id = body.get("session_id") or str(uuid.uuid4())[:8]
    user_id = body.get("user_id") or "default"
    use_past_sessions = body.get("use_past_sessions", False)

    store.get_or_create_chat(session_id, user_id)
    store.update_chat_title(session_id, query)

    def event_stream():
        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": 12,
        }

        token_tracker.reset_query()
        tool_timer.reset_query()
        start_time = time.time()

        reasoning_steps = []
        final_answer = ""
        query_type = "structured"
        sse_events: list[str] = []

        with _graph_lock:
            try:
                for event in _graph.stream(
                    {
                        "messages": [HumanMessage(content=query)],
                        "user_id": user_id,
                        "use_past_sessions": use_past_sessions,
                        "thread_id": session_id,
                    },
                    config=config,
                    stream_mode="updates",
                ):
                    for node_name, node_output in event.items():
                        if node_name == "__end__":
                            continue

                        if "query_type" in node_output:
                            query_type = node_output["query_type"]
                            step = {"type": "route", "query_type": query_type}
                            reasoning_steps.append(step)
                            sse_events.append(
                                f"event: route\ndata: {json.dumps(step)}\n\n"
                            )

                        for msg in node_output.get("messages", []):
                            if isinstance(msg, AIMessage):
                                if msg.tool_calls:
                                    for tc in msg.tool_calls:
                                        step = {
                                            "type": "tool_call",
                                            "name": tc["name"],
                                            "args": tc.get("args", {}),
                                        }
                                        reasoning_steps.append(step)
                                        sse_events.append(
                                            f"event: tool_call\ndata: {json.dumps(step)}\n\n"
                                        )
                                elif msg.content:
                                    final_answer = msg.content

                            elif isinstance(msg, ToolMessage):
                                content = (
                                    msg.content[:300] + "..."
                                    if len(msg.content) > 300
                                    else msg.content
                                )
                                step = {
                                    "type": "tool_result",
                                    "name": getattr(msg, "name", "tool"),
                                    "content": content,
                                }
                                reasoning_steps.append(step)
                                sse_events.append(
                                    f"event: tool_result\ndata: {json.dumps(step)}\n\n"
                                )

            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
                return

        for ev in sse_events:
            yield ev

        duration = time.time() - start_time
        query_tokens = {
            "prompt": token_tracker.query_prompt_tokens,
            "completion": token_tracker.query_completion_tokens,
            "total": (
                token_tracker.query_prompt_tokens
                + token_tracker.query_completion_tokens
            ),
        }

        auto_tags = []
        context_msgs = 0
        try:
            state = _get_state_safe(config)
            if state and state.values:
                auto_tags = suggest_tags(
                    state.values.get("messages", [])
                )
                for tag in auto_tags:
                    store.add_tag(session_id, tag)
                context_msgs = len(
                    state.values.get("messages", [])
                )
        except Exception:
            pass

        tool_count = sum(
            1 for s in reasoning_steps if s.get("type") == "tool_call"
        )

        semantic_ops = [
            s["name"]
            for s in reasoning_steps
            if s.get("type") == "tool_call"
            and s.get("name") in ("remember_fact", "recall_profile")
        ]
        profile = load_profile(user_id)

        memory_ops = {
            "semantic_tools": semantic_ops,
            "has_profile": bool(profile.facts),
            "profile_fact_count": len(profile.facts),
            "context_messages": context_msgs,
        }

        trace = QueryTrace(
            query_index=len(store.get_or_create_chat(session_id).queries),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            user_message=query,
            query_type=query_type,
            steps=[],
            tool_calls=[
                s for s in reasoning_steps if s.get("type") == "tool_call"
            ],
            tokens=query_tokens,
            total_duration_ms=int(duration * 1000),
            hit_fallback="wasn't able to complete" in final_answer.lower(),
            final_response_preview=final_answer[:200],
            memory_ops=memory_ops,
        )
        store.log_query(session_id, trace)

        answer_data = {
            "text": final_answer,
            "tokens": query_tokens,
            "duration_ms": int(duration * 1000),
            "query_type": query_type,
            "tool_count": tool_count,
            "auto_tags": auto_tags,
            "memory_ops": memory_ops,
        }
        yield f"event: answer\ndata: {json.dumps(answer_data)}\n\n"

    return StreamingResponse(
        event_stream(), media_type="text/event-stream"
    )


# ---------------------------------------------------------------------------
# API: Health (cached, instant response)
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health():
    with _health_lock:
        if _cached_health:
            return _cached_health
    return {
        "status": "loading",
        "checks": [],
        "message": "Health check running...",
    }


@app.post("/api/health/refresh")
def health_refresh():
    threading.Thread(target=_run_health_check, daemon=True).start()
    return {"ok": True}


# ---------------------------------------------------------------------------
# API: Chats (list, get, tags)
# ---------------------------------------------------------------------------


@app.get("/api/chats")
def list_chats(
    user_id: str = Query(default="default"),
    tag: str | None = Query(default=None),
    search: str | None = Query(default=None),
    all_users: bool = Query(default=False),
):
    chats = store.list_chats(
        tag_filter=[tag] if tag else None,
        search=search,
    )
    if not all_users:
        chats = [c for c in chats if c.user_id == user_id]
    return [
        {
            "thread_id": c.thread_id,
            "title": c.title,
            "tags": c.tags,
            "updated_at": c.updated_at,
            "message_count": c.message_count,
            "total_tokens": c.total_tokens,
            "query_count": c.query_count,
            "user_id": c.user_id,
        }
        for c in chats
    ]


@app.get("/api/chats/{thread_id}")
def get_chat(thread_id: str):
    chat = store.chats.get(thread_id)
    if not chat:
        return JSONResponse({"error": "Not found"}, status_code=404)

    messages = []
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = _get_state_safe(config)
        if state and state.values:
            for msg in state.values.get("messages", []):
                if isinstance(msg, HumanMessage):
                    messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    messages.append(
                        {"role": "assistant", "content": msg.content}
                    )
    except Exception:
        pass

    return {
        "thread_id": chat.thread_id,
        "title": chat.title,
        "tags": chat.tags,
        "created_at": chat.created_at,
        "updated_at": chat.updated_at,
        "message_count": chat.message_count,
        "total_tokens": chat.total_tokens,
        "query_count": chat.query_count,
        "user_id": chat.user_id,
        "messages": messages,
        "queries": [asdict(q) for q in chat.queries],
    }


@app.post("/api/chats/{thread_id}/tags")
async def add_tag(thread_id: str, request: Request):
    body = await request.json()
    tag = body.get("tag", "").strip().lower()
    if not tag:
        return JSONResponse({"error": "Empty tag"}, status_code=400)
    store.get_or_create_chat(thread_id)
    store.add_tag(thread_id, tag)
    return {"ok": True, "tags": store.chats[thread_id].tags}


@app.delete("/api/chats/{thread_id}/tags/{tag}")
def remove_tag(thread_id: str, tag: str):
    store.remove_tag(thread_id, tag)
    chat = store.chats.get(thread_id)
    return {"ok": True, "tags": chat.tags if chat else []}


@app.delete("/api/chats/{thread_id}")
def delete_chat(thread_id: str):
    found = store.delete_chat(thread_id)
    try:
        with _graph_lock:
            _conn.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,)
            )
            _conn.execute(
                "DELETE FROM writes WHERE thread_id = ?", (thread_id,)
            )
            _conn.commit()
    except Exception:
        pass
    return {"ok": found}


@app.delete("/api/chats")
def delete_all_chats(user_id: str | None = Query(default=None)):
    if user_id:
        deleted_ids = store.delete_user_chats(user_id)
        try:
            with _graph_lock:
                for tid in deleted_ids:
                    _conn.execute(
                        "DELETE FROM checkpoints WHERE thread_id = ?", (tid,)
                    )
                    _conn.execute(
                        "DELETE FROM writes WHERE thread_id = ?", (tid,)
                    )
                _conn.commit()
        except Exception:
            pass
        return {"ok": True, "deleted": len(deleted_ids)}

    store.delete_all_chats()
    try:
        with _graph_lock:
            _conn.execute("DELETE FROM checkpoints")
            _conn.execute("DELETE FROM writes")
            _conn.commit()
    except Exception:
        pass
    return {"ok": True}


# ---------------------------------------------------------------------------
# API: Admin stats (system overview for admin sidebar)
# ---------------------------------------------------------------------------


@app.get("/api/admin/stats")
def admin_stats():
    user_ids: set[str] = set()
    sessions_per_user: dict[str, int] = {}
    for chat in store.chats.values():
        uid = chat.user_id
        user_ids.add(uid)
        sessions_per_user[uid] = sessions_per_user.get(uid, 0) + 1

    for p in PROFILES_DIR.glob("*.json"):
        user_ids.add(p.stem)

    profile_count = sum(1 for _ in PROFILES_DIR.glob("*.json"))
    total_facts = 0
    for p in PROFILES_DIR.glob("*.json"):
        profile = load_profile(p.stem)
        total_facts += len(profile.facts)

    total_queries = sum(c.query_count for c in store.chats.values())

    return {
        "total_users": len(user_ids),
        "users": sorted(user_ids),
        "total_sessions": len(store.chats),
        "sessions_per_user": sessions_per_user,
        "total_profiles": profile_count,
        "total_facts": total_facts,
        "total_queries": total_queries,
    }


# ---------------------------------------------------------------------------
# API: Tags (all unique tags for filter UI)
# ---------------------------------------------------------------------------


@app.get("/api/tags")
def all_tags(user_id: str | None = Query(default=None)):
    if user_id:
        return {"tags": store.get_tags_for_user(user_id)}
    return {"tags": store.get_all_tags()}


# ---------------------------------------------------------------------------
# API: Meta + Users
# ---------------------------------------------------------------------------


@app.get("/api/meta")
def meta():
    return {
        "dataset": {
            "row_count": metadata.row_count,
            "num_categories": metadata.num_categories,
            "num_intents": metadata.num_intents,
        },
        "model": AGENT_MODEL,
        "model_short": AGENT_MODEL.split("/")[-1],
        "tokens": {
            "prompt": token_tracker.total_prompt_tokens,
            "completion": token_tracker.total_completion_tokens,
            "total": (
                token_tracker.total_prompt_tokens
                + token_tracker.total_completion_tokens
            ),
            "cost": round(
                (
                    token_tracker.total_prompt_tokens * 0.20
                    + token_tracker.total_completion_tokens * 0.60
                )
                / 1_000_000,
                6,
            ),
        },
    }


@app.get("/api/users")
def list_users():
    user_ids: set[str] = set()
    for chat in store.chats.values():
        if chat.user_id and chat.user_id != "default":
            user_ids.add(chat.user_id)
    for p in PROFILES_DIR.glob("*.json"):
        user_ids.add(p.stem)
    return {"users": sorted(user_ids)}


@app.delete("/api/users")
def delete_all_users():
    """Delete ALL users: profiles, chats, and checkpoint data."""
    store.delete_all_chats()
    try:
        with _graph_lock:
            _conn.execute("DELETE FROM checkpoints")
            _conn.execute("DELETE FROM writes")
            _conn.commit()
    except Exception:
        pass
    profile_count = 0
    for p in PROFILES_DIR.glob("*.json"):
        p.unlink(missing_ok=True)
        profile_count += 1
    return {"ok": True, "profiles_deleted": profile_count}


@app.delete("/api/users/{user_id}")
def delete_user(user_id: str):
    """Delete a user: their profile, all chats, and checkpoint data."""
    deleted_chats = store.delete_user_chats(user_id)
    try:
        with _graph_lock:
            for tid in deleted_chats:
                _conn.execute(
                    "DELETE FROM checkpoints WHERE thread_id = ?", (tid,)
                )
                _conn.execute(
                    "DELETE FROM writes WHERE thread_id = ?", (tid,)
                )
            _conn.commit()
    except Exception:
        pass
    profile_path = PROFILES_DIR / f"{user_id}.json"
    if profile_path.exists():
        profile_path.unlink(missing_ok=True)
    return {
        "ok": True,
        "deleted_chats": len(deleted_chats),
        "profile_deleted": not profile_path.exists(),
    }


# ---------------------------------------------------------------------------
# API: Memory Insights (aggregated view of all 4 memory types)
# ---------------------------------------------------------------------------


@app.get("/api/memory")
def memory_insights(session_id: str | None = Query(default=None)):
    # -- Working memory: current session stats --
    working = {"message_count": 0, "user_msgs": 0, "assistant_msgs": 0, "tool_msgs": 0, "estimated_tokens": 0}
    if session_id:
        try:
            config = {"configurable": {"thread_id": session_id}}
            state = _get_state_safe(config)
            if state and state.values:
                msgs = state.values.get("messages", [])
                working["message_count"] = len(msgs)
                for m in msgs:
                    if isinstance(m, HumanMessage):
                        working["user_msgs"] += 1
                    elif isinstance(m, AIMessage):
                        working["assistant_msgs"] += 1
                    elif isinstance(m, ToolMessage):
                        working["tool_msgs"] += 1
                working["estimated_tokens"] = sum(
                    len(getattr(m, "content", "") or "") * 4 // 3
                    for m in msgs
                )
        except Exception:
            pass

    # -- Episodic memory: all sessions --
    all_chats = list(store.chats.values())
    total_messages = sum(c.message_count for c in all_chats)
    avg_depth = round(total_messages / len(all_chats), 1) if all_chats else 0
    sessions_timeline = [
        {
            "thread_id": c.thread_id,
            "title": c.title,
            "user_id": c.user_id,
            "created_at": c.created_at,
            "message_count": c.message_count,
            "query_count": c.query_count,
        }
        for c in sorted(all_chats, key=lambda x: x.updated_at, reverse=True)[:20]
    ]
    episodic = {
        "total_sessions": len(all_chats),
        "total_messages": total_messages,
        "avg_depth": avg_depth,
        "sessions": sessions_timeline,
    }

    # -- Semantic memory: user profiles --
    profiles_data = []
    total_facts = 0
    for p in PROFILES_DIR.glob("*.json"):
        profile = load_profile(p.stem)
        facts = profile.facts
        total_facts += len(facts)
        profiles_data.append({
            "user_id": profile.user_id,
            "fact_count": len(facts),
            "facts": facts,
            "last_updated": profile.last_updated,
        })
    semantic = {
        "total_users": len(profiles_data),
        "total_facts": total_facts,
        "profiles": profiles_data,
    }

    # -- Procedural memory: tool usage + query type distribution --
    tool_usage: dict[str, int] = {}
    query_types: dict[str, int] = {}
    for chat in all_chats:
        for q in chat.queries:
            query_types[q.query_type] = query_types.get(q.query_type, 0) + 1
            for tc in q.tool_calls:
                name = tc.get("name", "unknown")
                tool_usage[name] = tool_usage.get(name, 0) + 1
    procedural = {
        "total_tool_calls": sum(tool_usage.values()),
        "unique_tools_used": len(tool_usage),
        "tool_usage": dict(sorted(tool_usage.items(), key=lambda x: -x[1])),
        "query_type_distribution": query_types,
    }

    return {
        "working": working,
        "episodic": episodic,
        "semantic": semantic,
        "procedural": procedural,
    }


# ---------------------------------------------------------------------------
# API: Query Recommendations (auto chips)
# ---------------------------------------------------------------------------


@app.get("/api/recommend")
def recommend(
    user_id: str = Query(default="default"),
    session_id: str | None = Query(default=None),
    use_past_sessions: bool = Query(default=False),
):
    """Generate query recommendations for auto suggestion chips."""
    current_messages = []
    if session_id:
        try:
            state = _get_state_safe(
                {"configurable": {"thread_id": session_id}}
            )
            if state and state.values:
                current_messages = state.values.get("messages", [])
        except Exception:
            pass

    recs = get_recommendations(
        user_id=user_id,
        current_messages=current_messages,
        current_thread_id=session_id,
        use_past_sessions=use_past_sessions,
    )
    return {"recommendations": recs}


# ---------------------------------------------------------------------------
# Static files: serve frontend/
# ---------------------------------------------------------------------------

_frontend_dir = Path(__file__).parent / "frontend"
_frontend_dir.mkdir(exist_ok=True)


@app.get("/")
def serve_index():
    index = _frontend_dir / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse(
        {"message": "Frontend not found. Place index.html in frontend/"},
        status_code=404,
    )


app.mount(
    "/frontend",
    StaticFiles(directory=str(_frontend_dir)),
    name="frontend",
)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
