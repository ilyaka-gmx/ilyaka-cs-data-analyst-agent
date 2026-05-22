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
from src.session_store import QueryTrace, store
from src.ui_helpers import suggest_tags

log = logging.getLogger(__name__)

app = FastAPI(title="CS Data Analyst API")

_conn = sqlite3.connect(str(CHECKPOINTS_DB), check_same_thread=False)
_checkpointer = SqliteSaver(_conn)
_graph = build_graph(checkpointer=_checkpointer)


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

        try:
            for event in _graph.stream(
                {
                    "messages": [HumanMessage(content=query)],
                    "user_id": user_id,
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
                        yield f"event: route\ndata: {json.dumps(step)}\n\n"

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
                                    yield f"event: tool_call\ndata: {json.dumps(step)}\n\n"
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
                            yield f"event: tool_result\ndata: {json.dumps(step)}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
            return

        duration = time.time() - start_time
        query_tokens = {
            "prompt": token_tracker.query_prompt_tokens,
            "completion": token_tracker.query_completion_tokens,
            "total": (
                token_tracker.query_prompt_tokens
                + token_tracker.query_completion_tokens
            ),
        }
        tool_count = sum(
            1 for s in reasoning_steps if s.get("type") == "tool_call"
        )

        auto_tags = []
        try:
            state = _graph.get_state(config)
            if state and state.values:
                auto_tags = suggest_tags(state.values.get("messages", []))
                for tag in auto_tags:
                    store.add_tag(session_id, tag)
        except Exception:
            pass

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
        )
        store.log_query(session_id, trace)

        answer_data = {
            "text": final_answer,
            "tokens": query_tokens,
            "duration_ms": int(duration * 1000),
            "query_type": query_type,
            "tool_count": tool_count,
            "auto_tags": auto_tags,
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
):
    chats = store.list_chats(
        tag_filter=[tag] if tag else None,
        search=search,
    )
    filtered = [c for c in chats if c.user_id == user_id]
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
        for c in filtered
    ]


@app.get("/api/chats/{thread_id}")
def get_chat(thread_id: str):
    chat = store.chats.get(thread_id)
    if not chat:
        return JSONResponse({"error": "Not found"}, status_code=404)

    messages = []
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = _graph.get_state(config)
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


# ---------------------------------------------------------------------------
# API: Tags (all unique tags for filter UI)
# ---------------------------------------------------------------------------


@app.get("/api/tags")
def all_tags():
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
        if chat.user_id:
            user_ids.add(chat.user_id)
    for p in PROFILES_DIR.glob("*.json"):
        user_ids.add(p.stem)
    user_ids.discard("default")
    return {"users": sorted(user_ids)}


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
