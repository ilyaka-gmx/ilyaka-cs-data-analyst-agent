"""
Tests for the FastAPI backend (api_server.py).

Uses FastAPI's TestClient for synchronous endpoint testing.
SSE streaming tests parse the event stream line by line.

IMPORTANT: Tests use isolated temp files for checkpoints.db and
session_store.json so they never touch production data.
"""

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite import SqliteSaver

import api_server
from api_server import app
from src.session_store import SessionStore


@pytest.fixture
def client(tmp_path):
    import src.config as cfg
    import src.memory as mem_mod

    orig_conn = api_server._conn
    orig_checkpointer = api_server._checkpointer
    orig_graph = api_server._graph
    orig_store = api_server.store
    orig_mem0_dir = cfg.MEM0_DATA_DIR

    # Redirect Mem0 storage to temp dir so tests don't conflict with a running server
    cfg.MEM0_DATA_DIR = tmp_path / "test_mem0_data"
    cfg.MEM0_CONFIG["vector_store"]["config"]["path"] = str(cfg.MEM0_DATA_DIR)
    mem_mod.reset_memory_instance()

    test_db = tmp_path / "test_checkpoints.db"
    test_conn = sqlite3.connect(str(test_db), check_same_thread=False)
    test_checkpointer = SqliteSaver(test_conn)
    test_graph = api_server.build_graph(checkpointer=test_checkpointer)
    test_store = SessionStore(path=tmp_path / "test_session_store.json")

    api_server._conn = test_conn
    api_server._checkpointer = test_checkpointer
    api_server._graph = test_graph
    api_server.store = test_store

    try:
        yield TestClient(app)
    finally:
        api_server._conn = orig_conn
        api_server._checkpointer = orig_checkpointer
        api_server._graph = orig_graph
        api_server.store = orig_store
        cfg.MEM0_DATA_DIR = orig_mem0_dir
        cfg.MEM0_CONFIG["vector_store"]["config"]["path"] = str(orig_mem0_dir)
        mem_mod.reset_memory_instance()
        test_conn.close()


# --- Health ---


def test_health_endpoint(client):
    import time

    # Health runs in a background thread; may still be "loading" initially
    for _ in range(10):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] != "loading":
            break
        time.sleep(1)

    assert data["status"] in ("healthy", "warning", "error", "loading")
    if data["status"] != "loading":
        assert isinstance(data["checks"], list)
        assert len(data["checks"]) >= 1
        for check in data["checks"]:
            assert "name" in check
            assert check["status"] in ("pass", "warn", "fail")


def test_health_refresh(client):
    resp = client.post("/api/health/refresh")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


# --- Meta ---


def test_meta_endpoint(client):
    resp = client.get("/api/meta")
    assert resp.status_code == 200
    data = resp.json()
    assert data["dataset"]["row_count"] > 0
    assert data["dataset"]["num_categories"] == 11
    assert "model" in data
    assert "tokens" in data
    assert "cost" in data["tokens"]


# --- Users ---


def test_users_endpoint(client):
    resp = client.get("/api/users")
    assert resp.status_code == 200
    data = resp.json()
    assert "users" in data
    assert isinstance(data["users"], list)


# --- Chats CRUD ---


def test_chats_list_empty_user(client):
    resp = client.get("/api/chats?user_id=nonexistent_test_user_xyz")
    assert resp.status_code == 200
    assert resp.json() == []


def test_chat_not_found(client):
    resp = client.get("/api/chats/nonexistent_thread_xyz")
    assert resp.status_code == 404


def test_chat_tag_add_remove(client):
    thread_id = "test_tag_thread"
    resp = client.post(
        f"/api/chats/{thread_id}/tags",
        json={"tag": "test_tag"},
    )
    assert resp.status_code == 200
    assert "test_tag" in resp.json()["tags"]

    resp = client.delete(f"/api/chats/{thread_id}/tags/test_tag")
    assert resp.status_code == 200
    assert "test_tag" not in resp.json()["tags"]


def test_chat_tag_empty_rejected(client):
    resp = client.post(
        "/api/chats/some_thread/tags",
        json={"tag": ""},
    )
    assert resp.status_code == 400


# --- Tags ---


def test_tags_endpoint(client):
    client.post("/api/chats/tag_test_t1/tags", json={"tag": "alpha"})
    client.post("/api/chats/tag_test_t2/tags", json={"tag": "beta"})
    resp = client.get("/api/tags")
    assert resp.status_code == 200
    data = resp.json()
    assert "alpha" in data["tags"]
    assert "beta" in data["tags"]


# --- Delete ---


def test_delete_single_chat(client):
    client.post("/api/chats/del_test_1/tags", json={"tag": "x"})
    resp = client.delete("/api/chats/del_test_1")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    resp2 = client.get("/api/chats/del_test_1")
    assert resp2.status_code == 404


def test_delete_all_chats(client):
    client.post("/api/chats/del_all_1/tags", json={"tag": "a"})
    client.post("/api/chats/del_all_2/tags", json={"tag": "b"})
    resp = client.delete("/api/chats")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


# --- Memory Insights ---


def test_memory_endpoint(client):
    resp = client.get("/api/memory")
    assert resp.status_code == 200
    data = resp.json()
    assert "working" in data
    assert "episodic" in data
    assert "semantic" in data
    assert "procedural" in data
    assert isinstance(data["episodic"]["total_sessions"], int)
    assert isinstance(data["semantic"]["total_facts"], int)
    assert isinstance(data["procedural"]["total_tool_calls"], int)


def test_memory_with_session(client):
    resp = client.get("/api/memory?session_id=nonexistent_session")
    assert resp.status_code == 200
    data = resp.json()
    assert data["working"]["message_count"] == 0


# --- SSE Chat (slow, LLM-dependent) ---


@pytest.mark.slow
def test_chat_sse_structured(client):
    """Send a structured query and verify SSE events arrive."""
    resp = client.post(
        "/api/chat",
        json={
            "query": "How many rows are in the dataset?",
            "session_id": "test_sse_session",
            "user_id": "test_user",
        },
    )
    assert resp.status_code == 200

    events = _parse_sse(resp.text)
    event_types = [e["event"] for e in events]

    assert "answer" in event_types, f"No answer event in: {event_types}"
    answer = next(e for e in events if e["event"] == "answer")
    assert "text" in answer["data"]
    assert "tokens" in answer["data"]
    assert answer["data"]["tokens"]["total"] > 0


@pytest.mark.slow
def test_chat_sse_out_of_scope(client):
    """Out-of-scope query should still return an answer event."""
    resp = client.post(
        "/api/chat",
        json={
            "query": "Who won the 2024 Champions League?",
            "session_id": "test_sse_oos",
            "user_id": "test_user",
        },
    )
    assert resp.status_code == 200

    events = _parse_sse(resp.text)
    event_types = [e["event"] for e in events]
    assert "answer" in event_types


@pytest.mark.slow
def test_chat_creates_session_store_entry(client):
    """After a chat, the session should appear in the chats list."""
    session_id = "test_store_entry"
    client.post(
        "/api/chat",
        json={
            "query": "What categories exist?",
            "session_id": session_id,
            "user_id": "test_api_user",
        },
    )
    resp = client.get("/api/chats?user_id=test_api_user")
    data = resp.json()
    thread_ids = [c["thread_id"] for c in data]
    assert session_id in thread_ids


@pytest.mark.slow
def test_chat_session_detail(client):
    """After a chat, GET /api/chats/{id} returns messages and traces."""
    session_id = "test_detail_session"
    client.post(
        "/api/chat",
        json={
            "query": "List all categories",
            "session_id": session_id,
            "user_id": "test_detail_user",
        },
    )
    resp = client.get(f"/api/chats/{session_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["messages"]) >= 2
    assert len(data["queries"]) >= 1
    assert data["queries"][0]["query_type"] in (
        "structured",
        "unstructured",
        "out_of_scope",
    )


# --- Helpers ---


@pytest.mark.slow
def test_recommend_endpoint(client):
    """Recommendation endpoint returns valid structure."""
    resp = client.get("/api/recommend?user_id=test_rec_user")
    assert resp.status_code == 200
    data = resp.json()
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)


@pytest.mark.slow
def test_recommend_with_past_sessions(client):
    """Recommendation with use_past_sessions flag."""
    resp = client.get(
        "/api/recommend?user_id=test_rec_user&use_past_sessions=true"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "recommendations" in data
    for rec in data["recommendations"]:
        assert "query" in rec
        assert isinstance(rec["query"], str)


# --- Helpers ---


def _parse_sse(text: str) -> list[dict]:
    """Parse SSE text into list of {event, data} dicts."""
    events = []
    current_event = None
    for line in text.split("\n"):
        if line.startswith("event: "):
            current_event = line[7:].strip()
        elif line.startswith("data: ") and current_event:
            try:
                data = json.loads(line[6:])
            except json.JSONDecodeError:
                data = line[6:]
            events.append({"event": current_event, "data": data})
            current_event = None
    return events
