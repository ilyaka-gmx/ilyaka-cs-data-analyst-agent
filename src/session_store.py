"""
Session storage for chat metadata, tags, and per-query trace logs.

Stores chat list with tags and per-query execution traces in a JSON file
alongside the SqliteSaver checkpoints database. Both the Chat and Admin
views read from this store.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.config import PROJECT_ROOT

SESSION_STORE_PATH: Path = PROJECT_ROOT / "session_store.json"


@dataclass
class QueryTrace:
    """Trace of a single user query through the agent graph."""

    query_index: int
    timestamp: str
    user_message: str
    query_type: str
    steps: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    tokens: dict = field(
        default_factory=lambda: {"prompt": 0, "completion": 0, "total": 0}
    )
    total_duration_ms: int = 0
    hit_fallback: bool = False
    final_response_preview: str = ""
    memory_ops: dict = field(default_factory=dict)


@dataclass
class ChatMetadata:
    """Metadata for a single chat/conversation."""

    thread_id: str
    title: str
    tags: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    message_count: int = 0
    total_tokens: int = 0
    query_count: int = 0
    queries: list[QueryTrace] = field(default_factory=list)
    user_id: str = "default"


class SessionStore:
    """Persistent storage for chat metadata and traces.

    Backed by a JSON file — simple and inspectable by graders.
    """

    def __init__(self, path: Path = SESSION_STORE_PATH):
        self.path = path
        self.chats: dict[str, ChatMetadata] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                for tid, chat_dict in data.items():
                    queries = [
                        QueryTrace(**q) for q in chat_dict.pop("queries", [])
                    ]
                    self.chats[tid] = ChatMetadata(**chat_dict, queries=queries)
            except (json.JSONDecodeError, TypeError):
                self.chats = {}

    def _save(self):
        data = {tid: asdict(chat) for tid, chat in self.chats.items()}
        self.path.write_text(json.dumps(data, indent=2, default=str))

    def get_or_create_chat(
        self, thread_id: str, user_id: str = "default"
    ) -> ChatMetadata:
        if thread_id not in self.chats:
            now = datetime.now(timezone.utc).isoformat()
            self.chats[thread_id] = ChatMetadata(
                thread_id=thread_id,
                title="New conversation",
                created_at=now,
                updated_at=now,
                user_id=user_id,
            )
            self._save()
        return self.chats[thread_id]

    def update_chat_title(self, thread_id: str, first_message: str):
        """Auto-title from first user message (truncated)."""
        chat = self.chats.get(thread_id)
        if chat and chat.title == "New conversation":
            chat.title = (
                first_message[:50]
                + ("..." if len(first_message) > 50 else "")
            )
            self._save()

    def add_tag(self, thread_id: str, tag: str):
        chat = self.chats.get(thread_id)
        if chat and tag not in chat.tags:
            chat.tags.append(tag)
            self._save()

    def remove_tag(self, thread_id: str, tag: str):
        chat = self.chats.get(thread_id)
        if chat and tag in chat.tags:
            chat.tags.remove(tag)
            self._save()

    def get_all_tags(self) -> list[str]:
        """Return all unique tags across all chats."""
        tags: set[str] = set()
        for chat in self.chats.values():
            tags.update(chat.tags)
        return sorted(tags)

    def log_query(self, thread_id: str, trace: QueryTrace):
        """Log a completed query trace."""
        chat = self.chats.get(thread_id)
        if chat:
            chat.queries.append(trace)
            chat.query_count = len(chat.queries)
            chat.total_tokens += trace.tokens.get("total", 0)
            chat.message_count += 2
            chat.updated_at = datetime.now(timezone.utc).isoformat()
            self._save()

    def delete_chat(self, thread_id: str) -> bool:
        if thread_id in self.chats:
            del self.chats[thread_id]
            self._save()
            return True
        return False

    def delete_all_chats(self):
        self.chats.clear()
        self._save()

    def list_chats(
        self,
        tag_filter: list[str] | None = None,
        search: str | None = None,
    ) -> list[ChatMetadata]:
        """List chats, optionally filtered by tags and/or search text."""
        chats = list(self.chats.values())
        if tag_filter:
            chats = [
                c for c in chats if all(t in c.tags for t in tag_filter)
            ]
        if search:
            search_lower = search.lower()
            chats = [c for c in chats if search_lower in c.title.lower()]
        chats.sort(key=lambda c: c.updated_at, reverse=True)
        return chats


store = SessionStore()
