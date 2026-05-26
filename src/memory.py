"""
User profile management — semantic memory backed by mem0.

Mem0 handles fact extraction, semantic deduplication, and vector-based
recall.  Storage: Qdrant in-process (on-disk at mem0_data/).
LLM and embeddings run on Nebius Token Factory (same API key).

The agent uses remember_fact / recall_profile / update_profile tools
to interact with this module.
"""

import logging
import os

from src.config import MEM0_CONFIG, MEM0_DATA_DIR

from mem0 import Memory

log = logging.getLogger(__name__)

QDRANT_LOCAL_POINT_LIMIT = 20_000

_memory: Memory | None = None


def _remove_stale_lock() -> None:
    """Remove a stale Qdrant .lock file left by a previous process."""
    lock_path = MEM0_DATA_DIR / ".lock"
    if not lock_path.exists():
        return
    try:
        lock_path.unlink()
        log.info("Removed stale Qdrant lock file: %s", lock_path)
    except OSError:
        log.warning("Could not remove Qdrant lock file: %s", lock_path)


def _get_memory() -> Memory:
    """Lazy-init the mem0 Memory instance (expensive first call)."""
    global _memory
    if _memory is None:
        log.info("Initializing mem0 Memory with Qdrant + Nebius Token Factory")
        try:
            _memory = Memory.from_config(MEM0_CONFIG)
        except Exception:
            log.warning("Qdrant init failed — removing stale lock and retrying")
            _remove_stale_lock()
            _memory = Memory.from_config(MEM0_CONFIG)
    return _memory


def add_fact(user_id: str, fact: str) -> str:
    """Add a fact to the user's semantic memory via mem0."""
    m = _get_memory()
    result = m.add(fact, user_id=user_id)
    log.info("mem0 add for user=%s: %s", user_id, result)
    return f"Remembered: {fact.strip()}"


def get_facts(user_id: str) -> str:
    """Retrieve all facts about a user from mem0."""
    m = _get_memory()
    result = m.get_all(filters={"user_id": user_id})
    memories = result.get("results", []) if isinstance(result, dict) else result
    if not memories:
        return "No profile information stored yet."
    facts = [entry.get("memory", str(entry)) for entry in memories]
    return "User profile:\n" + "\n".join(f"- {f}" for f in facts)


def search_facts(user_id: str, query: str, top_k: int = 10) -> str:
    """Semantic search over a user's stored facts."""
    m = _get_memory()
    result = m.search(query, filters={"user_id": user_id}, top_k=top_k)
    memories = result.get("results", []) if isinstance(result, dict) else result
    if not memories:
        return f"No memories found matching '{query}'."
    lines = []
    for entry in memories:
        text = entry.get("memory", str(entry))
        score = entry.get("score", "")
        suffix = f" (relevance: {score:.2f})" if isinstance(score, float) else ""
        lines.append(f"- {text}{suffix}")
    return "Matching memories:\n" + "\n".join(lines)


def replace_facts(user_id: str, facts: list[str]) -> str:
    """Replace the user's entire profile: delete all, then re-add."""
    m = _get_memory()
    m.delete_all(user_id=user_id)
    added = 0
    for fact in facts:
        if fact.strip():
            m.add(fact.strip(), user_id=user_id)
            added += 1
    return f"Profile updated: replaced with {added} facts."


def get_all_memories_raw(user_id: str) -> list[dict]:
    """Return raw mem0 memory dicts for a user (used by API endpoints)."""
    m = _get_memory()
    result = m.get_all(filters={"user_id": user_id})
    return result.get("results", []) if isinstance(result, dict) else result


def delete_user_memories(user_id: str) -> None:
    """Delete all memories for a user."""
    m = _get_memory()
    m.delete_all(user_id=user_id)


def get_storage_metrics() -> dict:
    """Return mem0 storage metrics for the admin panel."""
    total_bytes = 0
    file_count = 0
    if MEM0_DATA_DIR.exists():
        for dirpath, _dirnames, filenames in os.walk(MEM0_DATA_DIR):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_bytes += os.path.getsize(fp)
                file_count += 1

    if total_bytes < 1024:
        size_display = f"{total_bytes} B"
    elif total_bytes < 1024 * 1024:
        size_display = f"{total_bytes / 1024:.1f} KB"
    else:
        size_display = f"{total_bytes / (1024 * 1024):.1f} MB"

    return {
        "storage_bytes": total_bytes,
        "storage_display": size_display,
        "file_count": file_count,
        "qdrant_point_limit": QDRANT_LOCAL_POINT_LIMIT,
    }


def reset_memory_instance() -> None:
    """Reset the global Memory instance (for testing)."""
    global _memory
    _memory = None
