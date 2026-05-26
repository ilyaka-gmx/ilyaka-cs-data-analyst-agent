"""
Query recommendation engine — suggests follow-up queries based on
conversation history, user profile, and optionally past sessions.

Uses the router model (Qwen3-30B) for cheap, fast recommendations.
Shared by both the /api/recommend endpoint (auto chips) and the
agent's inline recommendation mode (explicit "what should I query?").
"""

import json
import logging
import re

from src.config import ROUTER_MODEL, get_llm
from src.data import CATEGORIES
from src.session_store import ChatMetadata, store

log = logging.getLogger(__name__)

RECOMMEND_SYSTEM_PROMPT = """You are a query recommendation engine for a customer service dataset analyst.

The dataset is the Bitext Customer Service dataset with {num_categories} categories: {categories}.

Given:
- The current conversation (if any)
- Summaries of the user's past sessions (if provided — these are CRITICAL)

Suggest 2-3 follow-up queries the user might find valuable. Prioritize:
1. MOST IMPORTANT: If past session summaries are provided, base your recommendations primarily on what the user explored before — suggest deeper dives, comparisons, or extensions of their past queries. Explicitly reference what they did before in the "reason" field.
2. Natural follow-ups to what they just asked about in the current conversation
3. Areas of the dataset they haven't explored yet

Respond with JSON:
{{"recommendations": [
  {{"query": "<the suggested query text>", "reason": "<brief why — reference past session topics if available>"}},
  ...
]}}

Rules:
- Queries MUST be answerable by the dataset. The dataset has ONLY these columns: category, intent, instruction (customer text), response (agent text). It has NO location, date, customer name, sentiment, priority, or satisfaction data. NEVER suggest queries that filter by information not in the dataset.
- Be specific: "Show 5 examples from REFUND" not "explore more data".
- If past sessions mention specific queries/tools/categories, build on those topics.
- If there's no conversation yet and no past sessions, suggest good starting queries based on the dataset categories and intents.
- Do NOT suggest queries the user already asked in this session.
- Keep each "reason" under 30 words. Keep each "query" under 20 words.
- Respond ONLY with the JSON object — no markdown fences, no extra text."""


def _build_session_summary(user_id: str, exclude_thread: str | None = None) -> str:
    """Build a compact summary of the user's past sessions from the store."""
    user_chats = [
        c for c in store.chats.values()
        if c.user_id == user_id and c.thread_id != exclude_thread
    ]
    user_chats.sort(key=lambda c: c.updated_at, reverse=True)
    user_chats = user_chats[:10]

    if not user_chats:
        log.info("No past sessions found for user=%s (exclude=%s, total chats=%d)",
                 user_id, exclude_thread, len(store.chats))
        return ""

    log.info("Building session summary for user=%s: %d past sessions found",
             user_id, len(user_chats))

    lines = ["The user had these previous sessions (use them to inform recommendations):"]
    for chat in user_chats:
        queries_summary = []
        for q in chat.queries[:5]:
            tools_used = [tc.get("name", "?") for tc in q.tool_calls]
            tools_str = f" (tools: {', '.join(tools_used)})" if tools_used else ""
            queries_summary.append(f"  - [{q.query_type}] {q.user_message[:80]}{tools_str}")
        lines.append(f"Session '{chat.title}' ({chat.query_count} queries, id={chat.thread_id}):")
        if queries_summary:
            lines.extend(queries_summary)
        else:
            lines.append(f"  (title indicates topic: {chat.title[:120]})")
    return "\n".join(lines)


def _build_conversation_context(messages: list) -> str:
    """Build compact conversation context from message objects."""
    if not messages:
        return ""
    lines = ["Current conversation:"]
    for msg in messages[-10:]:
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        if role == "human":
            lines.append(f"  User: {content[:120]}")
        elif role == "ai" and content and not getattr(msg, "tool_calls", None):
            lines.append(f"  Agent: {content[:120]}")
    return "\n".join(lines)


def get_recommendations(
    user_id: str = "default",
    current_messages: list | None = None,
    current_thread_id: str | None = None,
    use_past_sessions: bool = False,
) -> list[dict]:
    """Generate query recommendations.

    Args:
        user_id: The user to recommend for.
        current_messages: Messages from the current conversation (LangChain message objects).
        current_thread_id: Current thread ID (excluded from past session summary).
        use_past_sessions: Whether to include past session context.

    Returns:
        List of {"query": str, "reason": str} dicts.
    """
    conversation_text = ""
    if current_messages:
        conversation_text = _build_conversation_context(current_messages)

    session_text = ""
    if use_past_sessions:
        session_text = _build_session_summary(user_id, exclude_thread=current_thread_id)
        log.info("Past session context (use_past=%s, user=%s, exclude=%s): %s",
                 use_past_sessions, user_id, current_thread_id,
                 session_text[:200] if session_text else "<empty>")

    context_parts = [p for p in [conversation_text, session_text] if p]
    context = "\n\n".join(context_parts) if context_parts else "No conversation or profile yet."

    sys_prompt = RECOMMEND_SYSTEM_PROMPT.format(
        num_categories=len(CATEGORIES),
        categories=", ".join(CATEGORIES),
    )

    llm = get_llm(ROUTER_MODEL, temperature=0.7, max_tokens=800)

    from langchain_core.messages import HumanMessage, SystemMessage
    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=context),
    ]

    try:
        response = llm.invoke(messages)
        text = response.content

        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        json_match = re.search(r"\{[^{}]*\"recommendations\"\s*:\s*\[.*?\]\s*\}", cleaned, flags=re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            recs = parsed.get("recommendations", [])
            return [
                {"query": r["query"], "reason": r.get("reason", "")}
                for r in recs
                if isinstance(r, dict) and "query" in r
            ][:3]

        log.warning("Could not parse recommendations JSON from: %s", text[:200])
        return []

    except Exception as e:
        log.error("Recommendation engine failed: %s", e)
        return []


def get_recommendation_context_for_agent(
    user_id: str = "default",
    current_thread_id: str | None = None,
    use_past_sessions: bool = False,
) -> str:
    """Build a context block for the agent prompt when query_type=recommend.

    Instead of calling the recommender LLM, this provides raw context
    so the agent LLM itself can reason about what to recommend.
    """
    from src.memory import get_facts

    profile_text = get_facts(user_id)

    parts = []
    if "no profile" not in profile_text.lower():
        parts.append(profile_text)

    if use_past_sessions:
        session_text = _build_session_summary(user_id, exclude_thread=current_thread_id)
        if session_text:
            parts.append(session_text)

    categories_explored: set[str] = set()
    tools_used_total: dict[str, int] = {}

    user_chats = [c for c in store.chats.values() if c.user_id == user_id]
    for chat in user_chats:
        for q in chat.queries:
            for tc in q.tool_calls:
                name = tc.get("name", "")
                tools_used_total[name] = tools_used_total.get(name, 0) + 1
                for arg_val in tc.get("args", {}).values():
                    if isinstance(arg_val, str) and arg_val.upper() in [c.upper() for c in CATEGORIES]:
                        categories_explored.add(arg_val.upper())

    unexplored = [c for c in CATEGORIES if c.upper() not in categories_explored]

    if categories_explored:
        parts.append(f"Categories already explored: {', '.join(sorted(categories_explored))}")
    if unexplored:
        parts.append(f"Categories NOT yet explored: {', '.join(unexplored)}")
    if tools_used_total:
        parts.append(f"Tools used so far: {', '.join(f'{k}({v}x)' for k,v in sorted(tools_used_total.items(), key=lambda x: -x[1]))}")

    return "\n".join(parts) if parts else "No prior context available."
