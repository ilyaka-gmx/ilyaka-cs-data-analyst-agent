"""
Shared UI helper functions for the Streamlit web interface.

Handles: suggestion chips, reasoning step extraction, token formatting,
status bar data, conversation export, auto-tagging.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# --- Suggestion Chips ---

DEFAULT_SUGGESTIONS: list[dict[str, str]] = [
    {"label": "📂 What categories exist?", "query": "What categories exist in the dataset?"},
    {"label": "🔢 How many refund requests?", "query": "How many refund requests did we get?"},
    {"label": "📊 ACCOUNT intent distribution", "query": "What is the distribution of intents in the ACCOUNT category?"},
    {"label": "📝 3 examples from SHIPPING", "query": "Show me 3 examples from the SHIPPING category."},
    {"label": "📋 Summarize FEEDBACK", "query": "Summarize the FEEDBACK category."},
    {"label": "🔍 People wanting money back", "query": "Show me examples of people wanting their money back."},
]

CONTEXTUAL_SUGGESTIONS: dict[str, list[dict[str, str]]] = {
    "REFUND": [
        {"label": "📊 Refund intent distribution", "query": "What is the distribution of intents in the REFUND category?"},
        {"label": "📝 Refund response examples", "query": "Show me 3 examples from the REFUND category."},
    ],
    "SHIPPING": [
        {"label": "🔢 Shipping count", "query": "How many rows are in the SHIPPING category?"},
        {"label": "📋 Summarize SHIPPING", "query": "Summarize the SHIPPING category."},
    ],
    "ORDER": [
        {"label": "📊 Order intent distribution", "query": "What is the distribution of intents in the ORDER category?"},
        {"label": "📝 Order examples", "query": "Show me 3 examples from the ORDER category."},
    ],
    "ACCOUNT": [
        {"label": "🔢 Account count", "query": "How many rows are in the ACCOUNT category?"},
        {"label": "📝 Account examples", "query": "Show me 3 examples from the ACCOUNT category."},
    ],
}


def get_suggestions_after_response(
    query_type: str, response_text: str
) -> list[dict[str, str]]:
    """Return contextual suggestion chips based on the last response."""
    if query_type == "out_of_scope":
        return DEFAULT_SUGGESTIONS[:3]

    from src.data import CATEGORIES

    for cat in CATEGORIES:
        if cat in response_text.upper():
            if cat in CONTEXTUAL_SUGGESTIONS:
                return CONTEXTUAL_SUGGESTIONS[cat]

    return DEFAULT_SUGGESTIONS[:3]


# --- Reasoning Steps ---


def extract_reasoning_steps(
    messages: list, start_index: int = 0
) -> list[dict]:
    """Extract tool calls and results from agent messages.

    Returns list of dicts with type "tool_call" or "tool_result".
    """
    steps = []
    for msg in messages[start_index:]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                steps.append({
                    "type": "tool_call",
                    "name": tc["name"],
                    "args": tc.get("args", {}),
                })
        elif isinstance(msg, ToolMessage):
            preview = (
                msg.content[:300] + "..."
                if len(msg.content) > 300
                else msg.content
            )
            steps.append({
                "type": "tool_result",
                "name": getattr(msg, "name", "tool"),
                "content": preview,
            })
    return steps


def get_final_response(messages: list) -> str:
    """Get the final AI text response."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            return msg.content
    return "No response generated."


# --- Token Formatting ---


def format_per_query_tokens(query_tokens: dict) -> str:
    """Format per-query token usage for response footer."""
    p = query_tokens.get("prompt", 0)
    c = query_tokens.get("completion", 0)
    t = p + c
    cost = (p * 0.20 + c * 0.60) / 1_000_000
    return f"{t:,} tok · ~${cost:.4f}"


def format_response_footer(
    query_type: str,
    query_tokens: dict,
    duration_s: float,
    tool_count: int,
) -> str:
    """One-line footer below each response."""
    type_badge = {
        "structured": "📊 struct",
        "unstructured": "📝 open",
        "out_of_scope": "🚫 oos",
    }.get(query_type, "❓")
    token_str = format_per_query_tokens(query_tokens)
    return f"{type_badge} · {token_str} · ⏱ {duration_s:.1f}s · 🔧 {tool_count} calls"


# --- Auto-Tagging ---


def suggest_tags(messages: list) -> list[str]:
    """Suggest tags based on categories mentioned in the conversation."""
    from src.data import CATEGORIES

    text = " ".join(
        msg.content
        for msg in messages
        if isinstance(msg, (HumanMessage, AIMessage)) and msg.content
    ).upper()
    tags: set[str] = set()
    for cat in CATEGORIES:
        if cat in text:
            tags.add(cat.lower())
    return sorted(tags)[:5]


# --- Export ---


def export_conversation_markdown(messages: list) -> str:
    """Export conversation as markdown (user + assistant messages only)."""
    lines = ["# Conversation Export\n"]
    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"**User**: {msg.content}\n")
        elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            lines.append(f"**Agent**: {msg.content}\n")
    return "\n".join(lines)
