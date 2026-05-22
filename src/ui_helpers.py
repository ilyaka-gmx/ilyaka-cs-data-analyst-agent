"""
Shared UI helper functions used by both Chainlit and Streamlit UIs.

Provides formatting for reasoning steps, token summaries, suggestion chips,
session stats, and conversation export.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

SUGGESTION_CHIPS: list[dict[str, str]] = [
    {"label": "📂 What categories exist?", "query": "What categories exist in the dataset?"},
    {"label": "🔢 How many refund requests?", "query": "How many refund requests did we get?"},
    {"label": "📊 Intent distribution in ACCOUNT", "query": "What is the distribution of intents in the ACCOUNT category?"},
    {"label": "📝 3 examples from SHIPPING", "query": "Show me 3 examples from the SHIPPING category."},
    {"label": "📋 Summarize FEEDBACK", "query": "Summarize the FEEDBACK category."},
    {"label": "🔍 People wanting money back", "query": "Show me examples of people wanting their money back."},
]


def extract_reasoning_steps(messages: list, start_index: int = 0) -> list[dict]:
    """Extract tool calls and results from agent messages.

    Returns list of dicts with type "tool_call" or "tool_result".
    Pairs are ordered: tool_call followed by its tool_result.
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
            preview = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
            steps.append({
                "type": "tool_result",
                "name": msg.name if hasattr(msg, "name") else "tool",
                "content": preview,
            })
    return steps


def get_final_response(messages: list) -> str:
    """Get the final AI text response from a message list."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            return msg.content
    return "No response generated."


def format_token_summary(token_tracker) -> dict:
    """Format token usage data for display.

    Returns dict with: prompt_tokens, completion_tokens, total_tokens, estimated_cost.
    """
    prompt = token_tracker.total_prompt_tokens
    completion = token_tracker.total_completion_tokens
    total = prompt + completion
    cost = (prompt * 0.20 + completion * 0.60) / 1_000_000
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "estimated_cost": cost,
    }


def format_token_line(token_tracker) -> str:
    """One-line token summary for display below responses."""
    s = format_token_summary(token_tracker)
    return (
        f"📊 {s['total_tokens']:,} tokens "
        f"({s['prompt_tokens']:,} in + {s['completion_tokens']:,} out) "
        f"· ~${s['estimated_cost']:.4f}"
    )


class SessionStats:
    """Track session-level statistics for the Analyst Dashboard."""

    def __init__(self):
        self.query_count: int = 0
        self.tool_call_count: int = 0
        self.total_response_time: float = 0.0

    def record_query(self, reasoning_steps: list[dict], response_time: float):
        self.query_count += 1
        self.tool_call_count += sum(
            1 for s in reasoning_steps if s["type"] == "tool_call"
        )
        self.total_response_time += response_time

    @property
    def avg_response_time(self) -> float:
        return self.total_response_time / max(self.query_count, 1)


def export_conversation_markdown(messages: list) -> str:
    """Export conversation history as a markdown string."""
    lines = ["# Conversation Export\n"]
    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"**User**: {msg.content}\n")
        elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            lines.append(f"**Agent**: {msg.content}\n")
    return "\n".join(lines)
