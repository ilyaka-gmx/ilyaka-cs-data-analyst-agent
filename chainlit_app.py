"""
Chainlit UI for the Customer Service Data Analyst Agent.

Run: uv run chainlit run chainlit_app.py
"""

import sqlite3
import time

import chainlit as cl
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError

from src.agent import build_graph, token_tracker, tool_timer
from src.config import CHECKPOINTS_DB
from src.data import metadata
from src.health import run_startup_checks
from src.ui_helpers import (
    SUGGESTION_CHIPS,
    SessionStats,
    extract_reasoning_steps,
    format_token_line,
    get_final_response,
)

_conn = sqlite3.connect(str(CHECKPOINTS_DB), check_same_thread=False)
_checkpointer = SqliteSaver(_conn)
_graph = build_graph(checkpointer=_checkpointer)

_health_report = None


def _get_health_report():
    global _health_report
    if _health_report is None:
        _health_report = run_startup_checks()
    return _health_report


def _make_chip_actions(count: int | None = None) -> list[cl.Action]:
    chips = SUGGESTION_CHIPS[:count] if count else SUGGESTION_CHIPS
    return [
        cl.Action(
            name="suggestion",
            payload={"query": chip["query"]},
            label=chip["label"],
        )
        for chip in chips
    ]


@cl.on_chat_start
async def start():
    """Initialize session: welcome message + suggestion chips."""
    report = _get_health_report()
    if report.has_failures:
        await cl.Message(
            content=f"⚠️ System health issues:\n```\n{report.summary()}\n```"
        ).send()

    session_id = cl.user_session.get("id")
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("user_id", "default")
    cl.user_session.set("stats", SessionStats())

    await cl.Message(
        content=(
            f"👋 **Welcome to the Customer Service Data Analyst!**\n\n"
            f"I can analyze the Bitext dataset: **{metadata.row_count:,}** records "
            f"across **{metadata.num_categories}** categories and "
            f"**{metadata.num_intents}** intents.\n\n"
            f"Try one of these queries, or ask your own:"
        ),
        actions=_make_chip_actions(),
    ).send()


@cl.action_callback("suggestion")
async def on_suggestion(action: cl.Action):
    """Handle suggestion chip click."""
    query = action.payload["query"]
    msg = cl.Message(content=query, author="User")
    await msg.send()
    await process_query(query)


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user message."""
    await process_query(message.content)


async def process_query(user_input: str):
    """Core query processing — shared by message handler and suggestion chips."""
    session_id = cl.user_session.get("session_id")
    user_id = cl.user_session.get("user_id", "default")
    stats: SessionStats = cl.user_session.get("stats")

    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 12,
    }

    current_state = _graph.get_state(config)
    existing_count = (
        len(current_state.values.get("messages", []))
        if current_state.values
        else 0
    )

    token_tracker.reset_query()
    tool_timer.reset_query()
    start_time = time.time()

    try:
        result = _graph.invoke(
            {"messages": [HumanMessage(content=user_input)], "user_id": user_id},
            config=config,
        )
    except GraphRecursionError:
        await cl.Message(
            content=(
                "I wasn't able to complete the analysis within the allowed steps. "
                "Could you try rephrasing your question?"
            )
        ).send()
        return

    response_time = time.time() - start_time

    steps = extract_reasoning_steps(result["messages"], start_index=existing_count + 1)

    i = 0
    while i < len(steps):
        step = steps[i]
        if step["type"] == "tool_call":
            async with cl.Step(name=step["name"], type="tool") as tool_step:
                tool_step.input = str(step["args"])
                if i + 1 < len(steps) and steps[i + 1]["type"] == "tool_result":
                    tool_step.output = steps[i + 1]["content"]
                    i += 1
        i += 1

    stats.record_query(steps, response_time)

    final = get_final_response(result["messages"])
    token_line = format_token_line(token_tracker)

    await cl.Message(
        content=f"{final}\n\n---\n_{token_line} · ⏱ {response_time:.1f}s_",
        actions=_make_chip_actions(3),
    ).send()
