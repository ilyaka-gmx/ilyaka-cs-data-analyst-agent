"""
CLI entry point for the Customer Service Data Analyst Agent.

Usage:
    uv run python main.py                          # default session
    uv run python main.py --session my_session      # named session
    uv run python main.py --user ilya               # named user profile
    uv run python main.py --health                  # run health checks and exit
"""

import argparse
import sqlite3
import sys
import uuid

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError

from src.agent import build_graph, token_tracker, tool_timer
from src.config import CHECKPOINTS_DB
from src.health import run_diagnostics, run_startup_checks


def print_reasoning_steps(messages: list, start_index: int) -> None:
    """Print tool calls and observations from the agent's reasoning."""
    for msg in messages[start_index:]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                args_str = ", ".join(f"{k}={v!r}" for k, v in tc.get("args", {}).items())
                print(f"  >> Tool: {tc['name']}({args_str})")
        elif isinstance(msg, ToolMessage):
            preview = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            print(f"  << Result: {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Customer Service Data Analyst Agent")
    parser.add_argument("--session", default=None, help="Session ID for conversation persistence")
    parser.add_argument("--user", default="default", help="User ID for profile persistence")
    parser.add_argument("--health", action="store_true", help="Run health checks and exit")
    args = parser.parse_args()

    if args.health:
        report = run_diagnostics()
        print(report.summary())
        sys.exit(1 if report.has_failures else 0)

    print("Running startup checks...")
    report = run_startup_checks()
    print(report.summary())
    if report.has_failures:
        print("\nFatal errors detected. Fix the issues above before running the agent.")
        sys.exit(1)
    print()

    conn = sqlite3.connect(str(CHECKPOINTS_DB), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = build_graph(checkpointer=checkpointer)

    session_id = args.session or str(uuid.uuid4())
    user_id = args.user
    print(f"Session: {session_id} | User: {user_id}")
    print("Type 'quit' or 'exit' to end. Type '--health' to run diagnostics.\n")

    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 12,
    }

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input == "--health":
            report = run_diagnostics()
            print(report.summary())
            continue

        current_state = graph.get_state(config)
        existing_count = len(current_state.values.get("messages", [])) if current_state.values else 0

        token_tracker.reset_query()
        tool_timer.reset_query()

        try:
            from langchain_core.messages import HumanMessage

            result = graph.invoke(
                {"messages": [HumanMessage(content=user_input)], "user_id": user_id},
                config=config,
            )
        except GraphRecursionError:
            print(
                "\nAgent: I wasn't able to complete the analysis within the allowed "
                "steps. Could you try rephrasing your question?"
            )
            print(f"  [{token_tracker.summary()}]\n")
            continue

        all_messages = result["messages"]
        print_reasoning_steps(all_messages, existing_count + 1)

        final_message = all_messages[-1]
        if isinstance(final_message, AIMessage) and final_message.content:
            print(f"\nAgent: {final_message.content}")

        print(f"  [{token_tracker.summary()}]\n")

    conn.close()


if __name__ == "__main__":
    main()
