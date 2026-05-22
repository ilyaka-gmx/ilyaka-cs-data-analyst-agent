"""
FastMCP server exposing Bitext dataset analysis tools.

Exposes 7 data/analysis tools from src/tools.py as MCP-compatible endpoints.
Memory tools (remember_fact, recall_profile) are excluded — they require
agent-level user context (contextvars) that MCP clients don't provide.

Usage:
    uv run fastmcp dev inspector src/mcp_server.py   # Inspector UI
    uv run fastmcp run src/mcp_server.py:mcp          # stdio transport
    uv run python src/mcp_server.py                    # direct run
"""

import sys
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path so `src.*` imports resolve
# when fastmcp loads this file standalone (outside pytest/uv run).
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from fastmcp import FastMCP

from src.tools import (
    count_rows as _count_rows,
    get_distribution as _get_distribution,
    get_examples as _get_examples,
    list_categories as _list_categories,
    list_intents as _list_intents,
    search_instructions as _search_instructions,
    summarize_responses as _summarize_responses,
)

mcp = FastMCP(
    "Bitext Customer Service Dataset",
    instructions=(
        "Query and analyze the Bitext customer service training dataset "
        "(26,872 rows, 11 categories, 27 intents). "
        "Tools cover listing, counting, distribution, examples, search, "
        "and LLM-powered response summarization."
    ),
)


@mcp.tool
def list_categories() -> str:
    """List all 11 unique categories in the Bitext customer service dataset."""
    return _list_categories.invoke({})


@mcp.tool
def list_intents(category: Optional[str] = None) -> str:
    """List intents in the dataset, optionally filtered by category.

    Args:
        category: Filter intents by this category (e.g., 'ORDER', 'REFUND').
                  If omitted, returns all 27 intents.
    """
    return _list_intents.invoke({"category": category})


@mcp.tool
def count_rows(
    category: Optional[str] = None, intent: Optional[str] = None
) -> str:
    """Count matching rows, optionally filtered by category and/or intent.

    Args:
        category: Filter by category (e.g., 'REFUND').
        intent: Filter by intent (e.g., 'get_refund').
    """
    return _count_rows.invoke({"category": category, "intent": intent})


@mcp.tool
def get_distribution(
    group_by: str, filter_category: Optional[str] = None
) -> str:
    """Get the frequency distribution of categories or intents.

    Args:
        group_by: Column to group by — 'category' or 'intent'.
        filter_category: If grouping by intent, optionally filter to this
                         category first.
    """
    return _get_distribution.invoke(
        {"group_by": group_by, "filter_category": filter_category}
    )


@mcp.tool
def get_examples(
    n: int = 5,
    category: Optional[str] = None,
    intent: Optional[str] = None,
) -> str:
    """Get sample rows from the dataset in TOON format.

    Args:
        n: Number of examples to return (1–20, default 5).
        category: Filter by category.
        intent: Filter by intent.
    """
    return _get_examples.invoke(
        {"n": n, "category": category, "intent": intent}
    )


@mcp.tool
def search_instructions(query: str, n: int = 5) -> str:
    """Search customer instructions by keyword or phrase (case-insensitive).

    Args:
        query: Search term to find in customer instructions.
        n: Maximum number of results (1–20, default 5).
    """
    return _search_instructions.invoke({"query": query, "n": n})


@mcp.tool
def summarize_responses(
    category: Optional[str] = None,
    intent: Optional[str] = None,
    n_sample: int = 15,
) -> str:
    """Summarize how agents typically respond to a category or intent (LLM-powered).

    Samples responses from the dataset and uses an LLM to produce a summary
    of patterns, tone, and common approaches.

    Args:
        category: Filter by category.
        intent: Filter by intent.
        n_sample: Number of responses to sample (5–30, default 15).
    """
    return _summarize_responses.invoke(
        {"category": category, "intent": intent, "n_sample": n_sample}
    )


if __name__ == "__main__":
    mcp.run()
