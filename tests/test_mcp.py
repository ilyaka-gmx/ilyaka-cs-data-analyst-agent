"""
Tests for the MCP server (src/mcp_server.py).

Verifies that all 7 data tools are registered, callable, and return
correct results via the FastMCP in-process Client.
"""

import pytest
from fastmcp import Client

from src.mcp_server import mcp


@pytest.mark.asyncio
async def test_mcp_server_has_7_tools():
    tools = await mcp.list_tools()
    assert len(tools) == 7


@pytest.mark.asyncio
async def test_mcp_all_tool_names_registered():
    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    expected = {
        "list_categories",
        "list_intents",
        "count_rows",
        "get_distribution",
        "get_examples",
        "search_instructions",
        "summarize_responses",
    }
    assert names == expected


@pytest.mark.asyncio
async def test_mcp_no_memory_tools():
    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert "remember_fact" not in names
    assert "recall_profile" not in names


@pytest.mark.asyncio
async def test_mcp_list_categories():
    async with Client(mcp) as client:
        result = await client.call_tool("list_categories", {})
        text = result.content[0].text
        assert "REFUND" in text
        assert "ORDER" in text


@pytest.mark.asyncio
async def test_mcp_list_intents_all():
    async with Client(mcp) as client:
        result = await client.call_tool("list_intents", {})
        text = result.content[0].text
        assert "get_refund" in text


@pytest.mark.asyncio
async def test_mcp_list_intents_filtered():
    async with Client(mcp) as client:
        result = await client.call_tool(
            "list_intents", {"category": "REFUND"}
        )
        text = result.content[0].text
        assert "get_refund" in text
        assert "cancel_order" not in text


@pytest.mark.asyncio
async def test_mcp_count_rows_total():
    async with Client(mcp) as client:
        result = await client.call_tool("count_rows", {})
        text = result.content[0].text
        assert "26,872" in text


@pytest.mark.asyncio
async def test_mcp_count_rows_filtered():
    async with Client(mcp) as client:
        result = await client.call_tool(
            "count_rows", {"category": "REFUND"}
        )
        text = result.content[0].text
        assert "rows" in text.lower()


@pytest.mark.asyncio
async def test_mcp_get_distribution():
    async with Client(mcp) as client:
        result = await client.call_tool(
            "get_distribution", {"group_by": "category"}
        )
        text = result.content[0].text
        assert "Distribution" in text
        assert "REFUND" in text


@pytest.mark.asyncio
async def test_mcp_get_examples():
    async with Client(mcp) as client:
        result = await client.call_tool(
            "get_examples", {"n": 3, "category": "SHIPPING"}
        )
        text = result.content[0].text
        assert "instruction" in text.lower() or "intent" in text.lower()


@pytest.mark.asyncio
async def test_mcp_search_instructions():
    async with Client(mcp) as client:
        result = await client.call_tool(
            "search_instructions", {"query": "money back"}
        )
        text = result.content[0].text
        assert "money back" in text.lower() or "refund" in text.lower()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_mcp_summarize_responses():
    async with Client(mcp) as client:
        result = await client.call_tool(
            "summarize_responses", {"category": "REFUND", "n_sample": 5}
        )
        text = result.content[0].text
        assert len(text) > 50
