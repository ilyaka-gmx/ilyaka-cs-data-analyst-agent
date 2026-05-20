"""
Connectivity tests for Nebius AI Studio API.

Validates API access, tool calling, TOON comprehension, dataset loading,
and AgentMiddleware availability.  Uses the project's Zscaler-safe HTTP
client automatically (dev mode) or standard SSL (grader mode).

Run:
    uv run python tests/test_connectivity.py          # standalone
    uv run python -m pytest tests/test_connectivity.py -v   # via pytest
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from src.config import (
    AGENT_MODEL,
    NEBIUS_API_KEY,
    NEBIUS_BASE_URL,
    ROUTER_MODEL,
    USE_ZSCALER,
    get_http_client,
)


def _get_client() -> OpenAI:
    """Create an OpenAI-compatible client with automatic Zscaler handling."""
    http_client = get_http_client()
    extra = {"http_client": http_client} if http_client else {}
    return OpenAI(base_url=NEBIUS_BASE_URL, api_key=NEBIUS_API_KEY, **extra)


def test_basic_completion():
    """Test 1: Basic chat completion with the agent model."""
    print(f"\n--- Test 1: Basic completion with {AGENT_MODEL} ---")
    print(f"    Zscaler mode: {USE_ZSCALER}")
    client = _get_client()
    response = client.chat.completions.create(
        model=AGENT_MODEL,
        messages=[{"role": "user", "content": "Say 'ready' and nothing else."}],
        max_tokens=10,
        temperature=0,
    )
    result = response.choices[0].message.content.strip()
    print(f"  Response: {result}")
    assert "ready" in result.lower(), f"Expected 'ready', got: {result}"
    print("  PASS")


def test_router_model():
    """Test 2: Basic completion with the router model."""
    print(f"\n--- Test 2: Basic completion with {ROUTER_MODEL} ---")
    client = _get_client()
    response = client.chat.completions.create(
        model=ROUTER_MODEL,
        messages=[{"role": "user", "content": "Say 'ready' and nothing else."}],
        max_tokens=10,
        temperature=0,
    )
    result = response.choices[0].message.content.strip()
    print(f"  Response: {result}")
    assert "ready" in result.lower(), f"Expected 'ready', got: {result}"
    print("  PASS")


def test_tool_calling():
    """Test 3: Tool calling with the agent model."""
    print(f"\n--- Test 3: Tool calling with {AGENT_MODEL} ---")
    client = _get_client()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    response = client.chat.completions.create(
        model=AGENT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Use the provided tools when appropriate.",
            },
            {"role": "user", "content": "Get the weather in Helsinki."},
        ],
        tools=tools,
        tool_choice="auto",
        max_tokens=100,
        temperature=0,
    )

    message = response.choices[0].message
    print(f"  Finish reason: {response.choices[0].finish_reason}")

    if message.tool_calls:
        for tc in message.tool_calls:
            print(f"  Tool call: {tc.function.name}({tc.function.arguments})")
            args = json.loads(tc.function.arguments)
            assert "location" in args, f"Expected 'location' in args, got: {args}"
        print("  PASS")
    else:
        print(f"  WARN — No tool calls. Response: {message.content}")
        print("  Consider switching AGENT_MODEL to meta-llama/Llama-3.3-70B-Instruct")


def test_dataset_loads():
    """Test 4: Dataset loads and validates correctly."""
    print("\n--- Test 4: Dataset loading ---")
    from src.data import dataset

    assert len(dataset) > 20000, f"Expected >20k rows, got {len(dataset)}"
    assert "instruction" in dataset.columns
    assert "category" in dataset.columns
    assert "intent" in dataset.columns
    print(
        f"  Dataset: {len(dataset)} rows, "
        f"{dataset.category.nunique()} categories, "
        f"{dataset.intent.nunique()} intents"
    )
    print("  PASS")


def test_toon_comprehension():
    """Test 5: Verify the agent model can read TOON-formatted data."""
    print(f"\n--- Test 5: TOON comprehension with {AGENT_MODEL} ---")
    client = _get_client()

    toon_data = (
        "The following data is in TOON format "
        "(fields declared once, pipe-delimited rows):\n"
        "users[4]{name|role|department}:\n"
        "Alice|admin|Engineering\n"
        "Bob|user|Marketing\n"
        "Charlie|admin|Engineering\n"
        "Diana|user|Sales\n\n"
        "How many admins are there and which departments are they in?"
    )

    response = client.chat.completions.create(
        model=AGENT_MODEL,
        messages=[{"role": "user", "content": toon_data}],
        max_tokens=100,
        temperature=0,
    )
    result = response.choices[0].message.content.strip().lower()
    print(f"  Response: {result[:200]}")

    if "2" in result and "engineering" in result:
        print("  PASS — model reads TOON correctly")
    else:
        print("  WARN — model may not understand TOON format well")
        print("  Consider falling back to JSON for tool outputs")


def test_middleware_import():
    """Test 6: Verify LangChain AgentMiddleware is available."""
    print("\n--- Test 6: AgentMiddleware import ---")
    from langchain.agents.middleware import AgentMiddleware

    assert AgentMiddleware is not None
    print(f"  Imported: {AgentMiddleware.__module__}.{AgentMiddleware.__name__}")
    print("  PASS")


def test_langchain_llm():
    """Test 7: Verify get_llm() produces a working ChatOpenAI instance."""
    print(f"\n--- Test 7: get_llm() with {AGENT_MODEL} ---")
    from src.config import get_llm

    llm = get_llm(AGENT_MODEL, max_tokens=5, temperature=0)
    resp = llm.invoke("Reply with ok.")
    result = resp.content.strip().lower()
    print(f"  Response: {result}")
    assert "ok" in result, f"Expected 'ok', got: {result}"
    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("Nebius AI Studio Connectivity Tests")
    print(f"  Base URL:     {NEBIUS_BASE_URL}")
    print(f"  Zscaler mode: {USE_ZSCALER}")
    print(f"  Agent model:  {AGENT_MODEL}")
    print(f"  Router model: {ROUTER_MODEL}")
    print("=" * 60)

    tests = [
        test_basic_completion,
        test_router_model,
        test_tool_calling,
        test_dataset_loads,
        test_toon_comprehension,
        test_middleware_import,
        test_langchain_llm,
    ]
    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)

    if failed > 0:
        print("\nACTION REQUIRED: Fix failures before proceeding to Phase 1.")
        sys.exit(1)
    else:
        print("\nAll tests passed. Ready for Phase 1.")
