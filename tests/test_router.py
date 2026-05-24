"""Tests for the query router classification.

All tests make real LLM calls to the router model (ROUTER_MODEL).
Run: uv run pytest tests/test_router.py -v
"""

import pytest

from src.router import classify_query


# --- Structured queries (from crosscheck card) ---


@pytest.mark.slow
def test_structured_categories():
    result = classify_query("What categories exist in the dataset?")
    assert result.classification == "structured"


@pytest.mark.slow
def test_structured_count():
    result = classify_query("How many refund requests did we get?")
    assert result.classification == "structured"


@pytest.mark.slow
def test_structured_examples():
    result = classify_query("Show me 5 examples of the SHIPPING category.")
    assert result.classification == "structured"


@pytest.mark.slow
def test_structured_distribution():
    result = classify_query(
        "What is the distribution of intents in the ACCOUNT category?"
    )
    assert result.classification == "structured"


@pytest.mark.slow
def test_structured_search():
    result = classify_query("Show me examples of people wanting their money back.")
    assert result.classification == "structured"


# --- Unstructured queries (from crosscheck card) ---


@pytest.mark.slow
def test_unstructured_summarize():
    result = classify_query("Summarize how agents respond to complaint intents.")
    assert result.classification == "unstructured"


# --- Out-of-scope queries (from crosscheck card) ---


@pytest.mark.slow
def test_oos_crm():
    result = classify_query("What's the best CRM software for handling complaints?")
    assert result.classification == "out_of_scope"


@pytest.mark.slow
def test_oos_president():
    result = classify_query("Who is the president of France?")
    assert result.classification == "out_of_scope"


@pytest.mark.slow
def test_oos_poem():
    result = classify_query("Write me a poem about customer service.")
    assert result.classification == "out_of_scope"


# --- Edge cases ---


@pytest.mark.slow
def test_ambiguous_but_in_scope():
    """Questions about the dataset that could seem general but are in-scope."""
    result = classify_query("What are the most common customer issues?")
    assert result.classification in ("structured", "unstructured")


@pytest.mark.slow
def test_unstructured_patterns():
    result = classify_query(
        "How do customer service representatives typically respond to cancellation requests?"
    )
    assert result.classification == "unstructured"


@pytest.mark.slow
def test_reasoning_populated():
    """Verify the router provides reasoning for its classification."""
    result = classify_query("How many complaints did we get?")
    assert result.classification == "structured"
    assert len(result.reasoning) > 10


# --- Memory / profile queries (Gate 5) ---


@pytest.mark.slow
def test_memory_share_personal_info():
    """User sharing personal info should route to the agent (not out_of_scope)."""
    result = classify_query("My name is Alex and I work in the refund department.")
    assert result.classification in ("structured", "unstructured"), (
        f"Personal info should route to agent, got: {result.classification}"
    )


@pytest.mark.slow
def test_memory_recall():
    """Asking what the agent remembers should route to the agent."""
    result = classify_query("What do you remember about me?")
    assert result.classification in ("structured", "unstructured"), (
        f"Memory recall should route to agent, got: {result.classification}"
    )


# --- Recommendation queries (Gate 8) ---


@pytest.mark.slow
def test_recommend_what_should_i_query():
    result = classify_query("What should I query next?")
    assert result.classification == "recommend"


@pytest.mark.slow
def test_recommend_suggest_something():
    result = classify_query("Suggest something interesting to explore")
    assert result.classification == "recommend"


@pytest.mark.slow
def test_recommend_any_suggestions():
    result = classify_query("Any suggestions for what to look at?")
    assert result.classification == "recommend"
