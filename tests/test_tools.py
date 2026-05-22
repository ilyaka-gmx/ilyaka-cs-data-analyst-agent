"""Unit tests for all tools in src/tools.py and DatasetMetadata."""

import pytest

from src.tools import (
    count_rows,
    get_distribution,
    get_examples,
    list_categories,
    list_intents,
    recall_profile,
    remember_fact,
    search_instructions,
    set_current_user_id,
    summarize_responses,
)


# --- Tool 1: list_categories ---


def test_list_categories_returns_11():
    result = list_categories.invoke({})
    for cat in ["ACCOUNT", "ORDER", "REFUND", "SHIPPING", "FEEDBACK"]:
        assert cat in result


def test_list_categories_deterministic():
    r1 = list_categories.invoke({})
    r2 = list_categories.invoke({})
    assert r1 == r2


# --- Tool 2: list_intents ---


def test_list_intents_all():
    result = list_intents.invoke({})
    assert "cancel_order" in result
    assert "get_refund" in result


def test_list_intents_filtered():
    result = list_intents.invoke({"category": "REFUND"})
    assert "get_refund" in result
    assert "cancel_order" not in result


def test_list_intents_invalid_category():
    result = list_intents.invoke({"category": "NONEXISTENT"})
    assert "not found" in result.lower() or "invalid" in result.lower()


# --- Tool 3: count_rows ---


def test_count_rows_total():
    result = count_rows.invoke({})
    assert "26,872" in result or "26872" in result


def test_count_rows_by_category():
    result = count_rows.invoke({"category": "ORDER"})
    assert any(char.isdigit() for char in result)
    assert "ORDER" in result


def test_count_rows_by_intent():
    result = count_rows.invoke({"intent": "get_refund"})
    assert any(char.isdigit() for char in result)


def test_count_rows_invalid_category():
    result = count_rows.invoke({"category": "INVALID"})
    assert "not found" in result.lower()


# --- Tool 4: get_distribution ---


def test_get_distribution_by_category():
    result = get_distribution.invoke({"group_by": "category"})
    assert "ORDER" in result
    assert "REFUND" in result


def test_get_distribution_intents_in_category():
    result = get_distribution.invoke(
        {"group_by": "intent", "filter_category": "REFUND"}
    )
    assert "get_refund" in result
    assert "cancel_order" not in result


def test_get_distribution_invalid_group_by():
    result = get_distribution.invoke({"group_by": "invalid_column"})
    assert "must be" in result.lower()


# --- Tool 5: get_examples ---


def test_get_examples_returns_n():
    result = get_examples.invoke({"n": 3, "category": "SHIPPING"})
    assert "examples[3]" in result


def test_get_examples_excludes_flags():
    result = get_examples.invoke({"n": 1})
    header = result.split("\n")[0]
    assert "flags" not in header


def test_get_examples_filtered():
    result = get_examples.invoke({"n": 2, "intent": "get_refund"})
    assert "get_refund" in result


# --- Tool 6: search_instructions ---


def test_search_money_back():
    result = search_instructions.invoke({"query": "money back"})
    assert "search_results" in result


def test_search_no_results():
    result = search_instructions.invoke({"query": "xyznonexistent123"})
    assert "No instructions" in result


# --- Tool 7: summarize_responses ---


@pytest.mark.slow
def test_summarize_responses():
    result = summarize_responses.invoke(
        {"category": "FEEDBACK", "n_sample": 5}
    )
    assert len(result) > 50


@pytest.mark.slow
def test_summarize_responses_invalid_category():
    result = summarize_responses.invoke({"category": "INVALID"})
    assert "not found" in result.lower()


# --- Tools 8-9: remember_fact + recall_profile ---


def test_remember_and_recall(tmp_path, monkeypatch):
    """Test memory tools with a temporary profile directory."""
    import src.config
    import src.memory

    monkeypatch.setattr(src.config, "PROFILES_DIR", tmp_path)
    monkeypatch.setattr(src.memory, "PROFILES_DIR", tmp_path)

    set_current_user_id("test_user")

    result = recall_profile.invoke({})
    assert "no profile" in result.lower()

    result = remember_fact.invoke({"fact": "User likes refund data"})
    assert "remembered" in result.lower()

    result = recall_profile.invoke({})
    assert "refund data" in result.lower()

    result = remember_fact.invoke({"fact": "User likes refund data"})
    assert "already" in result.lower()


# --- DatasetMetadata ---


def test_metadata_loaded():
    from src.data import metadata

    assert metadata.row_count == 26872
    assert metadata.num_categories == 11
    assert metadata.num_intents == 27
    assert len(metadata.categories) == 11
    assert len(metadata.intents) == 27


def test_metadata_system_prompt():
    from src.data import metadata

    prompt = metadata.to_system_prompt_context()
    assert "26,872" in prompt
    assert "11 categories" in prompt
    assert "ORDER" in prompt
    assert "get_refund" in prompt


def test_metadata_category_intent_map():
    from src.data import metadata

    assert "get_refund" in metadata.category_intent_map["REFUND"]
    assert "cancel_order" in metadata.category_intent_map["ORDER"]
    assert "cancel_order" not in metadata.category_intent_map["REFUND"]


# --- Dynamic tool exposure ---


def test_tool_exposure_structured():
    from src.tools import get_tools_for_query_type

    tools = get_tools_for_query_type("structured")
    names = [t.name for t in tools]
    assert "count_rows" in names
    assert "list_categories" in names
    assert "remember_fact" in names
    assert "summarize_responses" not in names


def test_tool_exposure_unstructured():
    from src.tools import get_tools_for_query_type

    tools = get_tools_for_query_type("unstructured")
    names = [t.name for t in tools]
    assert "summarize_responses" in names
    assert "count_rows" in names
    assert "remember_fact" in names


def test_tool_exposure_includes_memory():
    from src.tools import get_tools_for_query_type

    for qt in ("structured", "unstructured"):
        tools = get_tools_for_query_type(qt)
        names = [t.name for t in tools]
        assert "remember_fact" in names
        assert "recall_profile" in names


def test_tool_exposure_fallback():
    from src.tools import get_tools_for_query_type

    tools = get_tools_for_query_type("unknown_type")
    assert len(tools) == 9
