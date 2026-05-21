"""
Tool definitions for the Customer Service Data Analyst Agent.

9 tools with Pydantic input schemas, TOON-formatted multi-record outputs,
and configurable summarization model.
"""

import contextvars
from functools import lru_cache
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.config import get_llm, get_summarizer_model
from src.data import CATEGORIES, CATEGORY_INTENT_MAP, INTENTS, dataset
from src.memory import add_fact, get_facts
from src.toon import to_toon

# --- User ID injection (thread-safe + async-safe) ---

_current_user_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_user_id", default="default"
)


def set_current_user_id(user_id: str) -> None:
    """Called by the agent graph before invoking tools."""
    _current_user_id.set(user_id)


def get_current_user_id() -> str:
    return _current_user_id.get()


# --- Helpers ---


def _validate_category(category: str) -> str | None:
    """Return an error message if category is invalid, else None."""
    if category.upper() not in [c.upper() for c in CATEGORIES]:
        return (
            f"Category '{category}' not found. "
            f"Valid categories: {', '.join(CATEGORIES)}"
        )
    return None


def _validate_intent(intent: str) -> str | None:
    """Return an error message if intent is invalid, else None."""
    if intent.lower() not in [i.lower() for i in INTENTS]:
        return (
            f"Intent '{intent}' not found. "
            f"Valid intents: {', '.join(INTENTS)}"
        )
    return None


def _normalize_category(category: str) -> str:
    """Match user input to actual category name (case-insensitive)."""
    for c in CATEGORIES:
        if c.upper() == category.upper():
            return c
    return category


def _normalize_intent(intent: str) -> str:
    """Match user input to actual intent name (case-insensitive)."""
    for i in INTENTS:
        if i.lower() == intent.lower():
            return i
    return intent


# --- Tool 1: list_categories ---


@tool
@lru_cache(maxsize=1)
def list_categories() -> str:
    """List all unique categories in the Bitext customer service dataset.

    Returns a comma-separated list of all 11 categories.
    Use this when the user asks what categories or topics exist.
    """
    return ", ".join(CATEGORIES)


# --- Tool 2: list_intents ---


class ListIntentsInput(BaseModel):
    category: Optional[str] = Field(
        None,
        description="Filter intents by this category (e.g., 'ORDER', 'REFUND'). "
        "If omitted, returns all 27 intents.",
    )


@tool(args_schema=ListIntentsInput)
def list_intents(category: Optional[str] = None) -> str:
    """List intents in the dataset, optionally filtered by category.

    Use this to discover what specific intents exist, or what intents
    belong to a particular category.
    """
    if category is None:
        return ", ".join(INTENTS)
    err = _validate_category(category)
    if err:
        return err
    cat = _normalize_category(category)
    return ", ".join(CATEGORY_INTENT_MAP[cat])


# --- Tool 3: count_rows ---


class CountRowsInput(BaseModel):
    category: Optional[str] = Field(
        None, description="Filter by category (e.g., 'REFUND')"
    )
    intent: Optional[str] = Field(
        None, description="Filter by intent (e.g., 'get_refund')"
    )


@tool(args_schema=CountRowsInput)
def count_rows(
    category: Optional[str] = None, intent: Optional[str] = None
) -> str:
    """Count the number of rows matching optional category and/or intent filters.

    Use this when the user asks 'how many' questions.
    Returns the count as a number. Can filter by category, intent, or both.
    """
    df = dataset
    label_parts = []

    if category is not None:
        err = _validate_category(category)
        if err:
            return err
        cat = _normalize_category(category)
        df = df[df["category"] == cat]
        label_parts.append(f"category='{cat}'")

    if intent is not None:
        err = _validate_intent(intent)
        if err:
            return err
        int_ = _normalize_intent(intent)
        df = df[df["intent"] == int_]
        label_parts.append(f"intent='{int_}'")

    label = " and ".join(label_parts) if label_parts else "no filters"
    return f"Found {len(df):,} rows matching {label}."


# --- Tool 4: get_distribution ---


class GetDistributionInput(BaseModel):
    group_by: str = Field(
        description="Column to group by: 'category' or 'intent'"
    )
    filter_category: Optional[str] = Field(
        None,
        description="If grouping by intent, optionally filter to this "
        "category first",
    )


@tool(args_schema=GetDistributionInput)
def get_distribution(
    group_by: str, filter_category: Optional[str] = None
) -> str:
    """Get the frequency distribution of categories or intents.

    Use this when the user asks about distribution, breakdown, or proportions.
    Returns counts sorted by frequency (descending), capped at top 15.
    """
    if group_by not in ("category", "intent"):
        return "group_by must be 'category' or 'intent'."

    df = dataset
    label = group_by

    if filter_category is not None:
        err = _validate_category(filter_category)
        if err:
            return err
        cat = _normalize_category(filter_category)
        df = df[df["category"] == cat]
        label = f"{group_by} in {cat}"

    counts = df[group_by].value_counts().head(15)
    lines = [f"Distribution of {label} (top {len(counts)}):"]
    for val, count in counts.items():
        lines.append(f"  {val}: {count:,}")
    return "\n".join(lines)


# --- Tool 5: get_examples ---


class GetExamplesInput(BaseModel):
    n: int = Field(
        5, description="Number of examples to return (1-20)", ge=1, le=20
    )
    category: Optional[str] = Field(None, description="Filter by category")
    intent: Optional[str] = Field(None, description="Filter by intent")


@tool(args_schema=GetExamplesInput)
def get_examples(
    n: int = 5,
    category: Optional[str] = None,
    intent: Optional[str] = None,
) -> str:
    """Get sample rows from the dataset.

    Returns N random examples with their instruction, intent, and response.
    Use this when the user asks to see examples or sample data.
    Output is in TOON format for efficiency.
    """
    df = dataset

    if category is not None:
        err = _validate_category(category)
        if err:
            return err
        df = df[df["category"] == _normalize_category(category)]

    if intent is not None:
        err = _validate_intent(intent)
        if err:
            return err
        df = df[df["intent"] == _normalize_intent(intent)]

    actual_n = min(n, len(df))
    if actual_n == 0:
        return "No rows match the given filters."

    sampled = df.sample(n=actual_n, random_state=42)
    records = sampled[["instruction", "intent", "response"]].to_dict("records")

    result = to_toon(records, "examples", ["instruction", "intent", "response"])
    if actual_n < n:
        result += f"\n(Only {actual_n} rows matched; {n} were requested.)"
    return result


# --- Tool 6: search_instructions ---


class SearchInstructionsInput(BaseModel):
    query: str = Field(
        description="Search term to find in customer instructions"
    )
    n: int = Field(
        5, description="Maximum number of results (1-20)", ge=1, le=20
    )


@tool(args_schema=SearchInstructionsInput)
def search_instructions(query: str, n: int = 5) -> str:
    """Search for customer instructions containing a keyword or phrase.

    Use this when the user describes a topic in their own words,
    e.g., 'people wanting their money back' or 'shipping problems'.
    Performs case-insensitive substring search.
    Output is in TOON format.
    """
    mask = dataset["instruction"].str.contains(query, case=False, na=False)
    matches = dataset[mask]

    if matches.empty:
        return f"No instructions found matching '{query}'."

    results = matches.head(n)
    records = results[["instruction", "intent", "category"]].to_dict("records")
    return to_toon(records, "search_results", ["instruction", "intent", "category"])


# --- Tool 7: summarize_responses ---


class SummarizeResponsesInput(BaseModel):
    category: Optional[str] = Field(None, description="Filter by category")
    intent: Optional[str] = Field(None, description="Filter by intent")
    n_sample: int = Field(
        15,
        description="Number of responses to sample for summarization (5-30)",
        ge=5,
        le=30,
    )


@tool(args_schema=SummarizeResponsesInput)
def summarize_responses(
    category: Optional[str] = None,
    intent: Optional[str] = None,
    n_sample: int = 15,
) -> str:
    """Summarize how customer service agents typically respond to a category or intent.

    Samples N responses from the dataset and uses an LLM to produce a summary.
    Use this for open-ended questions about response patterns or agent behavior.
    """
    df = dataset

    if category is not None:
        err = _validate_category(category)
        if err:
            return err
        df = df[df["category"] == _normalize_category(category)]

    if intent is not None:
        err = _validate_intent(intent)
        if err:
            return err
        df = df[df["intent"] == _normalize_intent(intent)]

    if df.empty:
        return "No rows match the given filters."

    sampled = df.sample(n=min(n_sample, len(df)), random_state=42)
    responses = sampled["response"].tolist()
    responses_text = "\n---\n".join(responses)

    prompt = (
        "Summarize the following customer service responses. "
        "What patterns, tone, and common approaches do you see?\n\n"
        + responses_text
    )

    llm = get_llm(get_summarizer_model(), max_tokens=500)
    result = llm.invoke(prompt)
    return result.content


# --- Tool 8: remember_fact ---


class RememberFactInput(BaseModel):
    fact: str = Field(
        description="A fact about the user to remember "
        "(e.g., 'User is interested in refund data')"
    )


@tool(args_schema=RememberFactInput)
def remember_fact(fact: str) -> str:
    """Save a fact about the user to their persistent profile.

    Use this when the user shares personal information, preferences, or interests.
    For example: their name, what topics they care about, or their role.
    """
    return add_fact(get_current_user_id(), fact)


# --- Tool 9: recall_profile ---


@tool
def recall_profile() -> str:
    """Retrieve everything stored in the user's profile.

    Use this when the user asks 'What do you know about me?' or
    'What do you remember?'
    """
    return get_facts(get_current_user_id())
