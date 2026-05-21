"""
Query router — classifies user queries before agent tool selection.

Uses a lightweight model (ROUTER_MODEL) to classify queries as:
- structured: concrete data questions (counts, lists, distributions, examples)
- unstructured: open-ended questions requiring summarization/analysis
- out_of_scope: questions unrelated to the customer service dataset
"""

import json
import logging
import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.config import ROUTER_MODEL, get_llm
from src.data import metadata

log = logging.getLogger(__name__)


class RouterOutput(BaseModel):
    """Structured output from the query router."""

    classification: Literal["structured", "unstructured", "out_of_scope"] = Field(
        description="The query type classification"
    )
    reasoning: str = Field(
        description="Brief explanation of why this classification was chosen"
    )


ROUTER_SYSTEM_PROMPT = f"""You are a query classifier for a customer service dataset analyst.

{metadata.to_system_prompt_context()}

Classify the user query into exactly one of:
- "structured": questions with concrete, data-driven answers — counts, lists, distributions, examples, filtering, searching. Examples: "How many refund requests?", "Show me 3 examples from SHIPPING", "What categories exist?"
- "unstructured": open-ended questions requiring summarization or qualitative analysis of the dataset. Examples: "Summarize the FEEDBACK category", "How do agents typically respond to complaints?"
- "out_of_scope": questions unrelated to the customer service dataset. Examples: "Who is the president of France?", "Write me a poem", "What's the best CRM software?"

Important rules:
- If the question is about the customer service data in ANY way, it is NOT out_of_scope.
- Questions asking to "show examples of people wanting X" are structured (they map to search/filter operations).
- Questions about how agents respond or patterns in the data are unstructured.
- Only classify as out_of_scope if the question has NO relation to the customer service dataset.

Respond with JSON: {{"classification": "...", "reasoning": "..."}}"""


def _parse_json_fallback(text: str) -> RouterOutput:
    """Extract a RouterOutput from raw LLM text when structured output fails.

    Handles Qwen3 thinking models that may wrap output in <think> tags.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    json_match = re.search(r"\{[^{}]*\}", cleaned, flags=re.DOTALL)
    if json_match:
        parsed = json.loads(json_match.group())
        return RouterOutput(**parsed)
    raise ValueError(f"No JSON object found in LLM response: {text[:200]}")


def classify_query(user_message: str) -> RouterOutput:
    """Classify a user query into structured, unstructured, or out_of_scope.

    Tries LangChain structured output first; falls back to manual JSON
    parsing if the model doesn't support it cleanly.

    Args:
        user_message: The user's query text.

    Returns:
        RouterOutput with classification and reasoning.
    """
    llm = get_llm(ROUTER_MODEL, temperature=0, max_tokens=150)
    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    try:
        structured_llm = llm.with_structured_output(RouterOutput)
        result = structured_llm.invoke(messages)
        if result is not None:
            return result
        log.warning("with_structured_output returned None; falling back to JSON parse")
    except Exception as exc:
        log.warning("with_structured_output failed (%s); falling back to JSON parse", exc)

    try:
        raw = llm.invoke(messages)
        return _parse_json_fallback(raw.content)
    except Exception as exc:
        log.error("Router fallback JSON parse also failed: %s", exc)
        return RouterOutput(
            classification="structured",
            reasoning=f"Router classification failed ({exc}); defaulting to structured",
        )
