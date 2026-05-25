"""
Cross-model quality scoring for agent responses.

Uses a different model (JUDGE_MODEL) than the agent to evaluate response
quality.  Scores on three dimensions: data_grounded, addresses_question,
conciseness.  Called as a post-processing step — the agent doesn't know
it's being scored.
"""

import json
import logging
import time
from dataclasses import dataclass

from src.config import JUDGE_MODEL, QUALITY_SCORE_THRESHOLD, get_llm

log = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Result of a quality evaluation."""

    data_grounded: int
    addresses_question: int
    conciseness: int
    overall: float
    issue: str
    judge_tokens: dict
    judge_duration_ms: int
    judge_model: str

    @property
    def passed(self) -> bool:
        return self.overall >= QUALITY_SCORE_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "data_grounded": self.data_grounded,
            "addresses_question": self.addresses_question,
            "conciseness": self.conciseness,
            "overall": self.overall,
            "issue": self.issue,
            "passed": self.passed,
            "judge_tokens": self.judge_tokens,
            "judge_duration_ms": self.judge_duration_ms,
            "judge_model": self.judge_model,
        }


JUDGE_PROMPT = """\
You are a quality evaluator for a customer service dataset analyst.

The analyst was asked a question, used tools, and provided a response. \
Evaluate the response quality.

User question: {user_query}
Tools used: {tool_details}
Analyst response: {agent_response}

Score 1-5 on each dimension:
- data_grounded: Does the answer cite specific data from tool results? \
(1=pure general knowledge, 5=all claims backed by tool data)
- addresses_question: Does it actually answer what was asked? \
(1=completely off-topic, 5=directly answers the question)
- conciseness: Is it appropriately brief? \
(1=extremely verbose/off-track, 5=concise and focused)

If any score is below 3, explain the issue briefly in the "issue" field.

Respond ONLY with JSON, no other text:
{{"data_grounded": N, "addresses_question": N, "conciseness": N, \
"issue": "brief note or empty string"}}"""


def score_response(
    user_query: str,
    agent_response: str,
    tool_calls: list[dict],
) -> QualityScore:
    """Evaluate response quality using the judge model.

    Args:
        user_query: The user's original question.
        agent_response: The agent's final text response.
        tool_calls: List of tool call dicts with 'name' and optional 'result'.

    Returns:
        QualityScore with dimension scores, token usage, and timing.
    """
    tool_lines = []
    for tc in tool_calls:
        name = tc.get("name", "?")
        result = str(tc.get("result", ""))[:150]
        tool_lines.append(f"  {name}: {result}")
    tool_details = "\n".join(tool_lines) or "  (no tools used)"

    prompt = JUDGE_PROMPT.format(
        user_query=user_query,
        agent_response=agent_response[:1000],
        tool_details=tool_details[:500],
    )

    judge_llm = get_llm(JUDGE_MODEL, temperature=0, max_tokens=150)

    start = time.time()
    response = judge_llm.invoke(prompt)
    duration_ms = int((time.time() - start) * 1000)

    usage = getattr(response, "usage_metadata", {}) or {}
    judge_tokens = {
        "prompt": usage.get("input_tokens", 0),
        "completion": usage.get("output_tokens", 0),
        "total": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
    }

    try:
        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        scores = json.loads(text)
    except (json.JSONDecodeError, IndexError):
        log.warning("Judge response parse error: %s", response.content[:200])
        scores = {
            "data_grounded": 0,
            "addresses_question": 0,
            "conciseness": 0,
            "issue": "judge parse error",
        }

    dg = scores.get("data_grounded", 0)
    aq = scores.get("addresses_question", 0)
    co = scores.get("conciseness", 0)

    return QualityScore(
        data_grounded=dg,
        addresses_question=aq,
        conciseness=co,
        overall=round((dg + aq + co) / 3, 1),
        issue=scores.get("issue", ""),
        judge_tokens=judge_tokens,
        judge_duration_ms=duration_ms,
        judge_model=JUDGE_MODEL,
    )


def build_retry_prompt(
    user_query: str, original_response: str, score: QualityScore
) -> str:
    """Build a prompt for retrying after a low quality score.

    Designed but not yet wired — will be used when AUTO_RETRY_ON_LOW_SCORE
    is implemented as an experimental feature.
    """
    feedback_parts = []
    if score.data_grounded < QUALITY_SCORE_THRESHOLD:
        feedback_parts.append(
            "Your answer wasn't grounded in tool data. "
            "Use tools to get specific numbers."
        )
    if score.addresses_question < QUALITY_SCORE_THRESHOLD:
        feedback_parts.append(
            f"Your answer didn't directly address: '{user_query}'"
        )
    if score.conciseness < QUALITY_SCORE_THRESHOLD:
        feedback_parts.append("Your answer was too verbose. Be concise.")

    feedback = " ".join(feedback_parts) or "Please try a different approach."
    return (
        f"Your previous answer was evaluated and found lacking. "
        f"Feedback: {feedback} Please try again."
    )
