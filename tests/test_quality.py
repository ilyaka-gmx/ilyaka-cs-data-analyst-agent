"""
Tests for src/quality.py — cross-model quality scoring.

Uses mocked LLM responses to avoid hitting the real API.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.quality import QualityScore, build_retry_prompt, score_response


def _mock_judge_response(scores: dict) -> MagicMock:
    """Build a mock AIMessage-like response from the judge."""
    resp = MagicMock()
    resp.content = json.dumps(scores)
    resp.usage_metadata = {
        "input_tokens": 200,
        "output_tokens": 30,
    }
    return resp


class TestQualityScore:
    def test_overall_pass(self):
        qs = QualityScore(
            data_grounded=5, addresses_question=4, conciseness=4,
            overall=4.3, issue="", judge_tokens={"prompt": 100, "completion": 20, "total": 120},
            judge_duration_ms=500, judge_model="test-model",
        )
        assert qs.passed is True

    def test_overall_fail(self):
        qs = QualityScore(
            data_grounded=2, addresses_question=2, conciseness=1,
            overall=1.7, issue="bad answer", judge_tokens={"prompt": 100, "completion": 20, "total": 120},
            judge_duration_ms=500, judge_model="test-model",
        )
        assert qs.passed is False

    def test_to_dict(self):
        qs = QualityScore(
            data_grounded=4, addresses_question=5, conciseness=3,
            overall=4.0, issue="", judge_tokens={"prompt": 100, "completion": 20, "total": 120},
            judge_duration_ms=300, judge_model="test-model",
        )
        d = qs.to_dict()
        assert d["data_grounded"] == 4
        assert d["addresses_question"] == 5
        assert d["conciseness"] == 3
        assert d["overall"] == 4.0
        assert d["passed"] is True
        assert d["judge_model"] == "test-model"
        assert d["judge_duration_ms"] == 300

    def test_threshold_boundary(self):
        qs = QualityScore(
            data_grounded=3, addresses_question=3, conciseness=3,
            overall=3.0, issue="", judge_tokens={}, judge_duration_ms=0,
            judge_model="test",
        )
        assert qs.passed is True

        qs2 = QualityScore(
            data_grounded=3, addresses_question=2, conciseness=3,
            overall=2.7, issue="low", judge_tokens={}, judge_duration_ms=0,
            judge_model="test",
        )
        assert qs2.passed is False


class TestScoreResponse:
    @patch("src.quality.get_llm")
    def test_score_response_happy_path(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_judge_response({
            "data_grounded": 5,
            "addresses_question": 4,
            "conciseness": 4,
            "issue": "",
        })
        mock_get_llm.return_value = mock_llm

        result = score_response(
            user_query="How many refund requests?",
            agent_response="There are 2,134 refund requests in the dataset.",
            tool_calls=[{"name": "count_rows", "result": "2134 rows match"}],
        )

        assert result.data_grounded == 5
        assert result.addresses_question == 4
        assert result.conciseness == 4
        assert result.overall == pytest.approx(4.3, abs=0.1)
        assert result.passed is True
        assert result.judge_tokens["prompt"] == 200
        assert result.judge_tokens["completion"] == 30

    @patch("src.quality.get_llm")
    def test_score_response_with_issue(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_judge_response({
            "data_grounded": 2,
            "addresses_question": 1,
            "conciseness": 3,
            "issue": "Answer not grounded in tool data",
        })
        mock_get_llm.return_value = mock_llm

        result = score_response(
            user_query="What categories exist?",
            agent_response="There are many categories.",
            tool_calls=[],
        )

        assert result.data_grounded == 2
        assert result.passed is False
        assert "grounded" in result.issue.lower()

    @patch("src.quality.get_llm")
    def test_score_response_parse_error(self, mock_get_llm):
        mock_llm = MagicMock()
        resp = MagicMock()
        resp.content = "invalid json here"
        resp.usage_metadata = {"input_tokens": 50, "output_tokens": 10}
        mock_llm.invoke.return_value = resp
        mock_get_llm.return_value = mock_llm

        result = score_response(
            user_query="test",
            agent_response="test response",
            tool_calls=[],
        )

        assert result.data_grounded == 0
        assert result.addresses_question == 0
        assert result.conciseness == 0
        assert "parse error" in result.issue

    @patch("src.quality.get_llm")
    def test_score_response_markdown_wrapped_json(self, mock_get_llm):
        mock_llm = MagicMock()
        resp = MagicMock()
        resp.content = '```json\n{"data_grounded":3,"addresses_question":4,"conciseness":5,"issue":""}\n```'
        resp.usage_metadata = {"input_tokens": 100, "output_tokens": 20}
        mock_llm.invoke.return_value = resp
        mock_get_llm.return_value = mock_llm

        result = score_response(
            user_query="test",
            agent_response="test response",
            tool_calls=[],
        )

        assert result.data_grounded == 3
        assert result.addresses_question == 4
        assert result.conciseness == 5


class TestBuildRetryPrompt:
    def test_low_grounded(self):
        qs = QualityScore(
            data_grounded=2, addresses_question=4, conciseness=4,
            overall=3.3, issue="", judge_tokens={}, judge_duration_ms=0,
            judge_model="test",
        )
        prompt = build_retry_prompt("How many rows?", "Many rows", qs)
        assert "grounded" in prompt.lower()
        assert "directly address" not in prompt.lower()

    def test_low_addresses(self):
        qs = QualityScore(
            data_grounded=4, addresses_question=2, conciseness=4,
            overall=3.3, issue="", judge_tokens={}, judge_duration_ms=0,
            judge_model="test",
        )
        prompt = build_retry_prompt("How many rows?", "Many rows", qs)
        assert "directly address" in prompt.lower()

    def test_low_conciseness(self):
        qs = QualityScore(
            data_grounded=4, addresses_question=4, conciseness=1,
            overall=3.0, issue="", judge_tokens={}, judge_duration_ms=0,
            judge_model="test",
        )
        prompt = build_retry_prompt("test", "long response", qs)
        assert "verbose" in prompt.lower()

    def test_all_pass(self):
        qs = QualityScore(
            data_grounded=5, addresses_question=5, conciseness=5,
            overall=5.0, issue="", judge_tokens={}, judge_duration_ms=0,
            judge_model="test",
        )
        prompt = build_retry_prompt("test", "good response", qs)
        assert "different approach" in prompt.lower()


class TestEstimateCost:
    def test_known_model(self):
        from src.config import estimate_cost
        cost = estimate_cost("meta-llama/Llama-3.3-70B-Instruct", 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.53, abs=0.01)

    def test_unknown_model_fallback(self):
        from src.config import estimate_cost
        cost = estimate_cost("unknown/model", 1_000_000, 0)
        assert cost == pytest.approx(0.20, abs=0.01)
