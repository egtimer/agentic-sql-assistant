"""Tests for the evaluation runner logic."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evals.run_eval import (
    EvalResult,
    EvalSummary,
    build_mock_response,
    build_tool_plan_json,
    check_answer_contains,
    check_tool_match,
    compute_summary,
    format_results_table,
    load_dataset,
    save_results,
)


class TestCheckToolMatch:
    def test_exact_match(self) -> None:
        assert check_tool_match(["sql_query"], ["sql_query"]) is True

    def test_order_independent(self) -> None:
        assert check_tool_match(["sql_query", "calculator"], ["calculator", "sql_query"]) is True

    def test_mismatch(self) -> None:
        assert check_tool_match(["sql_query"], ["calculator"]) is False

    def test_missing_tool(self) -> None:
        assert check_tool_match(["sql_query", "calculator"], ["sql_query"]) is False

    def test_extra_tool(self) -> None:
        assert check_tool_match(["sql_query"], ["sql_query", "calculator"]) is False

    def test_empty_both(self) -> None:
        assert check_tool_match([], []) is True


class TestCheckAnswerContains:
    def test_all_keywords_found(self) -> None:
        matched, score = check_answer_contains("The return policy is 30 days", ["return", "policy"])
        assert score == 1.0
        assert matched == ["return", "policy"]

    def test_partial_keywords(self) -> None:
        matched, score = check_answer_contains("The return period is 30 days", ["return", "policy"])
        assert score == 0.5
        assert matched == ["return"]

    def test_no_keywords_found(self) -> None:
        matched, score = check_answer_contains("Hello world", ["return", "policy"])
        assert score == 0.0
        assert matched == []

    def test_case_insensitive(self) -> None:
        matched, score = check_answer_contains("RETURN POLICY info", ["return", "policy"])
        assert score == 1.0

    def test_empty_keywords(self) -> None:
        matched, score = check_answer_contains("Some text", [])
        assert score == 1.0
        assert matched == []


class TestBuildToolPlanJson:
    def test_single_sql(self) -> None:
        entry = {"query": "test", "expected_tools": ["sql_query"], "expected_answer_contains": []}
        result = json.loads(build_tool_plan_json(entry))
        assert len(result) == 1
        assert result[0]["tool"] == "sql_query"

    def test_multi_tool_with_placeholder(self) -> None:
        entry = {
            "query": "growth?",
            "expected_tools": ["sql_query", "calculator"],
            "expected_answer_contains": [],
        }
        result = json.loads(build_tool_plan_json(entry))
        assert len(result) == 2
        assert result[1]["input"] == "PLACEHOLDER"

    def test_standalone_calculator(self) -> None:
        entry = {
            "query": "What is 15% of 100?",
            "expected_tools": ["calculator"],
            "expected_answer_contains": [],
        }
        result = json.loads(build_tool_plan_json(entry))
        assert result[0]["tool"] == "calculator"
        assert result[0]["input"] != "PLACEHOLDER"


class TestBuildMockResponse:
    def test_includes_keywords(self) -> None:
        entry = {
            "expected_tools": ["sql_query"],
            "expected_answer_contains": ["sales", "Q3"],
        }
        response = build_mock_response("How much in Q3?", entry)
        assert "sales" in response.lower()

    def test_includes_percentage_keyword(self) -> None:
        entry = {
            "expected_tools": ["calculator"],
            "expected_answer_contains": ["%"],
        }
        response = build_mock_response("What percentage?", entry)
        assert "%" in response


class TestComputeSummary:
    def _make_result(
        self, tool_match: bool = True, answer_score: float = 1.0, category: str = "single-tool-sql"
    ) -> EvalResult:
        return EvalResult(
            query_id="test",
            query="test query",
            category=category,
            expected_tools=["sql_query"],
            actual_tools=["sql_query"] if tool_match else ["calculator"],
            tool_match=tool_match,
            expected_keywords=["test"],
            matched_keywords=["test"] if answer_score == 1.0 else [],
            answer_contains_score=answer_score,
            response="test response",
            duration_ms=10.0,
        )

    def test_perfect_scores(self) -> None:
        results = [self._make_result() for _ in range(3)]
        summary = compute_summary(results)
        assert summary.tool_selection_accuracy == 1.0
        assert summary.answer_contains_accuracy == 1.0
        assert summary.total == 3

    def test_mixed_scores(self) -> None:
        results = [
            self._make_result(tool_match=True, answer_score=1.0),
            self._make_result(tool_match=False, answer_score=0.5),
        ]
        summary = compute_summary(results)
        assert summary.tool_selection_accuracy == 0.5
        assert summary.answer_contains_accuracy == 0.75

    def test_empty_results(self) -> None:
        summary = compute_summary([])
        assert summary.total == 0

    def test_category_breakdown(self) -> None:
        results = [
            self._make_result(category="single-tool-sql"),
            self._make_result(category="multi-tool", tool_match=False),
        ]
        summary = compute_summary(results)
        assert "single-tool-sql" in summary.category_scores
        assert "multi-tool" in summary.category_scores
        assert summary.category_scores["single-tool-sql"]["tool_accuracy"] == 1.0
        assert summary.category_scores["multi-tool"]["tool_accuracy"] == 0.0


class TestFormatResultsTable:
    def test_table_contains_header(self) -> None:
        results = [
            EvalResult(
                query_id="sql_01",
                query="test",
                category="single-tool-sql",
                expected_tools=["sql_query"],
                actual_tools=["sql_query"],
                tool_match=True,
                expected_keywords=["test"],
                matched_keywords=["test"],
                answer_contains_score=1.0,
                response="test",
                duration_ms=5.0,
            )
        ]
        summary = compute_summary(results)
        table = format_results_table(results, summary)
        assert "EVALUATION RESULTS" in table
        assert "SUMMARY" in table
        assert "sql_01" in table
        assert "PASS" in table


class TestSaveResults:
    def test_saves_json(self, tmp_path: Path) -> None:
        results = [
            EvalResult(
                query_id="sql_01",
                query="test",
                category="single-tool-sql",
                expected_tools=["sql_query"],
                actual_tools=["sql_query"],
                tool_match=True,
                expected_keywords=["test"],
                matched_keywords=["test"],
                answer_contains_score=1.0,
                response="test",
                duration_ms=5.0,
            )
        ]
        summary = compute_summary(results)

        with patch("evals.run_eval.RESULTS_DIR", tmp_path / "results"):
            output_path = save_results(results, summary)

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["summary"]["total"] == 1
        assert len(data["results"]) == 1


class TestLoadDataset:
    def test_loads_golden_dataset(self) -> None:
        dataset = load_dataset()
        assert len(dataset) == 15
        assert all("id" in entry for entry in dataset)

    def test_loads_custom_path(self, tmp_path: Path) -> None:
        custom = tmp_path / "custom.json"
        custom.write_text('[{"id": "t1", "query": "test", "expected_tools": [], "expected_answer_contains": [], "category": "test"}]')
        dataset = load_dataset(custom)
        assert len(dataset) == 1
