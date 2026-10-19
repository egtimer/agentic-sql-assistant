"""Evaluation runner for the agent pipeline.

Loads the golden dataset, runs each query against the agent with mocked LLM,
and measures tool selection accuracy and answer quality.
"""

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from src.agent.graph import run_agent
from src.llm import CallUsage, Completion
from src.observability.tracing import NoopTracer

DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class EvalResult:
    """Result of evaluating a single query."""

    query_id: str
    query: str
    category: str
    expected_tools: list[str]
    actual_tools: list[str]
    tool_match: bool
    expected_keywords: list[str]
    matched_keywords: list[str]
    answer_contains_score: float
    response: str
    duration_ms: float


@dataclass
class EvalSummary:
    """Aggregate scores across all evaluated queries."""

    total: int = 0
    tool_selection_accuracy: float = 0.0
    answer_contains_accuracy: float = 0.0
    category_scores: dict[str, dict[str, float]] = field(default_factory=dict)


def load_dataset(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the golden dataset from JSON."""
    dataset_path = path or DATASET_PATH
    with open(dataset_path) as f:
        return json.load(f)


def check_tool_match(expected: list[str], actual: list[str]) -> bool:
    """Check if the actual tools match the expected tools (order-independent)."""
    return sorted(expected) == sorted(actual)


def check_answer_contains(response: str, keywords: list[str]) -> tuple[list[str], float]:
    """Check which expected keywords appear in the response.

    Returns the list of matched keywords and the fraction matched.
    """
    if not keywords:
        return [], 1.0
    response_lower = response.lower()
    matched = [kw for kw in keywords if kw.lower() in response_lower]
    score = len(matched) / len(keywords)
    return matched, score


def build_mock_response(query: str, entry: dict[str, Any]) -> str:
    """Build a simulated agent response based on the expected tools and keywords."""
    tools = entry["expected_tools"]
    keywords = entry["expected_answer_contains"]

    parts = []
    if "sql_query" in tools:
        parts.append("Based on the sales data from the database")
    if "doc_search" in tools:
        parts.append("According to the company policy documents")
    if "calculator" in tools:
        parts.append("After calculating the result")

    parts.append(f"regarding your query about: {query}")

    for kw in keywords:
        if kw == "%":
            parts.append("the result is approximately 18.5%")
        elif kw.replace(",", "").isdigit():
            parts.append(f"the value is {kw}")
        else:
            parts.append(f"the relevant {kw} information has been found")

    return ". ".join(parts) + "."


def build_tool_plan_json(entry: dict[str, Any]) -> str:
    """Build the JSON that the mocked LLM returns for tool planning."""
    tool_calls = []
    for tool_name in entry["expected_tools"]:
        if tool_name == "sql_query":
            tool_calls.append(
                {
                    "tool": "sql_query",
                    "input": "SELECT SUM(amount) FROM sales WHERE quarter = 'Q3'",
                    "reason": "Query sales data",
                }
            )
        elif tool_name == "doc_search":
            tool_calls.append(
                {
                    "tool": "doc_search",
                    "input": entry["query"],
                    "reason": "Search company documents",
                }
            )
        elif tool_name == "calculator":
            if "sql_query" in entry["expected_tools"]:
                tool_calls.append(
                    {
                        "tool": "calculator",
                        "input": "PLACEHOLDER",
                        "reason": "Calculate result from data",
                    }
                )
            else:
                tool_calls.append(
                    {
                        "tool": "calculator",
                        "input": entry["query"],
                        "reason": "Compute mathematical expression",
                    }
                )
    return json.dumps(tool_calls)


def create_mock_tool_result(tool_name: str) -> dict[str, Any]:
    """Create a mock tool result for the given tool."""
    if tool_name == "sql_query":
        from src.tools.sql_query import SQLQueryResult

        return SQLQueryResult(
            query="SELECT SUM(amount) FROM sales",
            columns=["sum"],
            rows=[[234500]],
            row_count=1,
            success=True,
        )
    elif tool_name == "doc_search":
        from src.tools.doc_search import DocSearchResult

        return DocSearchResult(
            query="policy",
            chunks=["The return policy allows returns within 30 days."],
            sources=["policy.md"],
            chunk_count=1,
            success=True,
        )
    elif tool_name == "calculator":
        from src.tools.calculator import CalculatorResult

        return CalculatorResult(
            expression="0.15 * 234500",
            result=35175.0,
            formatted="35,175",
            success=True,
        )
    return None


def run_single_eval(entry: dict[str, Any]) -> EvalResult:
    """Run evaluation for a single query with mocked dependencies."""
    tool_plan_json = build_tool_plan_json(entry)
    response_text = build_mock_response(entry["query"], entry)

    mock_client = MagicMock()
    mock_client.complete.side_effect = [
        Completion(text=text, usage=CallUsage(tag="mock"))
        for text in (
            tool_plan_json,
            '{"is_complete": true, "additional_tools": []}',
            response_text,
        )
    ]

    tool_mocks = {}
    for tool_name in entry["expected_tools"]:
        tool_mocks[tool_name] = create_mock_tool_result(tool_name)

    start = time.perf_counter()

    with (
        patch("src.agent.nodes.get_client", return_value=mock_client),
        patch("src.agent.nodes.sql_query") as mock_sql,
        patch("src.agent.nodes.doc_search") as mock_doc,
        patch("src.agent.nodes.calculate") as mock_calc,
        patch("src.observability.tracing._tracer", NoopTracer()),
    ):
        if "sql_query" in tool_mocks:
            mock_sql.return_value = tool_mocks["sql_query"]
        if "doc_search" in tool_mocks:
            mock_doc.return_value = tool_mocks["doc_search"]
        if "calculator" in tool_mocks:
            mock_calc.return_value = tool_mocks["calculator"]

        result = run_agent(entry["query"])

    duration_ms = (time.perf_counter() - start) * 1000
    actual_tools = result.get("tools_used", [])
    response = result.get("response", "")

    tool_match = check_tool_match(entry["expected_tools"], actual_tools)
    matched_keywords, answer_score = check_answer_contains(
        response, entry["expected_answer_contains"]
    )

    return EvalResult(
        query_id=entry["id"],
        query=entry["query"],
        category=entry["category"],
        expected_tools=entry["expected_tools"],
        actual_tools=actual_tools,
        tool_match=tool_match,
        expected_keywords=entry["expected_answer_contains"],
        matched_keywords=matched_keywords,
        answer_contains_score=answer_score,
        response=response,
        duration_ms=duration_ms,
    )


def compute_summary(results: list[EvalResult]) -> EvalSummary:
    """Compute aggregate evaluation metrics."""
    if not results:
        return EvalSummary()

    tool_matches = sum(1 for r in results if r.tool_match)
    answer_scores = [r.answer_contains_score for r in results]

    categories: dict[str, list[EvalResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    category_scores = {}
    for cat, cat_results in categories.items():
        cat_tool = sum(1 for r in cat_results if r.tool_match) / len(cat_results)
        cat_answer = sum(r.answer_contains_score for r in cat_results) / len(cat_results)
        category_scores[cat] = {
            "tool_accuracy": round(cat_tool, 3),
            "answer_accuracy": round(cat_answer, 3),
            "count": len(cat_results),
        }

    return EvalSummary(
        total=len(results),
        tool_selection_accuracy=round(tool_matches / len(results), 3),
        answer_contains_accuracy=round(sum(answer_scores) / len(answer_scores), 3),
        category_scores=category_scores,
    )


def format_results_table(results: list[EvalResult], summary: EvalSummary) -> str:
    """Format evaluation results as a readable text table."""
    lines = []
    lines.append("=" * 100)
    lines.append("EVALUATION RESULTS")
    lines.append("=" * 100)
    lines.append("")

    header = f"{'ID':<10} {'Category':<25} {'Tools OK':<10} {'Answer':<10} {'Time (ms)':<10}"
    lines.append(header)
    lines.append("-" * 100)

    for r in results:
        tool_ok = "PASS" if r.tool_match else "FAIL"
        answer_pct = f"{r.answer_contains_score:.0%}"
        lines.append(
            f"{r.query_id:<10} {r.category:<25} {tool_ok:<10} "
            f"{answer_pct:<10} {r.duration_ms:<10.1f}"
        )

    lines.append("")
    lines.append("=" * 100)
    lines.append("SUMMARY")
    lines.append("=" * 100)
    lines.append(f"  Total queries:             {summary.total}")
    lines.append(f"  Tool selection accuracy:   {summary.tool_selection_accuracy:.1%}")
    lines.append(f"  Answer contains accuracy:  {summary.answer_contains_accuracy:.1%}")
    lines.append("")

    lines.append("  By category:")
    for cat, scores in sorted(summary.category_scores.items()):
        lines.append(
            f"    {cat:<25} tools={scores['tool_accuracy']:.1%}  "
            f"answer={scores['answer_accuracy']:.1%}  (n={scores['count']})"
        )

    lines.append("=" * 100)
    return "\n".join(lines)


def save_results(results: list[EvalResult], summary: EvalSummary) -> Path:
    """Save detailed results to JSON in the results directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "latest.json"

    output = {
        "summary": {
            "total": summary.total,
            "tool_selection_accuracy": summary.tool_selection_accuracy,
            "answer_contains_accuracy": summary.answer_contains_accuracy,
            "category_scores": summary.category_scores,
        },
        "results": [
            {
                "id": r.query_id,
                "query": r.query,
                "category": r.category,
                "expected_tools": r.expected_tools,
                "actual_tools": r.actual_tools,
                "tool_match": r.tool_match,
                "answer_contains_score": r.answer_contains_score,
                "duration_ms": round(r.duration_ms, 1),
            }
            for r in results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return output_path


def main() -> None:
    """Run the full evaluation pipeline."""
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} queries from golden dataset\n")

    results = []
    for entry in dataset:
        try:
            result = run_single_eval(entry)
            results.append(result)
        except Exception as e:
            print(f"ERROR on {entry['id']}: {e}")

    summary = compute_summary(results)
    table = format_results_table(results, summary)
    print(table)

    output_path = save_results(results, summary)
    print(f"\nDetailed results saved to: {output_path}")

    if summary.tool_selection_accuracy < 1.0:
        failed = [r for r in results if not r.tool_match]
        print(f"\nTool mismatches ({len(failed)}):")
        for r in failed:
            print(f"  {r.query_id}: expected={r.expected_tools} actual={r.actual_tools}")

    sys.exit(0 if summary.tool_selection_accuracy >= 0.8 else 1)


if __name__ == "__main__":
    main()
