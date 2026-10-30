"""Tests for the agent v1 tool-selection runner, with a fake LLM client and agent."""

from typing import Any

import pytest

from evals.tools import run
from src.agent import prompts
from src.llm import BudgetExceeded, CallUsage, UsageTracker

QUERY = {
    "id": "sql_01",
    "category": "sql",
    "query": "How much did we sell in Q3 2024?",
    "required": ["sql_query"],
    "optional": [],
    "checks": [{"kind": "sql_number", "values": [214067.19]}],
}


class FakeClient:
    """Records one call per ``complete`` in the tracker, like the real clients."""

    def __init__(self, tracker: UsageTracker, text: str = "") -> None:
        self.tracker = tracker
        self.text = text

    def complete(self, system: str | None, user: str, *, tag: str, prefix: str | None = None):
        self.tracker.check()
        self.tracker.record(CallUsage(tag=tag, input_tokens=100, output_tokens=20, cost_usd=0.001))
        return type("Completion", (), {"text": self.text})()


def test_simple_prompt_is_swapped_in_and_restored() -> None:
    original = prompts.TOOL_SELECTION_PROMPT
    with run.tool_selection_prompt(run.SIMPLE_TOOL_SELECTION_PROMPT):
        rendered = prompts.build_tool_selection_prompt("q?")
        assert rendered.startswith("Tools:") and rendered.endswith("User question: q?")
    assert prompts.TOOL_SELECTION_PROMPT is original


def test_agent_query_is_scored_with_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = UsageTracker()
    client = FakeClient(tracker)

    def fake_agent(query: str) -> dict[str, Any]:
        client.complete(None, query, tag="reason")
        client.complete(None, query, tag="respond")
        return {"response": "Q3 sales were €214,067.19.", "tools_used": ["sql_query"]}

    monkeypatch.setattr(run, "run_agent", fake_agent)
    row = run.run_query(QUERY, "agent_v1", client, tracker, "r")
    assert (row["exact"], row["precision"], row["answer_correct"]) == (True, 1.0, True)
    assert row["n_llm_calls"] == 2 and row["cost_usd"] == pytest.approx(0.002)
    assert row["error"] is None


def test_no_tools_answers_in_one_call() -> None:
    tracker = UsageTracker()
    row = run.run_query(QUERY, "no_tools", FakeClient(tracker, "About €200,000."), tracker, "r")
    assert row["tools_executed"] == [] and row["recall"] == 0.0 and row["exact"] is False
    assert row["answer_correct"] is False and row["n_llm_calls"] == 1


def test_agent_failure_is_recorded_but_budget_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = UsageTracker(budget_usd=0.0)

    def broken(query: str) -> dict[str, Any]:
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(run, "run_agent", broken)
    row = run.run_query(QUERY, "agent_v1", FakeClient(tracker), tracker, "r")
    assert row["error"] == "RuntimeError: qdrant down" and row["exact"] is False
    with pytest.raises(BudgetExceeded):
        run.run_query(QUERY, "no_tools", FakeClient(tracker), tracker, "r")


def test_summary_has_paired_differences_against_agent_v1() -> None:
    rows = []
    for i in range(6):
        for system, exact, correct in (("agent_v1", True, i < 4), ("no_tools", False, i < 1)):
            rows.append(
                {"id": f"q{i}", "category": "sql", "system": system, "exact": exact,
                 "precision": 1.0 if exact else None, "recall": 1.0 if exact else 0.0,
                 "answer_correct": correct, "cost_usd": 0.001, "latency_s": 1.0, "n_llm_calls": 1}
            )  # fmt: skip
    out = run.summarize(rows, ["agent_v1", "no_tools"])
    assert out["systems"]["agent_v1"]["all"]["answer_correct"]["mean"] == pytest.approx(4 / 6)
    diff = out["paired_vs_agent_v1"]["no_tools"]["answer_correct"]
    assert diff["diff"] == pytest.approx(-0.5)
    assert out["paired_vs_agent_v1"]["no_tools"]["exact"]["diff"] == -1.0


def test_rescore_uses_stored_responses(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    run_dir = tmp_path / "r"
    run_dir.mkdir()
    set_path = tmp_path / "queries.jsonl"
    set_path.write_text(json.dumps({**QUERY, "split": "dev"}) + "\n")
    row = {
        **run.score(QUERY, "agent_v1", "Vendimos 214.067 € en Q3.", ["sql_query"]),
        "answer_correct": False,  # as scored before the fix
        "cost_usd": 0.001,
        "latency_s": 1.0,
        "n_llm_calls": 2,
    }
    (run_dir / "predictions.jsonl").write_text(json.dumps(row) + "\n")
    (run_dir / "summary.json").write_text("{}")
    monkeypatch.setattr(run, "SET", set_path)
    monkeypatch.setattr(run, "REPORTS", tmp_path)
    assert run.main(["--run-id", "r", "--rescore"]) == 0
    stored = json.loads((run_dir / "predictions.jsonl").read_text())
    assert stored["answer_correct"] is True
    assert "rescored" in json.loads((run_dir / "summary.json").read_text())
