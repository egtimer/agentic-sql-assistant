"""Tests for the single-shot baseline and the agent adapter (fake LLM clients)."""

import json
import sqlite3
from pathlib import Path

import pytest

from evals.text2sql.agent_system import build_agent_query, predict_agent, select_sql
from evals.text2sql.baseline import (
    build_baseline_parts,
    build_baseline_prompt,
    extract_sql,
    predict_baseline,
)
from evals.text2sql.data import Question
from src.llm import CallUsage, Completion


class FakeClient:
    """Returns scripted texts in order and records the prompts it received."""

    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.calls: list[tuple[str | None, str, str]] = []
        self.prefixes: list[str | None] = []

    def complete(
        self, system: str | None, user: str, *, tag: str, prefix: str | None = None
    ) -> Completion:
        self.prefixes.append(prefix)
        self.calls.append((system, (prefix or "") + user, tag))
        text = self.texts[min(len(self.calls) - 1, len(self.texts) - 1)]
        return Completion(text=text, usage=CallUsage(tag=tag))


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "d.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE person (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
    conn.executemany("INSERT INTO person VALUES (?, ?, ?)", [(1, "Ann", 30), (2, "Bob", 40)])
    conn.commit()
    conn.close()
    return path


def make_question(evidence: str = "adult means age >= 18") -> Question:
    return Question(
        question_id=1,
        db_id="d",
        question="How many adults are there?",
        evidence=evidence,
        SQL="SELECT COUNT(*) FROM person WHERE age >= 18",
        difficulty="simple",
    )


class TestExtractSql:
    def test_sql_fence(self) -> None:
        assert extract_sql("Here:\n```sql\nSELECT 1;\n```\nDone") == "SELECT 1"

    def test_generic_fence(self) -> None:
        assert extract_sql("```\nSELECT 2\n```") == "SELECT 2"

    def test_sql_fence_preferred_over_other_fences(self) -> None:
        text = "```text\nnot sql\n```\n```sql\nSELECT 3\n```"
        assert extract_sql(text) == "SELECT 3"

    def test_multiline_fence_kept(self) -> None:
        assert extract_sql("```sql\nSELECT a\nFROM t\nWHERE x = 1;\n```") == (
            "SELECT a\nFROM t\nWHERE x = 1"
        )

    def test_raw_select(self) -> None:
        assert extract_sql("The query is SELECT name FROM t WHERE a = 1;  ") == (
            "SELECT name FROM t WHERE a = 1"
        )

    def test_raw_with(self) -> None:
        text = "with x as (select 1) select * from x;"
        assert extract_sql(text) == "with x as (select 1) select * from x"

    def test_raw_stops_at_first_statement(self) -> None:
        assert extract_sql("SELECT 1; SELECT 2;") == "SELECT 1"

    def test_semicolon_inside_literal_kept(self) -> None:
        assert extract_sql("SELECT 'a;b' FROM t;") == "SELECT 'a;b' FROM t"

    def test_trailing_semicolons_and_whitespace(self) -> None:
        assert extract_sql("```sql\n  SELECT 1 ;; \n```") == "SELECT 1"

    def test_no_sql(self) -> None:
        assert extract_sql("I cannot help with that.") == ""
        assert extract_sql("") == ""

    def test_unterminated_fence_falls_back_to_raw(self) -> None:
        assert extract_sql("```sql\nSELECT 1") == "SELECT 1"


class TestBaselinePrompt:
    def test_layout_stable_first_question_last(self) -> None:
        system, user = build_baseline_prompt(make_question(), "CREATE TABLE person (id)")
        assert "```sql" in system
        assert user.index("CREATE TABLE person") < user.index("External Knowledge")
        assert user.index("External Knowledge: adult means age >= 18") < user.index("Question:")
        assert user.endswith("Question: How many adults are there?")

    def test_prefix_shared_between_questions_of_one_db(self) -> None:
        other = make_question("other hint")
        other.question = "Another question?"
        _, a = build_baseline_prompt(make_question(), "CREATE TABLE person (id)")
        _, b = build_baseline_prompt(other, "CREATE TABLE person (id)")
        assert a[: a.index("-- External")] == b[: b.index("-- External")]

    @pytest.mark.parametrize("evidence", ["adult means age >= 18", "", "  padded  "])
    def test_parts_are_byte_identical_to_the_original_prompt(self, evidence: str) -> None:
        question = make_question(evidence)
        schema = "CREATE TABLE person (id)"
        parts = [f"Database schema (SQLite):\n\n{schema}\n"]  # original construction
        if evidence.strip():
            parts.append(f"-- External Knowledge: {evidence.strip()}")
        parts.append(f"Question: {question.question.strip()}")
        original = "\n".join(parts)
        _, prefix, suffix = build_baseline_parts(question, schema)
        assert prefix + suffix == original
        assert build_baseline_prompt(question, schema)[1] == original
        assert schema in prefix and question.question not in prefix

    def test_empty_evidence_omitted(self) -> None:
        _, user = build_baseline_prompt(make_question(""), "CREATE TABLE person (id)")
        assert "External Knowledge" not in user


class TestPredictBaseline:
    def test_single_call(self, db: Path) -> None:
        client = FakeClient("```sql\nSELECT COUNT(*) FROM person WHERE age >= 18;\n```")
        sql, n_calls = predict_baseline(make_question(), client, database=db)
        assert sql == "SELECT COUNT(*) FROM person WHERE age >= 18"
        assert n_calls == 1
        assert client.prefixes[0] is not None and "CREATE TABLE person" in client.prefixes[0]
        system, user, tag = client.calls[0]
        assert tag == "baseline"
        assert "CREATE TABLE person" in user
        assert "adult means age >= 18" in user

    def test_no_sql_in_answer(self, db: Path) -> None:
        sql, n_calls = predict_baseline(make_question(), FakeClient("sorry"), database=db)
        assert (sql, n_calls) == ("", 1)


class TestSelectSql:
    def test_last_successful(self) -> None:
        results = [
            {"tool": "sql_query", "input": "A", "success": True},
            {"tool": "calculator", "input": "1+1", "success": True},
            {"tool": "sql_query", "input": "B", "success": True},
            {"tool": "sql_query", "input": "C", "success": False},
        ]
        assert select_sql(results) == "B"

    def test_falls_back_to_last_failed(self) -> None:
        results = [
            {"tool": "sql_query", "input": "A", "success": False},
            {"tool": "sql_query", "input": "B", "success": False},
        ]
        assert select_sql(results) == "B"

    def test_no_sql_calls(self) -> None:
        assert select_sql([]) == ""
        assert select_sql([{"tool": "calculator", "input": "1", "success": True}]) == ""


class TestPredictAgent:
    def test_query_includes_evidence(self) -> None:
        assert build_agent_query(make_question()) == (
            "How many adults are there?\nExternal knowledge: adult means age >= 18"
        )
        assert build_agent_query(make_question("")) == "How many adults are there?"

    def test_runs_agent_on_given_db(self, db: Path) -> None:
        plan = json.dumps(
            [
                {
                    "tool": "sql_query",
                    "input": "SELECT COUNT(*) FROM person WHERE age >= 18",
                    "reason": "count",
                }
            ]
        )
        client = FakeClient(plan, "There are 2 adults.")
        sql, n_calls = predict_agent(make_question(), client, database=db)
        assert sql == "SELECT COUNT(*) FROM person WHERE age >= 18"
        assert n_calls == 2  # reason + respond
        system, user, tag = client.calls[0]
        assert tag == "reason"
        assert "CREATE TABLE person" in user
        assert user.endswith("adult means age >= 18")
        assert "doc_search" not in user

    def test_failed_sql_still_reported(self, db: Path) -> None:
        plan = json.dumps([{"tool": "sql_query", "input": "SELECT bad FROM person", "reason": "r"}])
        sql, _ = predict_agent(make_question(), FakeClient(plan, "failed"), database=db)
        assert sql == "SELECT bad FROM person"

    def test_unparseable_plan_gives_empty_sql(self, db: Path) -> None:
        sql, n_calls = predict_agent(make_question(), FakeClient("no idea"), database=db)
        assert sql == ""
        assert n_calls == 1
