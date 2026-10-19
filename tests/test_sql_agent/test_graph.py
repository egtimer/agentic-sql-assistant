"""Tests for the SQL agent graph, routing, check parsing and prompt-prefix stability."""

import sqlite3
from pathlib import Path

import pytest

from src.llm import CallUsage, Completion, reset_client, set_client
from src.sql_agent.executor import Attempt
from src.sql_agent.graph import build_graph, run_sql_agent
from src.sql_agent.nodes import parse_check, route_after_check, route_after_execute
from src.sql_agent.state import SqlAgentState, Verdict


class ScriptedLLM:
    """Answers per tag from queues and records every call."""

    def __init__(self, **script: list[str]) -> None:
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[tuple[str | None, str, str, str | None]] = []

    def complete(
        self, system: str | None, user: str, *, tag: str, prefix: str | None = None
    ) -> Completion:
        self.calls.append((system, user, tag, prefix))
        queue = self.script.get(tag, [])
        text = queue.pop(0) if len(queue) > 1 else (queue[0] if queue else "")
        return Completion(text=text, usage=CallUsage(tag=tag))

    @property
    def tags(self) -> list[str]:
        return [c[2] for c in self.calls]


def sql(q: str) -> str:
    return f"```sql\n{q}\n```"


GOOD = "SELECT name FROM city WHERE pop > 5"
BAD_COL = "SELECT nope FROM city"
EMPTY = "SELECT name FROM city WHERE name = 'zzz'"


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "d.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE city (id INTEGER PRIMARY KEY, name TEXT, pop INTEGER)")
    conn.executemany("INSERT INTO city VALUES (?, ?, ?)", [(1, "Oslo", 5), (2, "Rome", 9)])
    conn.commit()
    conn.close()
    return path


def run(llm: ScriptedLLM, db: Path, **kwargs: object):  # type: ignore[no-untyped-def]
    token = set_client(llm)
    try:
        return run_sql_agent("Which cities?", "pop > 1 is big", db, skip_respond=True, **kwargs)  # type: ignore[arg-type]
    finally:
        reset_client(token)


class TestParseCheck:
    def test_ok(self) -> None:
        assert parse_check("OK").kind == "ok"
        assert parse_check("  ok, looks right\n").kind == "ok"

    def test_fix_with_note(self) -> None:
        v = parse_check("Extra column.\n```sql\nSELECT 2;\n```", "SELECT 1")
        assert (v.kind, v.sql, v.note) == ("fix", "SELECT 2", "Extra column.")

    def test_same_sql_is_ok(self) -> None:
        assert parse_check(sql("select  1 ;"), "select 1").kind == "ok"

    def test_unparsed(self) -> None:
        v = parse_check("hmm, not sure")
        assert v.kind == "unparsed" and v.unparsed
        assert parse_check("").kind == "unparsed"
        assert parse_check("```\nSELECT 1\n```").kind == "unparsed"  # only ```sql counts


class TestRouting:
    def state(self, attempts: list[Attempt], **kw: object) -> SqlAgentState:
        return SqlAgentState(question="q", db_path="x", attempts=attempts, **kw)  # type: ignore[arg-type]

    ok = Attempt(sql="a", row_count=1)
    bad = Attempt(sql="a", error="boom")

    def test_success_with_check(self) -> None:
        assert route_after_execute(self.state([self.ok])) == "check"

    def test_success_without_check(self) -> None:
        assert route_after_execute(self.state([self.ok], use_check=False)) == "respond"

    def test_failure_always_checked(self) -> None:
        assert route_after_execute(self.state([self.bad], use_check=False)) == "check"

    def test_budget_exhausted(self) -> None:
        assert route_after_execute(self.state([self.bad], fixes_used=2)) == "respond"
        assert route_after_execute(self.state([self.ok], fixes_used=2)) == "respond"
        assert route_after_execute(self.state([self.bad], max_fixes=0)) == "respond"

    def test_after_check(self) -> None:
        fix = self.state([self.ok], verdicts=[Verdict(kind="fix", sql="b")])
        assert route_after_check(fix) == "execute"
        for kind in ("ok", "unparsed"):
            s = self.state([self.ok], verdicts=[Verdict(kind=kind)])  # type: ignore[arg-type]
            assert route_after_check(s) == "respond"


class TestGraph:
    def test_check_ok(self, db: Path) -> None:
        llm = ScriptedLLM(generate=[sql(GOOD)], check=["OK"])
        r = run(llm, db)
        assert llm.tags == ["generate", "check"]
        assert r.final_sql == GOOD and r.fixes_used == 0
        assert [v.kind for v in r.verdicts] == ["ok"]
        assert len(r.attempts) == 1 and r.attempts[0].preview == [["Rome"]]

    def test_check_fixes_then_ok(self, db: Path) -> None:
        llm = ScriptedLLM(generate=[sql(EMPTY)], check=[f"Wrong value.\n{sql(GOOD)}", "OK"])
        r = run(llm, db)
        assert llm.tags == ["generate", "check", "check"]
        assert r.final_sql == GOOD and r.fixes_used == 1
        assert [a.row_count for a in r.attempts] == [0, 1]
        assert "The result is empty." in llm.calls[1][1]
        assert "The result has 1 column." in llm.calls[1][1]

    def test_fix_bound(self, db: Path) -> None:
        llm = ScriptedLLM(generate=[sql(EMPTY)], check=[sql(BAD_COL), sql(GOOD), sql(EMPTY)])
        r = run(llm, db, max_fixes=2)
        assert llm.tags == ["generate", "check", "check"]  # at most 1 + max_fixes calls
        assert r.fixes_used == 2
        assert r.final_sql == GOOD  # last successful attempt

    def test_error_recovery_and_fallback(self, db: Path) -> None:
        llm = ScriptedLLM(generate=[sql(BAD_COL)], check=[sql(GOOD)])
        r = run(llm, db, max_fixes=1)
        assert llm.tags == ["generate", "check"]
        assert r.final_sql == GOOD and r.attempts[0].error is not None
        assert "Execution error" in llm.calls[1][1]

    def test_last_attempt_fails_falls_back_to_last_success(self, db: Path) -> None:
        llm = ScriptedLLM(generate=[sql(GOOD)], check=[sql(BAD_COL)])
        r = run(llm, db, max_fixes=1)
        assert r.attempts[-1].error is not None
        assert r.final_sql == GOOD

    def test_no_success_uses_last_sql(self, db: Path) -> None:
        llm = ScriptedLLM(generate=[sql(BAD_COL)], check=["OK"])
        r = run(llm, db)
        assert r.final_sql == BAD_COL

    def test_no_check_skips_successful_but_recovers_errors(self, db: Path) -> None:
        llm = ScriptedLLM(generate=[sql(GOOD)], check=["OK"])
        r = run(llm, db, use_check=False)
        assert llm.tags == ["generate"] and r.verdicts == []
        llm = ScriptedLLM(generate=[sql(BAD_COL)], check=[sql(GOOD)])
        r = run(llm, db, use_check=False)
        assert llm.tags == ["generate", "check"] and r.final_sql == GOOD

    def test_unparsed_is_recorded_and_ends(self, db: Path) -> None:
        llm = ScriptedLLM(generate=[sql(GOOD)], check=["I cannot tell"])
        r = run(llm, db)
        assert r.n_unparsed == 1 and r.final_sql == GOOD and llm.tags == ["generate", "check"]

    def test_no_sql_generated(self, db: Path) -> None:
        llm = ScriptedLLM(generate=["sorry"], check=["OK"])
        r = run(llm, db)
        assert r.attempts[0].error == "no sql" and r.final_sql == ""

    def test_respond_call(self, db: Path) -> None:
        llm = ScriptedLLM(generate=[sql(GOOD)], check=["OK"], respond=["Rome."])
        token = set_client(llm)
        try:
            r = run_sql_agent("Which cities?", "", db)
        finally:
            reset_client(token)
        assert llm.tags == ["generate", "check", "respond"] and r.response == "Rome."
        assert "Rome" in llm.calls[-1][1]

    def test_skip_respond_makes_no_call(self, db: Path) -> None:
        llm = ScriptedLLM(generate=[sql(GOOD)], check=["OK"])
        assert run(llm, db).response == "" and "respond" not in llm.tags

    def test_graph_is_cached_and_has_nodes(self) -> None:
        assert build_graph() is build_graph()
        assert {"introspect", "generate", "execute", "check", "respond"} <= set(
            build_graph().get_graph().nodes
        )


class TestPromptPrefix:
    def test_prefix_is_identical_across_questions_and_nodes(self, db: Path) -> None:
        llm = ScriptedLLM(generate=[sql(EMPTY)], check=[sql(GOOD), "OK"], respond=["x"])
        token = set_client(llm)
        try:
            for q, e in [("Which cities?", "a"), ("How many people?", "b")]:
                run_sql_agent(q, e, db)
        finally:
            reset_client(token)
        by_tag: dict[str, set[str | None]] = {}
        for _, _, tag, prefix in llm.calls:
            by_tag.setdefault(tag, set()).add(prefix)
        assert set(by_tag) == {"generate", "check", "respond"}
        assert len(by_tag["generate"]) == 1 and len(by_tag["check"]) == 1
        assert by_tag["generate"] == by_tag["check"]
        (prefix,) = by_tag["generate"]
        assert prefix is not None and "CREATE TABLE city" in prefix and "Which" not in prefix
        assert by_tag["respond"] == {None}

    def test_question_is_in_user_part(self, db: Path) -> None:
        llm = ScriptedLLM(generate=[sql(GOOD)], check=["OK"])
        run(llm, db)
        gen = llm.calls[0]
        assert "Question: Which cities?" in gen[1] and "pop > 1 is big" in gen[1]
        assert gen[1].rstrip().endswith("Which cities?")


def test_blocked_sql_reaches_check_and_can_be_rephrased(db: Path) -> None:
    llm = ScriptedLLM(
        generate=[sql("VACUUM INTO 'stolen.sqlite'")],
        check=[f"Not allowed, rephrase.\n{sql(GOOD)}"],
    )
    out = run(llm, db)
    assert out.attempts[0].blocked and (out.attempts[0].error or "").startswith("blocked:")
    assert "Execution error: blocked:" in llm.calls[1][1]
    assert out.final_sql == GOOD and out.attempts[-1].ok
    assert not (db.parent / "stolen.sqlite").exists()


def test_policy_hides_denied_tables_in_schema_and_blocks_reads(db: Path) -> None:
    from src.sql_guard import GuardPolicy

    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE secret (k TEXT)")
    conn.execute("INSERT INTO secret VALUES ('hidden-value')")
    conn.commit()
    conn.close()
    llm = ScriptedLLM(generate=[sql("SELECT k FROM secret")], check=["OK"])
    out = run(llm, db, policy=GuardPolicy(denied_tables={"secret"}))
    assert "secret" not in (llm.calls[0][3] or "") and "hidden-value" not in (llm.calls[0][3] or "")
    assert out.attempts[0].blocked and "denied by policy" in (out.attempts[0].error or "")
