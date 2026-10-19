"""Tests for the read-only executor and the final-SQL rule."""

import sqlite3
from pathlib import Path

import pytest

from src.sql_agent.executor import (
    Attempt,
    format_result,
    run_query,
    select_final_sql,
)


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "d.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.executemany(
        "INSERT INTO t VALUES (?, ?)",
        [(i, "w" * 100 if i == 1 else str(i)) for i in range(1, 1501)],
    )
    conn.commit()
    conn.close()
    return path


def test_success_preview_and_truncation(db: Path) -> None:
    a = run_query(db, "SELECT id, v FROM t WHERE id <= 30")
    assert a.ok and a.columns == ["id", "v"] and a.row_count == 30 and not a.more_rows
    assert len(a.preview) == 20
    assert a.preview[0] == ["1", "w" * 60 + "..."]
    assert a.row_count_text == "30"


def test_more_than_1000_rows(db: Path) -> None:
    a = run_query(db, "SELECT id FROM t")
    assert a.row_count == 1001 and a.more_rows and a.row_count_text == ">1000"


def test_null_and_empty(db: Path) -> None:
    assert run_query(db, "SELECT NULL").preview == [["NULL"]]
    a = run_query(db, "SELECT id FROM t WHERE id < 0")
    assert a.ok and a.row_count == 0 and a.columns == ["id"] and a.preview == []


def test_errors(db: Path) -> None:
    assert "no such column" in (run_query(db, "SELECT nope FROM t").error or "")
    assert run_query(db, "").error == "no sql"
    assert run_query(db, "SELECT 1; SELECT 2").error is not None


def test_write_is_refused(db: Path) -> None:
    assert run_query(db, "DELETE FROM t").error is not None
    assert run_query(db, "SELECT COUNT(*) FROM t").preview == [["1500"]]


def test_timeout(db: Path) -> None:
    slow = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 100000000) "
        "SELECT COUNT(*) FROM c"
    )
    assert run_query(db, slow, timeout_s=0.2).error == "timeout"


def test_select_final_sql() -> None:
    ok1, ok2 = Attempt(sql="a", row_count=1), Attempt(sql="b", row_count=0)
    bad = Attempt(sql="c", error="boom")
    assert select_final_sql([ok1, ok2]) == "b"
    assert select_final_sql([ok1, bad]) == "a"
    assert select_final_sql([bad, ok1, bad]) == "a"
    assert select_final_sql([bad, Attempt(sql="d", error="x")]) == "d"
    assert select_final_sql([]) == ""


def test_format_result(db: Path) -> None:
    assert format_result(Attempt(sql="x", error="boom")) == "Execution error: boom"
    text = format_result(run_query(db, "SELECT id FROM t WHERE id <= 2"))
    assert text == "Columns: id\nRow count: 2\nRows:\n1\n2"
    assert "First 20 rows:" in format_result(run_query(db, "SELECT id FROM t WHERE id <= 30"))
    assert "Row count: 0" in format_result(run_query(db, "SELECT id FROM t WHERE id < 0"))


def test_blocked_attempts_say_so(db: Path) -> None:
    for sql in (
        "VACUUM INTO 'x.sqlite'",
        "PRAGMA database_list",
        "SELECT * FROM pragma_table_info('t')",
    ):
        a = run_query(db, sql)
        assert a.blocked and not a.ok and (a.error or "").startswith("blocked:")
        assert format_result(a).startswith("Execution error: blocked:")
    assert not run_query(db, "SELECT nope FROM t").blocked
    assert not run_query(db, "SELECT 1 WHERE").blocked


def test_replace_function_runs(db: Path) -> None:
    a = run_query(db, "SELECT REPLACE('abc', 'b', 'x')")
    assert a.ok and a.preview == [["axc"]]


def test_policy_reaches_the_guard(db: Path) -> None:
    from src.sql_guard import GuardPolicy

    a = run_query(db, "SELECT v FROM t", policy=GuardPolicy(denied_columns={("t", "v")}))
    assert a.blocked and "denied by policy" in (a.error or "")
