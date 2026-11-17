"""The answer step sees the query that ran and, past the fetch limit, the exact row count."""

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from evals.failures.respond_versions import RESPOND_SYSTEM_V1, VERSIONS
from src.llm import CallUsage, Completion, reset_client, set_client
from src.sql_agent.executor import MAX_ROWS, count_rows, run_query
from src.sql_agent.graph import run_sql_agent
from src.sql_agent.prompts import RESPOND_SYSTEM, respond_user

N = MAX_ROWS + 234


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "big.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (v INTEGER)")
    conn.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(N)])
    conn.commit()
    conn.close()
    return path


def test_count_rows(db: Path) -> None:
    assert count_rows(db, "SELECT v FROM t;") == N
    assert count_rows(db, "SELECT v FROM t WHERE v < 10") == 10
    assert count_rows(db, "WITH x AS (SELECT v FROM t) SELECT v FROM x LIMIT 7") == 7
    assert count_rows(db, "SELECT nope FROM t") is None
    assert count_rows(db, "  ") is None


def test_respond_user_has_sql_and_exact_count(db: Path) -> None:
    attempt = run_query(db, "SELECT v FROM t")
    assert attempt.more_rows and attempt.row_count_text == f">{MAX_ROWS}"
    user = respond_user("List v?", attempt, N)
    assert "Query that ran:\nSELECT v FROM t" in user
    assert f"Row count: {N}\n" in user and f">{MAX_ROWS}" not in user
    small = run_query(db, "SELECT v FROM t WHERE v < 3")
    assert "Row count: 3\n" in respond_user("List v?", small, 999)  # total only past the limit
    assert respond_user("Q?", None).endswith("No query result.")


def test_versions_rebuild_old_and_new_prompts(db: Path) -> None:
    attempt = run_query(db, "SELECT v FROM t")
    system, user = VERSIONS["v1"]("List v?", attempt, db, 30)
    assert system == RESPOND_SYSTEM_V1 and "Query that ran" not in user
    assert f"Row count: >{MAX_ROWS}" in user
    system, user = VERSIONS["v2"]("List v?", attempt, db, 30)
    assert system == RESPOND_SYSTEM and f"Row count: {N}\n" in user


class Recorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def complete(self, system: Any, user: str, *, tag: str, prefix: Any = None) -> Completion:
        self.calls.append({"tag": tag, "user": user})
        text = "```sql\nSELECT v FROM t\n```" if tag == "generate" else "ok"
        return Completion(text=text, usage=CallUsage(tag=tag))


def test_agent_respond_step_gets_exact_count(db: Path) -> None:
    rec = Recorder()
    token = set_client(rec)
    try:
        run_sql_agent("List every v.", "", db)
    finally:
        reset_client(token)
    respond = [c for c in rec.calls if c["tag"] == "respond"]
    assert len(respond) == 1
    assert f"Row count: {N}\n" in respond[0]["user"]
    assert "Query that ran:\nSELECT v FROM t" in respond[0]["user"]
