"""Layer 1: lexical checks."""

import pytest

from src.sql_guard import check_sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "select 1;",
        "  \n\t SELECT 1 ;  ",
        "-- note\nSELECT 1",
        "/* a */ SELECT /* b */ 1",
        "WITH t AS (SELECT 1) SELECT * FROM t",
        "VALUES (1), (2)",
        "SELECT 'a; DROP TABLE x' AS s",
        'SELECT "semi;colon" FROM t',
        "SELECT `a;b` FROM t",
        "SELECT [a;b] FROM t",
        "SELECT 1 -- ; DELETE FROM t",
        "SELECT 1 /* ; DELETE FROM t */",
        "SELECT 'it''s; fine'",
        "SELECT replace('a', 'a', 'b')",
        "SELECT 1; -- trailing comment",
        "SELECT 1;/* trailing */ ",
    ],
)
def test_benign_passes(sql: str) -> None:
    assert check_sql(sql) is None


@pytest.mark.parametrize(
    ("sql", "fragment"),
    [
        ("", "Empty"),
        ("   \n ", "Empty"),
        ("-- only a comment", "Empty"),
        ("/* only */", "Empty"),
        ("INSERT INTO t VALUES (1)", "Write operations"),
        ("/* x */ delete from t", "Write operations"),
        ("-- c\n  DrOp TABLE t", "Write operations"),
        ("REPLACE INTO t VALUES (1)", "Write operations"),
        ("ATTACH 'x.db' AS x", "ATTACH"),
        ("VACUUM INTO 'x.db'", "VACUUM"),
        ("PRAGMA database_list", "PRAGMA"),
        ("BEGIN", "BEGIN"),
        ("EXPLAIN SELECT 1", "EXPLAIN"),
        ("(SELECT 1)", "statement starts with"),
        ("SELECT 1; SELECT 2", "Multiple"),
        ("SELECT 1; /* c */ DELETE FROM t", "Multiple"),
        ("SELECT 1 -- x\n; DROP TABLE t", "Multiple"),
        ("SELECT 'a'; SELECT 'b'", "Multiple"),
        ("SELECT 1;;", "Multiple"),
        ("SELECT 1\x00; SELECT 2", "NUL"),
        ("SELECTx 1", "SELECTX"),
    ],
)
def test_rejected_with_reason(sql: str, fragment: str) -> None:
    reason = check_sql(sql)
    assert reason is not None and fragment in reason


def test_length_precheck() -> None:
    assert check_sql("SELECT 1" + " " * 200, max_length=100) is not None
    assert check_sql("SELECT 1", max_length=100) is None


def test_quote_state_is_not_confused_by_escapes() -> None:
    # '' inside a string does not end it, so the ; stays inside the literal
    assert check_sql("SELECT 'a''; SELECT 2'") is None
    # an unclosed comment swallows the rest, like in SQLite
    assert check_sql("SELECT 1 /* ; SELECT 2") is None
