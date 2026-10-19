"""Tests for schema introspection."""

import sqlite3
from pathlib import Path

import pytest

from src.sql_agent.db import quote_identifier, sql_literal
from src.sql_agent.introspect import (
    _key_lines,
    build_schema_context,
    is_text_affinity,
    schema_block,
    schema_context,
)


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "d.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE city (id INTEGER PRIMARY KEY, name TEXT, note VARCHAR(500), pop INTEGER);
        CREATE TABLE person (pid INTEGER, city_id INTEGER, label TEXT,
                             FOREIGN KEY (city_id) REFERENCES city(id));
        CREATE TABLE loose (a INTEGER, b TEXT, PRIMARY KEY (a));
        CREATE TABLE nokeys (x INT, cid INT);
        CREATE TABLE "we""ird" ("co""l" TEXT);
        """
    )
    conn.executemany(
        "INSERT INTO city VALUES (?, ?, ?, ?)",
        [
            (1, "Oslo", "x" * 41, 1),
            (2, "O'Brien", "short", 2),
            (3, "Rome", None, 3),
            (4, "Paris", "ok", 4),
            (5, "Oslo", "ok", 5),
        ],
    )
    conn.execute('INSERT INTO "we""ird" VALUES (\'q\')')
    conn.commit()
    conn.close()
    return path


def test_ddl_and_example_values(db: Path) -> None:
    ctx = build_schema_context(db)
    assert "CREATE TABLE city" in ctx
    assert "Example values" in ctx
    assert "\"city\".\"name\": 'Oslo', 'O''Brien', 'Rome'" in ctx
    assert "\"city\".\"note\": 'short', 'ok'" in ctx  # long and NULL values skipped
    assert '"city"."pop"' not in ctx  # integer affinity
    assert '"we""ird"."co""l": \'q\'' in ctx


def test_without_sample_values(db: Path) -> None:
    ctx = build_schema_context(db, sample_values=False)
    assert "Example values" not in ctx
    assert "'Oslo'" not in ctx


def test_declared_keys_are_not_repeated(db: Path) -> None:
    ctx = build_schema_context(db, sample_values=False)
    assert "Keys not declared" not in ctx  # the DDL already states them


def test_key_lines_added_when_ddl_text_lacks_them(db: Path) -> None:
    conn = sqlite3.connect(db)
    try:
        # pass a DDL string that hides the keys, as if it were abbreviated
        assert _key_lines(conn, "person", "CREATE TABLE person (pid INTEGER)") == [
            "person: foreign key city_id -> city.id"
        ]
        assert _key_lines(conn, "loose", "CREATE TABLE loose (a INTEGER)") == [
            "loose: primary key (a)"
        ]
        assert _key_lines(conn, "nokeys", "CREATE TABLE nokeys (x INT)") == []
    finally:
        conn.close()


def test_deterministic_and_cached(db: Path) -> None:
    assert build_schema_context(db) == build_schema_context(db)
    assert schema_context(db) is schema_context(str(db))
    assert schema_context(db, sample_values=False) != schema_context(db)


def test_read_only(db: Path) -> None:
    before = db.read_bytes()
    build_schema_context(db)
    assert db.read_bytes() == before


def test_column_error_is_skipped(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.sql_agent.introspect as mod

    def boom(conn: sqlite3.Connection, timeout_s: float) -> None:
        raise sqlite3.OperationalError("interrupted")

    monkeypatch.setattr(mod, "set_deadline", boom)
    ctx = build_schema_context(db)
    assert "CREATE TABLE city" in ctx
    assert "Example values" not in ctx


def test_helpers() -> None:
    assert quote_identifier('a"b') == '"a""b"'
    assert sql_literal("it's") == "'it''s'"
    assert sql_literal(None) == "NULL"
    assert sql_literal(3) == "3"
    assert sql_literal(b"\x01") == "X'01'"
    assert is_text_affinity("VARCHAR(10)") and is_text_affinity("text")
    assert not is_text_affinity("INTEGER") and not is_text_affinity("")
    assert schema_block("X") == "Database schema (SQLite):\n\nX\n\n"
