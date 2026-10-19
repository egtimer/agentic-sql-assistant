"""Tests for the execution-accuracy scorer."""

import sqlite3
import time
from pathlib import Path

import pytest

from evals.text2sql.scorer import (
    GoldCache,
    execute,
    is_correct,
    score_prediction,
)

FOREVER = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "t.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, grp TEXT, blob BLOB)")
    conn.executemany(
        "INSERT INTO t VALUES (?, ?, ?, ?)",
        [(1, "a", "x", b"\x01"), (2, "b", "x", None), (3, "b", "y", None)],
    )
    conn.commit()
    conn.close()
    return path


class TestExecute:
    def test_rows(self, db: Path) -> None:
        result = execute(db, "SELECT id, name FROM t ORDER BY id")
        assert result.error is None
        assert result.rows == [(1, "a"), (2, "b"), (3, "b")]

    def test_error(self, db: Path) -> None:
        result = execute(db, "SELECT nope FROM t")
        assert result.rows is None
        assert result.error is not None and "nope" in result.error

    def test_read_only(self, db: Path) -> None:
        result = execute(db, "DELETE FROM t")
        assert result.rows is None
        assert result.error is not None
        assert len(execute(db, "SELECT * FROM t").rows or []) == 3

    def test_multiple_statements_error(self, db: Path) -> None:
        assert execute(db, "SELECT 1; SELECT 2").error is not None

    def test_timeout(self, db: Path) -> None:
        start = time.monotonic()
        result = execute(db, FOREVER, timeout_s=0.2)
        assert result.rows is None
        assert result.error == "timeout"
        assert time.monotonic() - start < 10


class TestIsCorrect:
    def test_order_ignored(self) -> None:
        assert is_correct([(1,), (2,)], [(2,), (1,)])

    def test_duplicates_collapse_as_set(self) -> None:
        assert is_correct([(1,), (1,), (2,)], [(1,), (2,)])

    def test_different_rows(self) -> None:
        assert not is_correct([(1,)], [(2,)])

    def test_missing_row(self) -> None:
        assert not is_correct([(1,)], [(1,), (2,)])

    def test_none_is_wrong(self) -> None:
        assert not is_correct(None, [(1,)])
        assert not is_correct([(1,)], None)

    def test_empty_equals_empty(self) -> None:
        assert is_correct([], [])


class TestScorePrediction:
    def test_correct_ignoring_column_names_and_order(self, db: Path) -> None:
        cache = GoldCache()
        score = score_prediction(
            1,
            db,
            "SELECT name AS foo FROM t ORDER BY id DESC",
            "SELECT name FROM t ORDER BY id",
            cache,
        )
        assert score.correct and score.error is None

    def test_wrong_rows(self, db: Path) -> None:
        score = score_prediction(1, db, "SELECT id FROM t", "SELECT name FROM t", GoldCache())
        assert not score.correct and score.error is None

    @pytest.mark.parametrize("sql", ["", "   ", None])
    def test_no_sql(self, db: Path, sql: str | None) -> None:
        score = score_prediction(1, db, sql, "SELECT 1", GoldCache())
        assert not score.correct and score.error == "no sql"

    def test_pred_error(self, db: Path) -> None:
        score = score_prediction(1, db, "SELECT bad FROM t", "SELECT 1", GoldCache())
        assert not score.correct and score.error and "bad" in score.error

    def test_pred_timeout(self, db: Path) -> None:
        score = score_prediction(1, db, FOREVER, "SELECT 1", GoldCache(), timeout_s=0.2)
        assert not score.correct and score.error == "timeout"

    def test_gold_error_is_reported(self, db: Path) -> None:
        score = score_prediction(1, db, "SELECT 1", "SELECT bad FROM t", GoldCache())
        assert not score.correct and score.error and score.error.startswith("gold sql failed")


class TestGoldCache:
    def test_gold_executed_once(self, db: Path) -> None:
        cache = GoldCache()
        first = cache.get(7, db, "SELECT id FROM t")
        # a different SQL for the same id is ignored: the cached result wins
        second = cache.get(7, db, "SELECT name FROM t")
        assert first.rows == second.rows == [(1,), (2,), (3,)]

    def test_errors_not_cached(self, db: Path) -> None:
        cache = GoldCache()
        assert cache.get(7, db, "SELECT bad FROM t").error is not None
        assert cache.get(7, db, "SELECT id FROM t").rows == [(1,), (2,), (3,)]

    def test_persisted_roundtrip(self, db: Path, tmp_path: Path) -> None:
        path = tmp_path / "res" / "gold.json"
        GoldCache(path).get(3, db, "SELECT id, name, blob FROM t ORDER BY id")
        reloaded = GoldCache(path)
        result = reloaded.get(3, tmp_path / "missing.sqlite", "SELECT 1")
        assert result.rows == [(1, "a", b"\x01"), (2, "b", None), (3, "b", None)]


def test_scorer_blocks_non_select_gold_or_prediction(db: Path) -> None:
    result = execute(db, "PRAGMA table_info(t)")
    assert result.rows is None and (result.error or "").startswith("blocked:")
    assert execute(db, "SELECT replace(name, 'a', 'z') FROM t").rows is not None
