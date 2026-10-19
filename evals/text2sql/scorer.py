"""Execution-accuracy (EX) scoring as in the BIRD paper.

Predicted and gold SQL run on the same read-only database; the result rows are compared
as sets, ignoring column names and row order. Errors, timeouts and missing SQL are wrong.
"""

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.sql_guard import GuardPolicy, guarded_execute

NO_SQL_ERROR = "no sql"
MAX_RESULT_ROWS = 1_000_000  # the scorer compares whole results; beyond this it is an error
RESULT_TOO_LARGE = f"result has more than {MAX_RESULT_ROWS} rows"


@dataclass
class ExecResult:
    """Rows of a query, or the error that prevented getting them."""

    rows: list[tuple[Any, ...]] | None
    error: str | None


def execute(db_path: Path | str, sql: str, timeout_s: float = 30) -> ExecResult:
    """Run ``sql`` under the SQL guard (read-only, wall-clock timeout).

    Predicted and gold SQL both go through it; a statement the guard blocks is an error
    starting with ``blocked:``.
    """
    policy = GuardPolicy(timeout_s=timeout_s, max_rows=MAX_RESULT_ROWS)
    result = guarded_execute(db_path, sql, policy)
    if result.error is not None:
        return ExecResult(rows=None, error=result.error)
    if result.truncated:
        return ExecResult(rows=None, error=RESULT_TOO_LARGE)
    return ExecResult(rows=result.rows, error=None)


def is_correct(
    pred_rows: list[tuple[Any, ...]] | None, gold_rows: list[tuple[Any, ...]] | None
) -> bool:
    """EX comparison: equal as sets of rows. Missing rows (error) are never correct."""
    if pred_rows is None or gold_rows is None:
        return False
    return set(pred_rows) == set(gold_rows)


def _encode(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, dict) and "__bytes__" in value:
        return bytes.fromhex(value["__bytes__"])
    return value


class GoldCache:
    """Gold results per question id: in memory, optionally persisted as JSON."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._rows: dict[int, list[tuple[Any, ...]]] = {}
        if path is not None and path.exists():
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            self._rows = {
                int(qid): [tuple(_decode(v) for v in row) for row in rows]
                for qid, rows in raw.items()
            }

    def get(
        self, question_id: int, db_path: Path | str, gold_sql: str, timeout_s: float = 30
    ) -> ExecResult:
        """Return the gold result, executing the gold SQL on first use."""
        with self._lock:
            cached = self._rows.get(question_id)
        if cached is not None:
            return ExecResult(rows=cached, error=None)
        result = execute(db_path, gold_sql, timeout_s)
        if result.rows is not None:
            with self._lock:
                self._rows[question_id] = result.rows
                self._save()
        return result

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw = {
            str(qid): [[_encode(v) for v in row] for row in rows]
            for qid, rows in self._rows.items()
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(raw), encoding="utf-8")
        tmp.replace(self.path)


@dataclass
class Score:
    """Outcome of scoring one prediction."""

    correct: bool
    error: str | None


def score_prediction(
    question_id: int,
    db_path: Path | str,
    pred_sql: str | None,
    gold_sql: str,
    gold_cache: GoldCache,
    timeout_s: float = 30,
) -> Score:
    """Score one predicted SQL against the gold SQL of ``question_id``."""
    if pred_sql is None or not pred_sql.strip():
        return Score(correct=False, error=NO_SQL_ERROR)
    pred = execute(db_path, pred_sql, timeout_s)
    if pred.rows is None:
        return Score(correct=False, error=pred.error)
    gold = gold_cache.get(question_id, db_path, gold_sql, timeout_s)
    if gold.rows is None:
        return Score(correct=False, error=f"gold sql failed: {gold.error}")
    return Score(correct=is_correct(pred.rows, gold.rows), error=None)
