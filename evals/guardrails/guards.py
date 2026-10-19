"""The guards compared by the SQL adversarial eval, behind one ``run`` interface.

``legacy_v1`` reproduces the tool of agent v1 before A4; ``readonly_only`` is the executor
that agent v2 and the scorer used before A4; ``sql_guard`` is the new guard; ``no_lexical``
is an ablation of ``sql_guard`` without layer 1.

``legacy_v1`` and ``readonly_only`` had no timeout. The harness gives them the same
wall-clock deadline as the others so that the eval terminates; ``bounded_by_guard`` marks
which guards enforce a bound on their own.
"""

import re
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.sql_guard import GuardPolicy, guarded_execute

OK, BLOCKED, ERROR, TIMEOUT, TRUNCATED = "ok", "blocked", "error", "timeout", "truncated"
PROGRESS_STEPS = 1000
LEGACY_MAX_ROWS = 50
ROWS_CAP = 1000

# Verbatim from src/tools/sql_query.py at commit d8b4fde (before A4).
LEGACY_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


@dataclass
class Outcome:
    kind: str
    error: str | None = None
    layer: str | None = None
    rows: list[tuple[Any, ...]] | None = None
    elapsed_s: float = 0.0


def legacy_validate(query: str) -> str | None:
    """The v1 ``_validate_query``: keyword regex and a ``;`` count."""
    stripped = query.strip().rstrip(";")
    if not stripped:
        return "Empty query"
    if LEGACY_FORBIDDEN.search(stripped):
        return "Write operations are not allowed. Only SELECT queries are permitted."
    if stripped.count(";") > 0:
        return "Multiple statements are not allowed. Submit one query at a time."
    return None


def _deadline_handler(conn: sqlite3.Connection, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, PROGRESS_STEPS)


def _classify_sqlite_error(e: sqlite3.Error, started: float) -> Outcome:
    elapsed = time.monotonic() - started
    if isinstance(e, sqlite3.OperationalError) and "interrupted" in str(e).lower():
        return Outcome(TIMEOUT, "timeout", elapsed_s=elapsed)
    return Outcome(ERROR, str(e), elapsed_s=elapsed)


def run_legacy_v1(db: Path, sql: str, policy: GuardPolicy, timeout_s: float) -> Outcome:
    started = time.monotonic()
    problem = legacy_validate(sql)
    if problem:
        return Outcome(BLOCKED, problem, layer="regex", elapsed_s=time.monotonic() - started)
    conn = sqlite3.connect(str(db))  # normal read-write connection, as in v1
    try:
        _deadline_handler(conn, timeout_s)
        cursor = conn.execute(sql)
        rows = [tuple(r) for r in cursor.fetchmany(LEGACY_MAX_ROWS)]
        return Outcome(OK, rows=rows, elapsed_s=time.monotonic() - started)
    except sqlite3.Error as e:
        return _classify_sqlite_error(e, started)
    finally:
        conn.close()  # v1 never committed


def run_readonly_only(db: Path, sql: str, policy: GuardPolicy, timeout_s: float) -> Outcome:
    started = time.monotonic()
    conn = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        _deadline_handler(conn, timeout_s)
        fetched = conn.execute(sql).fetchmany(ROWS_CAP + 1)
        rows = [tuple(r) for r in fetched[:ROWS_CAP]]
        kind = TRUNCATED if len(fetched) > ROWS_CAP else OK
        return Outcome(kind, rows=rows, elapsed_s=time.monotonic() - started)
    except sqlite3.Error as e:
        return _classify_sqlite_error(e, started)
    finally:
        conn.close()


def _run_guard(
    db: Path, sql: str, policy: GuardPolicy, timeout_s: float, *, lexical: bool
) -> Outcome:
    result = guarded_execute(
        db,
        sql,
        policy.model_copy(update={"timeout_s": timeout_s, "max_rows": ROWS_CAP}),
        lexical=lexical,
    )
    if result.blocked:
        return Outcome(BLOCKED, result.error, layer=result.layer, elapsed_s=result.elapsed_s)
    if result.timed_out:
        return Outcome(TIMEOUT, result.error, elapsed_s=result.elapsed_s)
    if result.error is not None:
        return Outcome(ERROR, result.error, elapsed_s=result.elapsed_s)
    kind = TRUNCATED if result.truncated else OK
    return Outcome(kind, rows=result.rows, elapsed_s=result.elapsed_s)


def run_sql_guard(db: Path, sql: str, policy: GuardPolicy, timeout_s: float) -> Outcome:
    return _run_guard(db, sql, policy, timeout_s, lexical=True)


def run_no_lexical(db: Path, sql: str, policy: GuardPolicy, timeout_s: float) -> Outcome:
    return _run_guard(db, sql, policy, timeout_s, lexical=False)


GuardRunner = Callable[[Path, str, GuardPolicy, float], Outcome]
GUARDS: dict[str, GuardRunner] = {
    "legacy_v1": run_legacy_v1,
    "readonly_only": run_readonly_only,
    "sql_guard": run_sql_guard,
    "no_lexical": run_no_lexical,
}
# Policy exists only in the new guard; the others cannot enforce one.
BOUNDED_BY_GUARD = {
    "legacy_v1": False,
    "readonly_only": False,
    "sql_guard": True,
    "no_lexical": True,
}
