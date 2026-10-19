"""Layers 3 and 4 and the public entry point: guarded execution of one statement."""

import sqlite3
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.sql_guard.authorizer import Denial, make_authorizer
from src.sql_guard.lexical import check_sql
from src.sql_guard.policy import GuardPolicy

TIMEOUT_ERROR = "timeout"
BLOCKED_PREFIX = "blocked: "
PROGRESS_STEPS = 1000
# Substrings of the engine's messages when a SQLITE_LIMIT_* or the heap limit is hit.
_LIMIT_MESSAGES = (
    "too big",
    "too long",
    "too large",
    "too complex",
    "too many",
    "out of memory",
    "parser stack overflow",
)


class GuardResult(BaseModel):
    """Outcome of :func:`guarded_execute`. ``blocked`` is only for security or policy blocks."""

    columns: list[str] = []
    rows: list[tuple[Any, ...]] = []
    truncated: bool = False
    error: str | None = None
    blocked: bool = False
    block_reason: str | None = None
    layer: str | None = None  # "lexical", "authorizer", "limit" or "engine" when blocked
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def timed_out(self) -> bool:
        return self.error == TIMEOUT_ERROR


def open_guarded(
    db_path: Path | str,
    policy: GuardPolicy | None = None,
    denial: Denial | None = None,
    *,
    authorizer: bool = True,
) -> sqlite3.Connection:
    """Layer 3 (and the layer 2 hook): read-only connection with engine limits.

    ``authorizer=False`` leaves out the allowlist; it is for trusted code that needs
    PRAGMAs (schema introspection) and for tests that probe the layers one by one.
    """
    policy = policy or GuardPolicy()
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, isolation_level=None)
    try:
        if hasattr(conn, "enable_load_extension"):
            conn.enable_load_extension(False)
        conn.execute("PRAGMA query_only = ON")
        if policy.max_memory_bytes > 0:
            conn.execute(f"PRAGMA hard_heap_limit = {int(policy.max_memory_bytes)}")
        conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, policy.max_value_bytes)
        conn.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, policy.max_sql_length)
        conn.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 100)
        conn.setlimit(sqlite3.SQLITE_LIMIT_LIKE_PATTERN_LENGTH, 10_000)
        if authorizer:
            conn.set_authorizer(make_authorizer(policy, denial or Denial()))
    except BaseException:
        conn.close()
        raise
    return conn


def _blocked(reason: str, layer: str, started: float) -> GuardResult:
    return GuardResult(
        error=BLOCKED_PREFIX + reason,
        blocked=True,
        block_reason=reason,
        layer=layer,
        elapsed_s=time.monotonic() - started,
    )


def guarded_execute(
    db_path: Path | str,
    sql: str,
    policy: GuardPolicy | None = None,
    *,
    authorizer: bool = True,
    lexical: bool = True,
) -> GuardResult:
    """Run one model-written statement under all four layers. Never raises for SQL problems.

    ``authorizer`` and ``lexical`` exist for ablation studies; leave them on.
    """
    policy = policy or GuardPolicy()
    started = time.monotonic()
    if lexical:
        reason = check_sql(sql, policy.max_sql_length)
        if reason is not None:
            return _blocked(reason.rstrip("."), "lexical", started)
    denial = Denial()
    conn: sqlite3.Connection | None = None
    try:
        conn = open_guarded(db_path, policy, denial, authorizer=authorizer)
        deadline = started + policy.timeout_s
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, PROGRESS_STEPS)
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description or []]
        fetched = cursor.fetchmany(policy.max_rows + 1)
        return GuardResult(
            columns=columns,
            rows=[tuple(row) for row in fetched[: policy.max_rows]],
            truncated=len(fetched) > policy.max_rows,
            elapsed_s=time.monotonic() - started,
        )
    except sqlite3.Error as e:
        message = str(e)
        if denial.reason is not None:
            return _blocked(denial.reason, "authorizer", started)
        if isinstance(e, sqlite3.ProgrammingError) and "one statement" in message:
            return _blocked("Multiple statements are not allowed", "engine", started)
        if isinstance(e, sqlite3.OperationalError) and "interrupted" in message.lower():
            return GuardResult(error=TIMEOUT_ERROR, elapsed_s=time.monotonic() - started)
        if any(m in message.lower() for m in _LIMIT_MESSAGES):
            return _blocked(f"limit exceeded ({message})", "limit", started)
        return GuardResult(error=message, elapsed_s=time.monotonic() - started)
    finally:
        if conn is not None:
            conn.close()
