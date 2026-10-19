"""Read-only SQLite access with a wall-clock timeout."""

import sqlite3
import time
from pathlib import Path

from src.sql_guard import open_guarded

PROGRESS_STEPS = 1000


def connect_readonly(path: Path | str) -> sqlite3.Connection:
    """Read-only connection with the engine limits of the SQL guard, for trusted code.

    No authorizer: schema introspection runs fixed PRAGMAs. Model-written SQL must go
    through :func:`src.sql_guard.guarded_execute` instead.
    """
    return open_guarded(path, authorizer=False)


def set_deadline(conn: sqlite3.Connection, timeout_s: float) -> None:
    """Interrupt statements of ``conn`` that run past ``timeout_s`` seconds from now."""
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, PROGRESS_STEPS)


def quote_identifier(name: str) -> str:
    """Double-quoted SQL identifier with embedded quotes escaped."""
    return '"' + name.replace('"', '""') + '"'


def sql_literal(value: object) -> str:
    """SQL literal for a value read from the database."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, bytes):
        return "X'" + value.hex() + "'"
    return "'" + str(value).replace("'", "''") + "'"
