"""Read-only query execution with a timeout and a compact result preview."""

import sqlite3
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.sql_agent.db import connect_readonly, set_deadline

TIMEOUT_ERROR = "timeout"
NO_SQL_ERROR = "no sql"
DEFAULT_TIMEOUT_S = 30.0
MAX_ROWS = 1000  # fetch MAX_ROWS + 1 rows to know whether there are more
PREVIEW_ROWS = 20
MAX_CELL_CHARS = 60


class Attempt(BaseModel):
    """One executed candidate query."""

    sql: str
    columns: list[str] = []
    row_count: int | None = None  # capped at MAX_ROWS + 1; see ``more_rows``
    more_rows: bool = False
    preview: list[list[str]] = []
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def row_count_text(self) -> str:
        if self.row_count is None:
            return "n/a"
        return f">{MAX_ROWS}" if self.more_rows else str(self.row_count)


def _cell(value: Any) -> str:
    text = "NULL" if value is None else repr(value) if isinstance(value, bytes) else str(value)
    return text if len(text) <= MAX_CELL_CHARS else text[:MAX_CELL_CHARS] + "..."


def run_query(db_path: Path | str, sql: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> Attempt:
    """Run ``sql`` read-only. Never raises for SQL problems: they land in ``Attempt.error``."""
    if not sql.strip():
        return Attempt(sql=sql, error=NO_SQL_ERROR)
    conn = None
    try:
        conn = connect_readonly(db_path)
        set_deadline(conn, timeout_s)
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description or []]
        rows = cursor.fetchmany(MAX_ROWS + 1)
        return Attempt(
            sql=sql,
            columns=columns,
            row_count=len(rows),
            more_rows=len(rows) > MAX_ROWS,
            preview=[[_cell(v) for v in row] for row in rows[:PREVIEW_ROWS]],
        )
    except sqlite3.OperationalError as e:
        message = TIMEOUT_ERROR if "interrupted" in str(e).lower() else str(e)
        return Attempt(sql=sql, error=message)
    except sqlite3.Error as e:
        return Attempt(sql=sql, error=str(e))
    finally:
        if conn is not None:
            conn.close()


def select_final_sql(attempts: list[Attempt]) -> str:
    """Last attempt that executed successfully; else the last attempt's SQL; else ""."""
    for attempt in reversed(attempts):
        if attempt.ok:
            return attempt.sql
    return attempts[-1].sql if attempts else ""


def format_result(attempt: Attempt) -> str:
    """Text rendering of an attempt's outcome for the prompts."""
    if attempt.error is not None:
        return f"Execution error: {attempt.error}"
    lines = [f"Columns: {', '.join(attempt.columns)}", f"Row count: {attempt.row_count_text}"]
    if attempt.preview:
        shown = len(attempt.preview)
        lines.append(f"First {shown} rows:" if shown < (attempt.row_count or 0) else "Rows:")
        lines.extend(" | ".join(row) for row in attempt.preview)
    return "\n".join(lines)
