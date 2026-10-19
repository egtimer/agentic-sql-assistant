"""Read-only query execution with a timeout and a compact result preview."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.sql_guard import GuardPolicy, guarded_execute

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
    blocked: bool = False  # refused by the SQL guard (error starts with "blocked:")
    block_reason: str | None = None  # why the guard refused it (traces only)
    layer: str | None = None  # guard layer that refused it (traces only)
    elapsed_s: float = 0.0  # execution time (traces only)

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


def run_query(
    db_path: Path | str,
    sql: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    policy: GuardPolicy | None = None,
) -> Attempt:
    """Run ``sql`` under the SQL guard. Never raises for SQL problems: they land in ``error``.

    A statement the guard refuses gets an error starting with ``blocked:``.
    """
    if not sql.strip():
        return Attempt(sql=sql, error=NO_SQL_ERROR)
    policy = (policy or GuardPolicy()).model_copy(
        update={"timeout_s": timeout_s, "max_rows": MAX_ROWS}
    )
    result = guarded_execute(db_path, sql, policy)
    if result.error is not None:
        return Attempt(
            sql=sql,
            error=result.error,
            blocked=result.blocked,
            block_reason=result.block_reason,
            layer=result.layer,
            elapsed_s=result.elapsed_s,
        )
    return Attempt(
        sql=sql,
        elapsed_s=result.elapsed_s,
        columns=result.columns,
        row_count=len(result.rows) + (1 if result.truncated else 0),
        more_rows=result.truncated,
        preview=[[_cell(v) for v in row] for row in result.rows[:PREVIEW_ROWS]],
    )


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
