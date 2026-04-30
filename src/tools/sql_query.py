"""Tool for executing read-only SQL queries against the SQLite database."""

import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


class SQLQueryResult(BaseModel):
    """Result of a SQL query execution."""

    query: str = Field(description="The executed SQL query")
    columns: list[str] = Field(description="Column names from the result set")
    rows: list[list[Any]] = Field(description="Row data as a list of lists")
    row_count: int = Field(description="Number of rows returned")
    success: bool = Field(description="Whether the query executed successfully")
    error: str | None = Field(default=None, description="Error message if query failed")


def _validate_query(query: str) -> str | None:
    """Check if a query is safe to execute. Returns error message if unsafe, None if ok."""
    stripped = query.strip().rstrip(";")
    if not stripped:
        return "Empty query"

    if FORBIDDEN_KEYWORDS.search(stripped):
        return "Write operations are not allowed. Only SELECT queries are permitted."

    if stripped.count(";") > 0:
        return "Multiple statements are not allowed. Submit one query at a time."

    return None


def sql_query(query: str, db_path: str | Path | None = None, max_rows: int = 50) -> SQLQueryResult:
    """Execute a read-only SQL query against the sales database.

    Args:
        query: SQL SELECT query to execute.
        db_path: Path to SQLite database. Defaults to the project sample database.
        max_rows: Maximum number of rows to return.

    Returns:
        SQLQueryResult with columns, rows, and metadata.
    """
    if db_path is None:
        db_path = Path(__file__).parent.parent / "db" / "sample.db"

    db_path = Path(db_path)
    if not db_path.exists():
        return SQLQueryResult(
            query=query,
            columns=[],
            rows=[],
            row_count=0,
            success=False,
            error=f"Database not found at {db_path}. Run 'make seed' to create it.",
        )

    validation_error = _validate_query(query)
    if validation_error:
        return SQLQueryResult(
            query=query,
            columns=[],
            rows=[],
            row_count=0,
            success=False,
            error=validation_error,
        )

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(query)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = [list(row) for row in cursor.fetchmany(max_rows)]

        logger.info("SQL query returned %d rows: %s", len(rows), query[:100])

        return SQLQueryResult(
            query=query,
            columns=columns,
            rows=rows,
            row_count=len(rows),
            success=True,
        )
    except sqlite3.Error as e:
        logger.warning("SQL query failed: %s — %s", query[:100], str(e))
        return SQLQueryResult(
            query=query,
            columns=[],
            rows=[],
            row_count=0,
            success=False,
            error=str(e),
        )
    finally:
        conn.close()
