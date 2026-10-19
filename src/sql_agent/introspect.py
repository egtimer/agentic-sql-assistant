"""Schema context for the SQL agent: DDL, keys and example values (deterministic, no LLM)."""

import sqlite3
from functools import lru_cache
from pathlib import Path

from src.sql_agent.db import connect_readonly, quote_identifier, set_deadline, sql_literal
from src.sql_guard import GuardPolicy

MAX_EXAMPLES = 3
MAX_VALUE_LENGTH = 40
SAMPLE_TIMEOUT_S = 2.0


def is_text_affinity(declared_type: str) -> bool:
    """SQLite affinity rule: a declared type containing CHAR, CLOB or TEXT has TEXT affinity."""
    t = declared_type.upper()
    return "CHAR" in t or "CLOB" in t or "TEXT" in t


def _tables(conn: sqlite3.Connection, policy: GuardPolicy) -> list[tuple[str, str]]:
    rows = conn.execute(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type = 'table' AND sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
    ).fetchall()
    return [
        (str(name), str(sql).strip())
        for name, sql in rows
        if str(name).lower() not in policy.denied_tables
    ]


def _key_lines(conn: sqlite3.Connection, table: str, ddl: str) -> list[str]:
    """Primary and foreign keys of ``table`` that its DDL does not already state."""
    upper = ddl.upper()
    quoted = quote_identifier(table)
    lines: list[str] = []
    info = conn.execute(f"PRAGMA table_info({quoted})").fetchall()
    pk = [str(r[1]) for r in sorted(info, key=lambda r: r[5]) if r[5] > 0]
    if pk and "PRIMARY KEY" not in upper:
        lines.append(f"{table}: primary key ({', '.join(pk)})")
    if "REFERENCES" not in upper:
        for row in conn.execute(f"PRAGMA foreign_key_list({quoted})").fetchall():
            target = row[4] if row[4] is not None else "(primary key)"
            lines.append(f"{table}: foreign key {row[3]} -> {row[2]}.{target}")
    return lines


def _example_values(conn: sqlite3.Connection, table: str, policy: GuardPolicy) -> list[str]:
    """``"table"."column": 'a', 'b'`` lines for the TEXT columns of ``table``."""
    lines: list[str] = []
    for row in conn.execute(f"PRAGMA table_info({quote_identifier(table)})").fetchall():
        column, declared = str(row[1]), str(row[2] or "")
        if (
            not is_text_affinity(declared)
            or (table.lower(), column.lower()) in policy.denied_columns
        ):
            continue
        q_table, q_col = quote_identifier(table), quote_identifier(column)
        sql = (
            f"SELECT DISTINCT {q_col} FROM {q_table} "
            f"WHERE {q_col} IS NOT NULL AND length({q_col}) <= {MAX_VALUE_LENGTH} "
            f"LIMIT {MAX_EXAMPLES}"
        )
        try:
            set_deadline(conn, SAMPLE_TIMEOUT_S)
            values = [r[0] for r in conn.execute(sql).fetchall()]
        except sqlite3.Error:  # timeout or any other error: skip the column
            continue
        finally:
            conn.set_progress_handler(None, 0)
        if values:
            lines.append(f"{q_table}.{q_col}: " + ", ".join(sql_literal(v) for v in values))
    return lines


def build_schema_context(
    db_path: Path | str, *, sample_values: bool = True, policy: GuardPolicy | None = None
) -> str:
    """DDL, undeclared keys and (optionally) example values of a database.

    Tables denied by ``policy`` are left out, and so are example values of denied columns.
    """
    policy = policy or GuardPolicy()
    conn = connect_readonly(db_path)
    try:
        tables = _tables(conn, policy)
        parts = ["\n\n".join(ddl for _, ddl in tables)]
        keys = [line for name, ddl in tables for line in _key_lines(conn, name, ddl)]
        if keys:
            parts.append("Keys not declared in the statements above:\n" + "\n".join(keys))
        if sample_values:
            examples = [line for name, _ in tables for line in _example_values(conn, name, policy)]
            if examples:
                parts.append(
                    f"Example values (up to {MAX_EXAMPLES} distinct per text column):\n"
                    + "\n".join(examples)
                )
    finally:
        conn.close()
    return "\n\n".join(parts)


@lru_cache(maxsize=64)
def _cached_context(resolved_path: str, sample_values: bool, policy_json: str) -> str:
    return build_schema_context(
        resolved_path,
        sample_values=sample_values,
        policy=GuardPolicy.model_validate_json(policy_json),
    )


def schema_context(
    db_path: Path | str, *, sample_values: bool = True, policy: GuardPolicy | None = None
) -> str:
    """Cached :func:`build_schema_context` (the context is the same for every question)."""
    policy_json = (policy or GuardPolicy()).model_dump_json()
    return _cached_context(str(Path(db_path).resolve()), sample_values, policy_json)


def schema_block(context: str) -> str:
    """The stable prompt prefix shared by every node of one database."""
    return f"Database schema (SQLite):\n\n{context}\n\n"
