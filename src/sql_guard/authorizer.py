"""Layer 2: SQLite authorizer used as an allowlist of operations."""

import sqlite3
from dataclasses import dataclass
from typing import Any

from src.sql_guard.policy import GuardPolicy

ALLOWED_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
)

SCHEMA_TABLES = frozenset(
    {"sqlite_master", "sqlite_schema", "sqlite_temp_master", "sqlite_temp_schema"}
)
WRITE_ACTIONS = frozenset({sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE})

# Eponymous or system virtual tables that leak metadata or raw database pages.
DENIED_TABLE_PREFIXES = ("pragma_",)
DENIED_TABLES = frozenset({"sqlite_dbpage", "dbstat"})

# Functions that load code, touch files or expose build details. Size abuse
# (randomblob, zeroblob, printf) is handled by the length limit, not by denial.
DENIED_FUNCTIONS = frozenset(
    {
        "load_extension",
        "fts3_tokenizer",
        "readfile",
        "writefile",
        "edit",
        "sqlite_source_id",
        "sqlite_compileoption_get",
        "sqlite_compileoption_used",
    }
)

_ACTION_NAMES = {
    sqlite3.SQLITE_CREATE_INDEX: "CREATE INDEX",
    sqlite3.SQLITE_CREATE_TABLE: "CREATE TABLE",
    sqlite3.SQLITE_CREATE_TEMP_INDEX: "CREATE TEMP INDEX",
    sqlite3.SQLITE_CREATE_TEMP_TABLE: "CREATE TEMP TABLE",
    sqlite3.SQLITE_CREATE_TEMP_TRIGGER: "CREATE TEMP TRIGGER",
    sqlite3.SQLITE_CREATE_TEMP_VIEW: "CREATE TEMP VIEW",
    sqlite3.SQLITE_CREATE_TRIGGER: "CREATE TRIGGER",
    sqlite3.SQLITE_CREATE_VIEW: "CREATE VIEW",
    sqlite3.SQLITE_DELETE: "DELETE",
    sqlite3.SQLITE_DROP_INDEX: "DROP INDEX",
    sqlite3.SQLITE_DROP_TABLE: "DROP TABLE",
    sqlite3.SQLITE_DROP_TEMP_INDEX: "DROP TEMP INDEX",
    sqlite3.SQLITE_DROP_TEMP_TABLE: "DROP TEMP TABLE",
    sqlite3.SQLITE_DROP_TEMP_TRIGGER: "DROP TEMP TRIGGER",
    sqlite3.SQLITE_DROP_TEMP_VIEW: "DROP TEMP VIEW",
    sqlite3.SQLITE_DROP_TRIGGER: "DROP TRIGGER",
    sqlite3.SQLITE_DROP_VIEW: "DROP VIEW",
    sqlite3.SQLITE_INSERT: "INSERT",
    sqlite3.SQLITE_PRAGMA: "PRAGMA",
    sqlite3.SQLITE_TRANSACTION: "transaction control",
    sqlite3.SQLITE_UPDATE: "UPDATE",
    sqlite3.SQLITE_ATTACH: "ATTACH",
    sqlite3.SQLITE_DETACH: "DETACH",
    sqlite3.SQLITE_ALTER_TABLE: "ALTER TABLE",
    sqlite3.SQLITE_REINDEX: "REINDEX",
    sqlite3.SQLITE_ANALYZE: "ANALYZE",
    sqlite3.SQLITE_CREATE_VTABLE: "CREATE VIRTUAL TABLE",
    sqlite3.SQLITE_DROP_VTABLE: "DROP VIRTUAL TABLE",
    sqlite3.SQLITE_SAVEPOINT: "SAVEPOINT",
}


@dataclass
class Denial:
    """First action the authorizer refused (later ones are consequences)."""

    reason: str | None = None


def decide(policy: GuardPolicy, action: int, arg1: str | None, arg2: str | None) -> str | None:
    """None when allowed, else the reason for refusing (pure, also used for dry checks)."""
    if action not in ALLOWED_ACTIONS:
        if (arg1 or "").lower() in SCHEMA_TABLES and action in WRITE_ACTIONS:
            return (
                f"writing to {arg1} is not allowed "
                "(schema change, ANALYZE or a virtual table such as pragma_*)"
            )
        name = _ACTION_NAMES.get(action, f"operation code {action}")
        return f"{name} is not allowed"
    if action == sqlite3.SQLITE_READ:
        table = (arg1 or "").lower()
        column = (arg2 or "").lower()
        if table in DENIED_TABLES or table.startswith(DENIED_TABLE_PREFIXES):
            return f"reading system table {arg1} is not allowed"
        if table in policy.denied_tables:
            return f"table {arg1} is denied by policy"
        if (table, column) in policy.denied_columns:
            return f"column {arg1}.{arg2} is denied by policy"
    elif action == sqlite3.SQLITE_FUNCTION:
        function = (arg2 or "").lower()
        if function in DENIED_FUNCTIONS or function.startswith(DENIED_TABLE_PREFIXES):
            return f"function {arg2} is not allowed"
    return None


def make_authorizer(policy: GuardPolicy, denial: Denial) -> Any:
    """Authorizer callback that records the first refusal in ``denial``."""

    def authorizer(
        action: int, arg1: str | None, arg2: str | None, dbname: str | None, source: str | None
    ) -> int:
        reason = decide(policy, action, arg1, arg2)
        if reason is None:
            return sqlite3.SQLITE_OK
        if denial.reason is None:
            denial.reason = reason
        return sqlite3.SQLITE_DENY

    return authorizer
