"""Guarded execution of model-written SQL (see docs/ADR-005-sql-guardrails.md)."""

from src.sql_guard.execute import (
    BLOCKED_PREFIX,
    TIMEOUT_ERROR,
    GuardResult,
    guarded_execute,
    open_guarded,
)
from src.sql_guard.lexical import check_sql
from src.sql_guard.policy import GuardPolicy

__all__ = [
    "BLOCKED_PREFIX",
    "TIMEOUT_ERROR",
    "GuardPolicy",
    "GuardResult",
    "check_sql",
    "guarded_execute",
    "open_guarded",
]
