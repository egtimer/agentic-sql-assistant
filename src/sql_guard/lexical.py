"""Layer 1: lexical check of the statement text.

Mirrors the SQLite tokenizer closely enough to find statement boundaries and the first
keyword without being fooled by quotes or comments. It covers statements that never reach
the authorizer (``VACUUM``, ``ATTACH`` and friends) and gives clear reasons early.
"""

import re

MAX_SQL_LENGTH = 100_000
ALLOWED_FIRST_KEYWORDS = frozenset({"SELECT", "WITH", "VALUES"})
WRITE_KEYWORDS = frozenset(
    {"INSERT", "UPDATE", "DELETE", "REPLACE", "DROP", "ALTER", "CREATE", "TRUNCATE"}
)
_WHITESPACE = " \t\n\f\r"  # exactly what SQLite treats as whitespace
_KEYWORD = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_CLOSERS = {"'": "'", '"': '"', "`": "`", "[": "]"}


def _skip_trivia(sql: str, i: int) -> int:
    """Index of the first character at or after ``i`` that is neither whitespace nor comment."""
    n = len(sql)
    while i < n:
        if sql[i] in _WHITESPACE:
            i += 1
        elif sql.startswith("--", i):
            end = sql.find("\n", i)
            i = n if end == -1 else end + 1
        elif sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            i = n if end == -1 else end + 2
        else:
            break
    return i


def _skip_quoted(sql: str, i: int) -> int:
    """Index after the quoted token starting at ``i`` (doubled closers escape)."""
    closer = _CLOSERS[sql[i]]
    n = len(sql)
    j = i + 1
    while j < n:
        if sql[j] == closer:
            if closer != "]" and j + 1 < n and sql[j + 1] == closer:
                j += 2
                continue
            return j + 1
        j += 1
    return n


def _has_second_statement(sql: str, start: int) -> bool:
    """True when a ``;`` outside quotes and comments is followed by more than trivia."""
    n = len(sql)
    i = start
    while i < n:
        c = sql[i]
        if c in _CLOSERS:
            i = _skip_quoted(sql, i)
        elif sql.startswith("--", i) or sql.startswith("/*", i):
            i = _skip_trivia(sql, i)
        elif c == ";":
            return _skip_trivia(sql, i + 1) < n
        else:
            i += 1
    return False


def check_sql(sql: str, max_length: int = MAX_SQL_LENGTH) -> str | None:
    """Reason why ``sql`` must not run, or None when it passes layer 1."""
    if "\x00" in sql:
        return "NUL characters are not allowed in SQL."
    if len(sql) > max_length:
        return f"SQL is longer than {max_length} characters."
    start = _skip_trivia(sql, 0)
    if start >= len(sql):
        return "Empty query."
    match = _KEYWORD.match(sql, start)
    keyword = match.group(0).upper() if match else sql[start : start + 12]
    if keyword in WRITE_KEYWORDS:
        return "Write operations are not allowed. Only SELECT queries are permitted."
    if keyword not in ALLOWED_FIRST_KEYWORDS:
        return f"Only SELECT queries are permitted (statement starts with {keyword})."
    if _has_second_statement(sql, start):
        return "Multiple statements are not allowed. Submit one query at a time."
    return None
