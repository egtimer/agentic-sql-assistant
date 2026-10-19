"""Pull SQL out of LLM answers."""

import re

_FENCE = re.compile(r"```[ \t]*([A-Za-z]*)[ \t]*\n?(.*?)```", re.DOTALL)
_STATEMENT_START = re.compile(
    r"\bSELECT\s|^[ \t]*WITH\s+(?:RECURSIVE\s+)?\w+\s*(?:\([^)]*\))?\s*AS\b",
    re.IGNORECASE | re.MULTILINE,
)


def _first_statement(text: str) -> str:
    """Cut ``text`` at the first semicolon that is outside quotes."""
    quote: str | None = None
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"', "`"):
            quote = ch
        elif ch == ";":
            return text[:i]
    return text


def _clean(sql: str) -> str:
    return sql.strip().rstrip(";").strip()


def extract_sql(text: str) -> str:
    """Pull the SQL out of an LLM answer.

    Uses the first ```sql fenced block (else the first fenced block). Without a fence, uses
    the first statement that starts with SELECT or WITH. Returns "" when nothing is found.
    """
    blocks = _FENCE.findall(text)
    for lang, body in blocks:
        if lang.lower() == "sql" and body.strip():
            return _clean(body)
    for _, body in blocks:
        if body.strip():
            return _clean(body)
    match = _STATEMENT_START.search(text)
    if match:
        return _clean(_first_statement(text[match.start() :]))
    return ""


_LEADING_COMMENTS = re.compile(r"(?:\s+|--[^\n]*(?:\n|$)|/\*.*?\*/)*", re.DOTALL)
_SQL_START = re.compile(r"(?:SELECT|WITH)\b", re.IGNORECASE)
_SQL_TAGS = frozenset({"sql", "sqlite"})


def extract_check_sql(text: str) -> str:
    """Body of the first fenced block that holds a query, or "" (no fallback).

    A block counts when it is tagged ``sql`` or ``sqlite`` (any case), or when it is
    untagged and its body starts, after whitespace and SQL comments, with SELECT or WITH.
    """
    for lang, body in _FENCE.findall(text):
        if not body.strip():
            continue
        if lang.lower() in _SQL_TAGS:
            return _clean(body)
        if not lang:
            start = _LEADING_COMMENTS.match(body)
            if start and _SQL_START.match(body, start.end()):
                return _clean(body)
    return ""
