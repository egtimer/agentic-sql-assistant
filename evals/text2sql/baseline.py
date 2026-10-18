"""Single-shot text-to-SQL baseline: one LLM call, no tools, no retries."""

import re
from functools import lru_cache
from pathlib import Path

from evals.text2sql.data import Question, db_path, schema_ddl
from src.llm import Completion, LLMClient

BASELINE_SYSTEM = (
    "You are an expert SQLite analyst. Given a database schema, optional external knowledge "
    "and a question, answer with exactly one SQLite query inside a ```sql code block and "
    "nothing else."
)


class CountingClient:
    """Wraps a client and counts the calls made through it."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client
        self.n_calls = 0

    def complete(self, system: str | None, user: str, *, tag: str) -> Completion:
        self.n_calls += 1
        return self._client.complete(system, user, tag=tag)


@lru_cache(maxsize=32)
def cached_schema(database: str) -> str:
    """Schema DDL of a database file (cached: it is the same for every question)."""
    return schema_ddl(database)


def build_baseline_prompt(question: Question, schema: str) -> tuple[str, str]:
    """Return ``(system, user)``.

    Stable content (instructions, schema) comes first and the per-question text last, so
    local servers can reuse the cached prompt prefix across questions of one database.
    """
    parts = [f"Database schema (SQLite):\n\n{schema}\n"]
    if question.evidence.strip():
        parts.append(f"-- External Knowledge: {question.evidence.strip()}")
    parts.append(f"Question: {question.question.strip()}")
    return BASELINE_SYSTEM, "\n".join(parts)


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


def predict_baseline(
    question: Question, client: LLMClient, *, database: Path | None = None
) -> tuple[str, int]:
    """Predict SQL with a single LLM call. Returns ``(sql, n_llm_calls)``."""
    counting = CountingClient(client)
    schema = cached_schema(str(database or db_path(question.db_id)))
    system, user = build_baseline_prompt(question, schema)
    completion = counting.complete(system, user, tag="baseline")
    return extract_sql(completion.text), counting.n_calls
