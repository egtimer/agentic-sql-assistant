"""Single-shot text-to-SQL baseline: one LLM call, no tools, no retries."""

from functools import lru_cache
from pathlib import Path

from evals.text2sql.data import Question, db_path, schema_ddl
from src.llm import CallUsage, Completion, LLMClient
from src.sql_agent.extract import extract_sql

BASELINE_SYSTEM = (
    "You are an expert SQLite analyst. Given a database schema, optional external knowledge "
    "and a question, answer with exactly one SQLite query inside a ```sql code block and "
    "nothing else."
)


class CountingClient:
    """Wraps a client; counts the calls made through it and keeps their usage."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client
        self.usages: list[CallUsage] = []

    @property
    def n_calls(self) -> int:
        return len(self.usages)

    def complete(
        self, system: str | None, user: str, *, tag: str, prefix: str | None = None
    ) -> Completion:
        completion = self._client.complete(system, user, tag=tag, prefix=prefix)
        self.usages.append(completion.usage)
        return completion


@lru_cache(maxsize=32)
def cached_schema(database: str) -> str:
    """Schema DDL of a database file (cached: it is the same for every question)."""
    return schema_ddl(database)


def build_baseline_parts(question: Question, schema: str) -> tuple[str, str, str]:
    """Return ``(system, prefix, suffix)``.

    The prefix (schema block) is stable across the questions of one database and is what
    gets cached; the suffix is the per-question text (evidence and question). Stable content
    comes first so local servers can also reuse their KV prefix.
    """
    prefix = f"Database schema (SQLite):\n\n{schema}\n\n"
    parts = []
    if question.evidence.strip():
        parts.append(f"-- External Knowledge: {question.evidence.strip()}")
    parts.append(f"Question: {question.question.strip()}")
    return BASELINE_SYSTEM, prefix, "\n".join(parts)


def build_baseline_prompt(question: Question, schema: str) -> tuple[str, str]:
    """Return ``(system, user)``: the whole user prompt, ``prefix + suffix``."""
    system, prefix, suffix = build_baseline_parts(question, schema)
    return system, prefix + suffix


def predict_baseline(
    question: Question, client: LLMClient, *, database: Path | None = None
) -> tuple[str, int]:
    """Predict SQL with a single LLM call. Returns ``(sql, n_llm_calls)``."""
    counting = CountingClient(client)
    schema = cached_schema(str(database or db_path(question.db_id)))
    system, prefix, suffix = build_baseline_parts(question, schema)
    completion = counting.complete(system, suffix, tag="baseline", prefix=prefix)
    return extract_sql(completion.text), counting.n_calls
