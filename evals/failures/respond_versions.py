"""The answer step's prompts, by version, so a final-answer pass can rebuild either.

- ``v1``: the prompt of the committed runs up to 2026-11-17 (question and result only, one or
  two sentences). The failure analysis of 2026-11-17 measured it.
- ``v2``: the agent's current prompt (``src/sql_agent/prompts.py``): the final SQL in the
  prompt, the exact row count when the result is larger than the fetch limit, and an
  instruction not to present a partial list as complete.
"""

from collections.abc import Callable
from pathlib import Path

from src.sql_agent.executor import Attempt, count_rows, format_result
from src.sql_agent.prompts import RESPOND_SYSTEM, respond_user

RESPOND_SYSTEM_V1 = (
    "You answer a data question in one or two short sentences, using only the query result "
    "you are given. Do not mention SQL."
)


def respond_user_v1(question: str, attempt: Attempt | None) -> str:
    result = format_result(attempt) if attempt is not None else "No query result."
    return f"Question: {question.strip()}\n\nQuery result:\n{result}"


PromptBuilder = Callable[[str, Attempt, Path, float], tuple[str, str]]


def _v1(question: str, attempt: Attempt, database: Path, timeout_s: float) -> tuple[str, str]:
    return RESPOND_SYSTEM_V1, respond_user_v1(question, attempt)


def _v2(question: str, attempt: Attempt, database: Path, timeout_s: float) -> tuple[str, str]:
    total = count_rows(database, attempt.sql, timeout_s) if attempt.more_rows else None
    return RESPOND_SYSTEM, respond_user(question, attempt, total)


VERSIONS: dict[str, PromptBuilder] = {"v1": _v1, "v2": _v2}
