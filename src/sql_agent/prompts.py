"""Prompts of the SQL agent v2.

Every node shares the same stable prefix (the schema block of one database) and puts the
per-question text in the user part, so the prefix can be cached by providers that support it.
"""

from collections.abc import Sequence

from src.sql_agent.executor import Attempt, format_result
from src.sql_agent.state import Turn

GENERATE_SYSTEM = """\
You are an expert SQLite analyst. Write exactly one SQLite query whose result IS the answer \
to the question.

Rules:
- Select only the columns the question asks for, in the order asked. No helper, id or context \
columns that were not requested.
- Compute ratios, percentages, differences and averages in SQL so the result is a single \
value. There is no calculator and no second step.
- For superlatives (highest, lowest, most, least, top N) use ORDER BY with LIMIT. Mind ties \
only if the question asks for all of them.
- Use DISTINCT when listing entities that can repeat because of joins.
- Apply the external knowledge (formulas and value mappings) exactly as given.
- Use only tables and columns that exist in the schema, and filter values consistent with the \
example values shown.
- Answer with the SQL in a ```sql block and nothing else."""

CHECK_SYSTEM = """\
You are a strict reviewer of SQLite queries. You get a question, optional external knowledge, \
a candidate query and the result of running it. Decide whether the query answers the \
question.

Checklist:
(a) Columns: exactly what the question asks, nothing extra, nothing missing, in the right \
order.
(b) Aggregation: a ratio, percentage, count or average must come back as one value.
(c) Row count: LIMIT for superlatives, DISTINCT for lists of entities.
(d) Empty result or execution error: the filters probably use wrong values or columns; fix \
them using the example values in the schema.
(e) The external knowledge formula is applied exactly.
(f) Joins and filters match the question.

Do not change a query that is correct; only fix concrete problems you can name.
If you find no concrete problem, answer exactly: OK
Otherwise optionally write one short line naming the problem, then give the corrected full \
query in a ```sql block."""

RESPOND_SYSTEM = (
    "You answer a data question in one or two short sentences, using only the query result "
    "you are given. Do not mention SQL."
)


HISTORY_TURNS = 3  # earlier turns shown in the prompts
HISTORY_HEADER = "Earlier in this conversation:"


def history_block(history: Sequence[Turn]) -> str:
    """The last ``HISTORY_TURNS`` turns (question and final SQL); "" for an empty history."""
    if not history:
        return ""
    lines = [HISTORY_HEADER]
    for turn in history[-HISTORY_TURNS:]:
        sql = " ".join(turn.final_sql.split()) or "(no query)"
        lines.append(f"- Question: {turn.question.strip()}")
        lines.append(f"  SQL: {sql}")
    return "\n".join(lines)


def question_block(question: str, evidence: str, history: Sequence[Turn] = ()) -> str:
    """Per-question text of the generate prompt (history, evidence, question last).

    Without history the text is exactly what it was before conversations existed.
    """
    parts = []
    if history:
        parts.append(history_block(history) + "\n")
    if evidence.strip():
        parts.append(f"-- External Knowledge: {evidence.strip()}")
    parts.append(f"Question: {question.strip()}")
    return "\n".join(parts)


def signals(attempt: Attempt) -> list[str]:
    """Facts about a result computed in code, so the reviewer does not have to count."""
    if not attempt.ok:
        return []
    out = []
    if attempt.row_count == 0:
        out.append("The result is empty.")
    n = len(attempt.columns)
    out.append(f"The result has {n} column{'' if n == 1 else 's'}.")
    count = attempt.row_count_text
    out.append(f"The result has {count} row{'' if count == '1' else 's'}.")
    return out


def check_user(question: str, evidence: str, attempt: Attempt, history: Sequence[Turn] = ()) -> str:
    """Per-question text of the check prompt."""
    parts = [
        question_block(question, evidence, history),
        f"Candidate query:\n```sql\n{attempt.sql}\n```",
    ]
    parts.append("Execution result:\n" + format_result(attempt))
    sig = signals(attempt)
    if sig:
        parts.append("Signals:\n" + "\n".join(f"- {s}" for s in sig))
    return "\n\n".join(parts)


def respond_user(question: str, attempt: Attempt | None) -> str:
    result = format_result(attempt) if attempt is not None else "No query result."
    return f"Question: {question.strip()}\n\nQuery result:\n{result}"


__all__ = [
    "CHECK_SYSTEM",
    "GENERATE_SYSTEM",
    "HISTORY_HEADER",
    "HISTORY_TURNS",
    "RESPOND_SYSTEM",
    "check_user",
    "history_block",
    "question_block",
    "respond_user",
    "signals",
]
