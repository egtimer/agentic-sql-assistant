"""Nodes and routing of the SQL agent v2. The flow is decided here, in code."""

from typing import Any

from src.llm import get_client
from src.sql_agent.executor import run_query, select_final_sql
from src.sql_agent.extract import extract_check_sql, extract_sql
from src.sql_agent.introspect import schema_block, schema_context
from src.sql_agent.prompts import (
    CHECK_SYSTEM,
    GENERATE_SYSTEM,
    RESPOND_SYSTEM,
    check_user,
    question_block,
    respond_user,
)
from src.sql_agent.state import SqlAgentState, Verdict


def parse_check(text: str, current_sql: str = "") -> Verdict:
    """Reviewer answer -> verdict. Rules, in order:

    1. A fenced block with a query is a fix (``sql``/``sqlite`` tag, or untagged and
       starting with SELECT/WITH). It wins over an OK anywhere in the text: a reviewer that
       says "OK" but also proposes a query is changing the query, and the proposal is what
       the model actually wrote. A fix identical to the current query counts as ok.
    2. Otherwise the first or the last non-empty line, stripped of markdown emphasis,
       backticks and trailing punctuation, equal to OK (any case) is ok. Reviewers often
       explain first and end with "OK", so a leading OK is not required.
    3. Anything else is treated as ok but flagged ``unparsed``.

    ``Verdict.text`` always keeps the full answer.
    """
    fixed = extract_check_sql(text)
    if fixed:
        note = text.split("```", 1)[0].strip()
        if _normalized(fixed) == _normalized(current_sql):
            return Verdict(kind="ok", note=note, text=text)
        return Verdict(kind="fix", sql=fixed, note=note, text=text)
    if _says_ok(text):
        return Verdict(kind="ok", text=text)
    return Verdict(kind="unparsed", note=text.strip()[:200], text=text)


def _says_ok(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    return any(_is_ok_line(line) for line in (lines[:1] + lines[-1:]))


def _is_ok_line(line: str) -> bool:
    return line.strip().strip("*_`~ \t").rstrip(".!:;,").strip("*_`~ \t").upper() == "OK"


def _normalized(sql: str) -> str:
    return " ".join(sql.split()).rstrip(";").strip()


def introspect_node(state: SqlAgentState) -> dict[str, Any]:
    return {
        "schema_context": schema_context(
            state.db_path, sample_values=state.sample_values, policy=state.policy
        )
    }


def generate_node(state: SqlAgentState) -> dict[str, Any]:
    completion = get_client().complete(
        GENERATE_SYSTEM,
        question_block(state.question, state.evidence),
        tag="generate",
        prefix=schema_block(state.schema_context),
    )
    return {"candidate_sql": extract_sql(completion.text)}


def execute_node(state: SqlAgentState) -> dict[str, Any]:
    attempt = run_query(state.db_path, state.candidate_sql, policy=state.policy)
    return {"attempts": [*state.attempts, attempt]}


def check_node(state: SqlAgentState) -> dict[str, Any]:
    attempt = state.attempts[-1]
    completion = get_client().complete(
        CHECK_SYSTEM,
        check_user(state.question, state.evidence, attempt),
        tag="check",
        prefix=schema_block(state.schema_context),
    )
    verdict = parse_check(completion.text, attempt.sql)
    update: dict[str, Any] = {"verdicts": [*state.verdicts, verdict]}
    if verdict.kind == "fix":
        update["fixes_used"] = state.fixes_used + 1
        update["candidate_sql"] = verdict.sql
    return update


def respond_node(state: SqlAgentState) -> dict[str, Any]:
    final_sql = select_final_sql(state.attempts)
    if state.skip_respond:
        return {"final_sql": final_sql, "response": ""}
    final = next((a for a in reversed(state.attempts) if a.sql == final_sql), None)
    completion = get_client().complete(
        RESPOND_SYSTEM, respond_user(state.question, final), tag="respond"
    )
    return {"final_sql": final_sql, "response": completion.text.strip()}


def route_after_execute(state: SqlAgentState) -> str:
    """``check`` for failures (error recovery) and, with ``use_check``, for successes."""
    if state.fixes_used >= state.max_fixes:
        return "respond"
    if not state.attempts[-1].ok or state.use_check:
        return "check"
    return "respond"


def route_after_check(state: SqlAgentState) -> str:
    """A fix goes back to ``execute``; anything else ends the loop."""
    return "execute" if state.verdicts and state.verdicts[-1].kind == "fix" else "respond"
