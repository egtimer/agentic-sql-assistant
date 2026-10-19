"""Nodes and routing of the SQL agent v2. The flow is decided here, in code."""

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

from langgraph.store.base import BaseStore

from src.llm import get_client
from src.observability import tracing
from src.sql_agent.executor import Attempt, format_result, run_query, select_final_sql
from src.sql_agent.extract import extract_check_sql, extract_sql
from src.sql_agent.introspect import schema_block, schema_context
from src.sql_agent.memory import recall
from src.sql_agent.prompts import (
    CHECK_SYSTEM,
    GENERATE_SYSTEM,
    RESPOND_SYSTEM,
    check_user,
    question_block,
    respond_user,
)
from src.sql_agent.state import SqlAgentState, Turn, Verdict


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


_NodeFn = TypeVar("_NodeFn", bound=Callable[[SqlAgentState], dict[str, Any]])


def _traced(
    name: str,
    span_input: Callable[[SqlAgentState], dict[str, Any]],
    span_output: Callable[[SqlAgentState, dict[str, Any]], dict[str, Any]],
) -> Callable[[_NodeFn], _NodeFn]:
    """Record a node as a span of the current trace (nothing happens without a trace)."""

    def decorator(func: _NodeFn) -> _NodeFn:
        @functools.wraps(func)
        def wrapper(state: SqlAgentState) -> dict[str, Any]:
            if not tracing.active():
                return func(state)
            with tracing.span(name, input=span_input(state)) as handle:
                update = func(state)
                handle.update(output=span_output(state, update))
                return update

        return cast(_NodeFn, wrapper)

    return decorator


def _attempt_output(attempt: Attempt) -> dict[str, Any]:
    return {
        "columns": attempt.columns,
        "row_count": attempt.row_count_text,
        "truncated": attempt.more_rows,
        "error": attempt.error,
        "blocked": attempt.blocked,
        "block_reason": attempt.block_reason,
        "layer": attempt.layer,
        "elapsed_s": round(attempt.elapsed_s, 4),
    }


def _check_output(update: dict[str, Any]) -> dict[str, Any]:
    verdict: Verdict = update["verdicts"][-1]
    return {"verdict": verdict.kind, "fix_sql": verdict.sql or None, "note": verdict.note}


@_traced(
    "introspect",
    lambda s: {"db_path": s.db_path, "sample_values": s.sample_values},
    lambda s, u: {"schema_context": u["schema_context"], "chars": len(u["schema_context"])},
)
def introspect_node(state: SqlAgentState) -> dict[str, Any]:
    return {
        "schema_context": schema_context(
            state.db_path, sample_values=state.sample_values, policy=state.policy
        )
    }


def recall_node(state: SqlAgentState, *, store: BaseStore | None = None) -> dict[str, Any]:
    """Read this user's memories for the database and keep the ones relevant to the question.

    ``store`` is injected by LangGraph (the store the graph was compiled with). Without a
    store or a ``user_id`` nothing is recalled and the prompts stay what they were.
    """
    db_id = Path(state.db_path).stem
    if store is None or not state.user_id:
        return {"recalled": [], "n_memories": 0}
    with tracing.span(
        "recall", input={"user_id": state.user_id, "db_id": db_id, "question": state.question}
    ) as handle:
        memories, available = recall(store, state.user_id, db_id, state.question, state.evidence)
        handle.update(
            output={
                "available": available,
                "injected": [{"key": m.key, "definition": m.definition} for m in memories],
            }
        )
    return {"recalled": memories, "n_memories": available}


@_traced(
    "generate",
    lambda s: {"question": s.question, "evidence": s.evidence},
    lambda s, u: {"sql": u["candidate_sql"]},
)
def generate_node(state: SqlAgentState) -> dict[str, Any]:
    completion = get_client().complete(
        GENERATE_SYSTEM,
        question_block(state.question, state.evidence, state.history, state.recalled),
        tag="generate",
        prefix=schema_block(state.schema_context),
    )
    return {"candidate_sql": extract_sql(completion.text)}


@_traced(
    "execute",
    lambda s: {"sql": s.candidate_sql},
    lambda s, u: _attempt_output(u["attempts"][-1]),
)
def execute_node(state: SqlAgentState) -> dict[str, Any]:
    attempt = run_query(state.db_path, state.candidate_sql, policy=state.policy)
    return {"attempts": [*state.attempts, attempt]}


@_traced(
    "check",
    lambda s: {"sql": s.attempts[-1].sql, "result": format_result(s.attempts[-1])},
    lambda s, u: _check_output(u),
)
def check_node(state: SqlAgentState) -> dict[str, Any]:
    attempt = state.attempts[-1]
    completion = get_client().complete(
        CHECK_SYSTEM,
        check_user(state.question, state.evidence, attempt, state.history, state.recalled),
        tag="check",
        prefix=schema_block(state.schema_context),
    )
    verdict = parse_check(completion.text, attempt.sql)
    update: dict[str, Any] = {"verdicts": [*state.verdicts, verdict]}
    if verdict.kind == "fix":
        update["fixes_used"] = state.fixes_used + 1
        update["candidate_sql"] = verdict.sql
    return update


@_traced(
    "respond",
    lambda s: {"final_sql": select_final_sql(s.attempts), "skip_respond": s.skip_respond},
    lambda s, u: {"final_sql": u["final_sql"], "response": u["response"]},
)
def respond_node(state: SqlAgentState) -> dict[str, Any]:
    """Pick the final SQL, write the answer and add this turn to the conversation history."""
    final_sql = select_final_sql(state.attempts)
    final = next((a for a in reversed(state.attempts) if a.sql == final_sql), None)
    turn = Turn(
        question=state.question,
        final_sql=final_sql,
        row_count=final.row_count if final is not None and final.ok else None,
        columns=list(final.columns) if final is not None and final.ok else [],
    )
    history = [*state.history, turn]
    if state.skip_respond:
        return {"final_sql": final_sql, "response": "", "history": history}
    completion = get_client().complete(
        RESPOND_SYSTEM, respond_user(state.question, final), tag="respond"
    )
    return {"final_sql": final_sql, "response": completion.text.strip(), "history": history}


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
