"""Adapter that runs the LangGraph agent on a BIRD question and extracts its SQL."""

from pathlib import Path
from typing import Any

from evals.text2sql.baseline import CountingClient, cached_schema
from evals.text2sql.data import Question, db_path
from src.agent.graph import run_agent
from src.agent.state import ToolProfile
from src.llm import LLMClient, reset_client, set_client
from src.sql_agent.graph import run_sql_agent


def build_agent_query(question: Question) -> str:
    """Question followed by the BIRD evidence, as the agent's user query."""
    query = question.question.strip()
    if question.evidence.strip():
        query += "\nExternal knowledge: " + question.evidence.strip()
    return query


def select_sql(tool_results: list[dict[str, Any]]) -> str:
    """Predicted SQL: input of the last successful sql_query, else of the last sql_query."""
    sql_calls = [r for r in tool_results if r.get("tool") == "sql_query"]
    for result in reversed(sql_calls):
        if result.get("success"):
            return str(result["input"])
    if sql_calls:
        return str(sql_calls[-1]["input"])
    return ""


def predict_agent(
    question: Question,
    client: LLMClient,
    *,
    database: Path | None = None,
    skip_respond: bool = False,
) -> tuple[str, int]:
    """Run the agent (sql_only profile) on one question. Returns ``(sql, n_llm_calls)``.

    ``skip_respond`` drops the final natural-language answer call, which EX does not score.
    """
    database = database or db_path(question.db_id)
    counting = CountingClient(client)
    token = set_client(counting)
    try:
        result = run_agent(
            build_agent_query(question),
            schema_description=cached_schema(str(database)),
            db_path=str(database),
            profile=ToolProfile.SQL_ONLY,
            skip_respond=skip_respond,
        )
    finally:
        reset_client(token)
    return select_sql(result["tool_results"]), counting.n_calls


def predict_agent_v2(
    question: Question,
    client: LLMClient,
    *,
    database: Path | None = None,
    skip_respond: bool = False,
    use_check: bool = True,
    max_fixes: int = 2,
) -> tuple[str, int, dict[str, Any]]:
    """Run SQL agent v2 on one question. Returns ``(sql, n_llm_calls, details)``.

    ``details`` holds the attempts (sql, error, row count, number of columns), the check
    verdicts and the number of fixes, for the runner to record.
    """
    database = database or db_path(question.db_id)
    counting = CountingClient(client)
    token = set_client(counting)
    try:
        result = run_sql_agent(
            question.question,
            question.evidence,
            database,
            use_check=use_check,
            max_fixes=max_fixes,
            skip_respond=skip_respond,
        )
    finally:
        reset_client(token)
    details = {
        "attempts": [
            {
                "sql": a.sql,
                "error": a.error,
                "row_count": a.row_count_text if a.ok else None,
                "n_columns": len(a.columns),
            }
            for a in result.attempts
        ],
        "verdicts": [
            {"kind": v.kind, "unparsed": v.unparsed, "note": v.note} for v in result.verdicts
        ],
        "fixes_used": result.fixes_used,
    }
    return result.final_sql, counting.n_calls, details


def predict_agent_v2_nocheck(
    question: Question, client: LLMClient, **options: Any
) -> tuple[str, int, dict[str, Any]]:
    """Agent v2 without the review of successful queries (error recovery only)."""
    return predict_agent_v2(question, client, use_check=False, **options)
