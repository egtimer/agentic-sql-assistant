"""Adapter that runs the LangGraph agent on a BIRD question and extracts its SQL."""

from pathlib import Path
from typing import Any

from evals.text2sql.baseline import CountingClient, cached_schema
from evals.text2sql.data import Question, db_path
from src.agent.graph import run_agent
from src.agent.state import ToolProfile
from src.llm import LLMClient, reset_client, set_client


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
