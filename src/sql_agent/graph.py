"""LangGraph graph and public entry point of the SQL agent v2."""

from functools import lru_cache
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.sql_agent.nodes import (
    check_node,
    execute_node,
    generate_node,
    introspect_node,
    respond_node,
    route_after_check,
    route_after_execute,
)
from src.sql_agent.state import SqlAgentResult, SqlAgentState


@lru_cache(maxsize=1)
def build_graph() -> Any:
    """Compile the graph: introspect -> generate -> execute <-> check -> respond."""
    graph = StateGraph(SqlAgentState)
    graph.add_node("introspect", introspect_node)
    graph.add_node("generate", generate_node)
    graph.add_node("execute", execute_node)
    graph.add_node("check", check_node)
    graph.add_node("respond", respond_node)
    graph.add_edge(START, "introspect")
    graph.add_edge("introspect", "generate")
    graph.add_edge("generate", "execute")
    graph.add_conditional_edges("execute", route_after_execute, ["check", "respond"])
    graph.add_conditional_edges("check", route_after_check, ["execute", "respond"])
    graph.add_edge("respond", END)
    return graph.compile()


def run_sql_agent(
    question: str,
    evidence: str,
    db_path: Path | str,
    *,
    sample_values: bool = True,
    use_check: bool = True,
    max_fixes: int = 2,
    skip_respond: bool = False,
) -> SqlAgentResult:
    """Answer ``question`` over the SQLite file at ``db_path`` with one final SQL query.

    LLM calls: 1 generate + at most ``max_fixes`` checks (each ends the loop or yields a
    fix), plus ``respond`` unless ``skip_respond``.
    """
    initial = SqlAgentState(
        question=question,
        evidence=evidence,
        db_path=str(db_path),
        sample_values=sample_values,
        use_check=use_check,
        max_fixes=max_fixes,
        skip_respond=skip_respond,
    )
    out = SqlAgentState.model_validate(build_graph().invoke(initial))
    return SqlAgentResult(
        final_sql=out.final_sql,
        attempts=out.attempts,
        verdicts=out.verdicts,
        fixes_used=out.fixes_used,
        response=out.response,
        schema_context=out.schema_context,
    )
