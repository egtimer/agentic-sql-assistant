"""LangGraph graph and public entry point of the SQL agent v2."""

import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore

from src.observability import tracing
from src.sql_agent.nodes import (
    check_node,
    execute_node,
    generate_node,
    introspect_node,
    recall_node,
    respond_node,
    route_after_check,
    route_after_execute,
)
from src.sql_agent.state import SqlAgentResult, SqlAgentState
from src.sql_guard import GuardPolicy

# Types that end up in a checkpoint. LangGraph warns about (and a future version blocks)
# types it was not told about, so savers made here list them.
CHECKPOINT_TYPES = (
    ("src.sql_guard.policy", "GuardPolicy"),
    ("src.sql_agent.executor", "Attempt"),
    ("src.sql_agent.state", "Verdict"),
    ("src.sql_agent.state", "Turn"),
    ("src.sql_agent.memory", "Memory"),
)


def checkpoint_serde() -> JsonPlusSerializer:
    """Serializer that accepts the state types of this agent (and blocks all others)."""
    return JsonPlusSerializer(allowed_msgpack_modules=list(CHECKPOINT_TYPES))


def memory_checkpointer() -> InMemorySaver:
    """Checkpointer that keeps conversations in this process only."""
    return InMemorySaver(serde=checkpoint_serde())


def open_checkpointer(path: Path | str) -> SqliteSaver:
    """Checkpointer that keeps conversations in the SQLite file ``path`` (kept open)."""
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn, serde=checkpoint_serde())


def build_graph(
    checkpointer: BaseCheckpointSaver[Any] | None = None, store: BaseStore | None = None
) -> Any:
    """Compile the graph: introspect [-> recall] -> generate -> execute <-> check -> respond.

    ``recall`` is part of the graph only with a ``store``. Without a checkpointer and a store
    the compiled graph is cached; otherwise a new one is compiled around them.
    """
    if checkpointer is None and store is None:
        return _default_graph()
    return _state_graph(with_recall=store is not None).compile(
        checkpointer=checkpointer, store=store
    )


@lru_cache(maxsize=1)
def _default_graph() -> Any:
    return _state_graph(with_recall=False).compile()


def _state_graph(*, with_recall: bool) -> StateGraph[SqlAgentState]:
    graph = StateGraph(SqlAgentState)
    graph.add_node("introspect", introspect_node)
    graph.add_node("generate", generate_node)
    graph.add_node("execute", execute_node)
    graph.add_node("check", check_node)
    graph.add_node("respond", respond_node)
    graph.add_edge(START, "introspect")
    if with_recall:
        graph.add_node("recall", recall_node)
        graph.add_edge("introspect", "recall")
        graph.add_edge("recall", "generate")
    else:
        graph.add_edge("introspect", "generate")
    graph.add_edge("generate", "execute")
    graph.add_conditional_edges("execute", route_after_execute, ["check", "respond"])
    graph.add_conditional_edges("check", route_after_check, ["execute", "respond"])
    graph.add_edge("respond", END)
    return graph


def run_sql_agent(
    question: str,
    evidence: str,
    db_path: Path | str,
    *,
    sample_values: bool = True,
    use_check: bool = True,
    max_fixes: int = 2,
    skip_respond: bool = False,
    policy: GuardPolicy | None = None,
    thread_id: str | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    store: BaseStore | None = None,
    user_id: str | None = None,
) -> SqlAgentResult:
    """Answer ``question`` over the SQLite file at ``db_path`` with one final SQL query.

    LLM calls: 1 generate + at most ``max_fixes`` checks (each ends the loop or yields a
    fix), plus ``respond`` unless ``skip_respond``. Every candidate runs under the SQL
    guard with ``policy`` (tables and columns it denies are also hidden from the schema
    context); a blocked attempt reaches ``check`` as an error starting with ``blocked:``.

    Conversations: with ``thread_id`` and a ``checkpointer`` the turns that share a thread id
    form one conversation. Every per-turn field starts from scratch on each turn; only the
    ``history`` (question, final SQL, row count and columns per turn, no rows) is carried over
    and shown to ``generate`` and ``check`` as "Earlier in this conversation:". Without them
    nothing is remembered and the prompts are those of a single question.

    Long-term memory: with a ``store`` and a ``user_id`` a ``recall`` step reads the
    definitions that user taught for this database (see ``src.sql_agent.memory``) and shows
    the relevant ones to ``generate`` and ``check`` as "Definitions this user gave in earlier
    sessions:". The agent never writes to the store.
    """
    if (thread_id is None) != (checkpointer is None):
        raise ValueError("thread_id and checkpointer go together")
    initial = SqlAgentState(
        question=question,
        evidence=evidence,
        db_path=str(db_path),
        sample_values=sample_values,
        use_check=use_check,
        max_fixes=max_fixes,
        skip_respond=skip_respond,
        policy=policy or GuardPolicy(),
        user_id=user_id,
    )
    with tracing.trace_or_span(
        "sql_agent_v2",
        input={"question": question, "evidence": evidence},
        metadata={"use_check": use_check, "max_fixes": max_fixes, "skip_respond": skip_respond},
        session_id=thread_id,
        user_id=user_id,
    ) as handle:
        out = SqlAgentState.model_validate(
            build_graph(checkpointer, store).invoke(_turn_input(initial), _config(thread_id))
        )
        handle.update(
            output={
                "final_sql": out.final_sql,
                "fixes_used": out.fixes_used,
                "attempts": len(out.attempts),
                "response": out.response,
            }
        )
    return SqlAgentResult(
        final_sql=out.final_sql,
        attempts=out.attempts,
        verdicts=out.verdicts,
        fixes_used=out.fixes_used,
        response=out.response,
        schema_context=out.schema_context,
        history=out.history,
        recalled=out.recalled,
        n_memories=out.n_memories,
    )


def _turn_input(initial: SqlAgentState) -> dict[str, Any]:
    """Every field of the state except ``history``: the turn starts clean, the conversation
    (stored by the checkpointer) is not touched."""
    return {
        name: getattr(initial, name) for name in SqlAgentState.model_fields if name != "history"
    }


def _config(thread_id: str | None) -> dict[str, Any] | None:
    return None if thread_id is None else {"configurable": {"thread_id": thread_id}}
