"""LangGraph agent definition with stateful tool execution."""

import contextlib
import logging
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agent.nodes import execute_node, reason_node, respond_node, validate_node
from src.agent.state import AgentState, ToolProfile
from src.observability.langfuse_setup import create_trace, flush

logger = logging.getLogger(__name__)


def _should_continue_execution(state: AgentState) -> str:
    """Route after execution: continue to next tool or move to validation."""
    if state.current_tool_index < len(state.planned_tools):
        return "execute"
    return "validate"


def _should_loop_or_respond(state: AgentState) -> str:
    """Route after validation: execute more tools or generate response."""
    if state.needs_more_tools:
        return "execute"
    return "respond"


def _should_reason_or_error(state: AgentState) -> str:
    """Route after reasoning: proceed to execute or end on error."""
    if state.error:
        return "respond"
    if not state.planned_tools:
        return "respond"
    return "execute"


def build_graph() -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Construct the agent's execution graph.

    Flow:
        reason → execute → (loop if more tools) → validate → (loop or respond)
        reason → respond (on error or no tools)

    Returns:
        Compiled LangGraph StateGraph ready for invocation.
    """
    graph = StateGraph(AgentState)

    graph.add_node("reason", reason_node)
    graph.add_node("execute", execute_node)
    graph.add_node("validate", validate_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("reason")

    graph.add_conditional_edges(
        "reason",
        _should_reason_or_error,
        {
            "execute": "execute",
            "respond": "respond",
        },
    )

    graph.add_conditional_edges(
        "execute",
        _should_continue_execution,
        {
            "execute": "execute",
            "validate": "validate",
        },
    )

    graph.add_conditional_edges(
        "validate",
        _should_loop_or_respond,
        {
            "execute": "execute",
            "respond": "respond",
        },
    )

    graph.add_edge("respond", END)

    return graph.compile()


_compiled_graph: CompiledStateGraph[AgentState, None, AgentState, AgentState] | None = None


def get_graph() -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Return a cached compiled graph instance."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_agent(
    query: str,
    trace_metadata: dict[str, Any] | None = None,
    *,
    schema_description: str | None = None,
    db_path: str | None = None,
    profile: ToolProfile = ToolProfile.DEFAULT,
) -> dict[str, Any]:
    """Execute the agent pipeline for a user query.

    Args:
        query: Natural language question from the user.
        trace_metadata: Optional metadata for the Langfuse trace.
        schema_description: Database schema shown to the LLM (non-default profiles).
        db_path: SQLite file for ``sql_query``; None uses the sample sales database.
        profile: Tool set and prompt wording; the default is the sales assistant.

    Returns:
        Dictionary with 'response', 'tools_used', and 'tool_results'.
    """
    trace = create_trace("agent_query", query, trace_metadata)

    initial_state = AgentState(
        query=query, schema_description=schema_description, db_path=db_path, profile=profile
    )
    if trace is not None:
        initial_state._trace = trace  # type: ignore[attr-defined]

    graph = get_graph()
    final_state = graph.invoke(initial_state)

    tools_used = []
    tool_details = []
    if isinstance(final_state, dict):
        for r in final_state.get("tool_results", []):
            if isinstance(r, dict):
                tools_used.append(r.get("tool", "unknown"))
                tool_details.append(r)
            else:
                tools_used.append(r.tool)
                tool_details.append(r.model_dump() if hasattr(r, "model_dump") else r)
        response = final_state.get("response", "")
    else:
        for r in final_state.tool_results:
            tools_used.append(r.tool)
            tool_details.append(r.model_dump())
        response = final_state.response

    if trace is not None:
        with contextlib.suppress(Exception):
            trace.update(output={"response": response, "tools_used": tools_used})

    flush()

    logger.info("Agent completed query with tools: %s", tools_used)

    return {
        "response": response,
        "tools_used": list(dict.fromkeys(tools_used)),
        "tool_results": tool_details,
    }
