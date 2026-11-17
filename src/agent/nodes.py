"""Graph nodes for the agent: reasoning, execution, validation, and response."""

import json
import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.prompts import (
    CALC_FILL_PROMPT,
    DIRECT_RESPONSE_PROMPT,
    build_response_prompt,
    build_tool_selection_parts,
)
from src.agent.state import (
    PROFILE_TOOLS,
    AgentState,
    ToolCall,
    ToolName,
    ToolProfile,
    ToolResult,
)
from src.llm import get_client
from src.observability import tracing
from src.observability.langfuse_setup import trace_node
from src.tools.calculator import calculate
from src.tools.doc_search import doc_search
from src.tools.sql_query import sql_query

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> Any:
    """Extract JSON from LLM output, handling markdown code blocks."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()

    candidates = []
    for start, end in [("[", "]"), ("{", "}")]:
        idx_start = text.find(start)
        idx_end = text.rfind(end)
        if idx_start != -1 and idx_end != -1 and idx_end > idx_start:
            candidates.append((idx_start, text[idx_start : idx_end + 1]))

    candidates.sort(key=lambda c: c[0])
    for _, candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return json.loads(text)


# Characters of each tool's output kept in the prompts. Was 1,500 until the 2026-11-17 failure
# analysis found the passage with the answer cut off at about character 1,800 (ADR-011).
MAX_TOOL_OUTPUT_CHARS = 6000


def _format_tool_results(results: list[ToolResult]) -> str:
    """Format tool results for inclusion in prompts, saying when an output was cut."""
    parts = []
    for i, r in enumerate(results, 1):
        status = "SUCCESS" if r.success else "FAILED"
        output_str = json.dumps(r.output, ensure_ascii=False, default=str)
        if len(output_str) > MAX_TOOL_OUTPUT_CHARS:
            cut = len(output_str) - MAX_TOOL_OUTPUT_CHARS
            output_str = (
                f"{output_str[:MAX_TOOL_OUTPUT_CHARS]}... [{cut} more characters not shown]"
            )
        parts.append(f"[{i}] {r.tool}(input={r.input!r}) → {status}\n{output_str}")
    return "\n\n".join(parts)


@trace_node("reason")
def reason_node(state: AgentState) -> dict[str, Any]:
    """Analyze the user query and plan which tools to use.

    Sends the query to the LLM with tool descriptions and expects
    a JSON array of tool calls in response.
    """
    prefix, prompt = build_tool_selection_parts(
        state.query, state.profile, state.schema_description
    )
    # The default profile's call stays exactly as it was: no prefix argument at all.
    extra = {} if prefix is None else {"prefix": prefix}
    response_text = (
        get_client()
        .complete("You output only valid JSON arrays.", prompt, tag="reason", **extra)
        .text
    )

    try:
        tool_calls_raw = _extract_json(response_text)
        if tool_calls_raw == [] and state.profile is ToolProfile.DEFAULT:
            # an explicit empty plan: the question needs no tool
            logger.info("No tool needed for query: %s", state.query[:80])
            return {
                "planned_tools": [],
                "direct_answer": True,
                "messages": [HumanMessage(content=state.query), AIMessage(content=response_text)],
            }
        if not isinstance(tool_calls_raw, list):
            tool_calls_raw = [tool_calls_raw]

        allowed = PROFILE_TOOLS[state.profile]
        planned = []
        for tc in tool_calls_raw:
            try:
                call = ToolCall(
                    tool=ToolName(tc["tool"]),
                    input=tc.get("input", ""),
                    reason=tc.get("reason", ""),
                )
                if call.tool not in allowed:
                    logger.warning("Skipping tool %s: not in profile %s", call.tool, state.profile)
                    continue
                planned.append(call)
            except (ValueError, KeyError) as e:
                logger.warning("Skipping invalid tool call %s: %s", tc, e)

        if not planned:
            return {
                "error": f"No valid tools selected. LLM output: {response_text[:200]}",
                "planned_tools": [],
            }

        logger.info("Planned %d tool(s) for query: %s", len(planned), state.query[:80])
        return {
            "planned_tools": planned,
            "messages": [HumanMessage(content=state.query), AIMessage(content=response_text)],
        }

    except (json.JSONDecodeError, TypeError) as e:
        logger.error("Failed to parse tool selection: %s — Output: %s", e, response_text[:200])
        return {
            "error": f"Failed to parse tool plan: {e}",
            "planned_tools": [],
        }


@trace_node("execute")
def execute_node(state: AgentState) -> dict[str, Any]:
    """Execute the next planned tool and store its result.

    A planned input of ``PLACEHOLDER`` (a calculation that depends on earlier results) is
    written by the model from those results just before the tool runs (``_fill_placeholder``).
    """
    if state.current_tool_index >= len(state.planned_tools):
        return {}

    tool_call = state.planned_tools[state.current_tool_index]
    tool_input = tool_call.input

    if "PLACEHOLDER" in tool_input:
        filled = _fill_placeholder(state.query, tool_call, state.tool_results)
        if filled is None:
            result = ToolResult(
                tool=tool_call.tool.value,
                input=tool_input,
                output={"error": "no expression: the earlier results lack the numbers needed"},
                success=False,
            )
            return {
                "tool_results": [*state.tool_results, result],
                "current_tool_index": state.current_tool_index + 1,
            }
        tool_input = filled

    with tracing.span(f"{tracing.TOOL_SPAN_PREFIX}{tool_call.tool.value}") as tool_span:
        result = _execute_tool(tool_call.tool, tool_input, state.db_path)
        if not result.success:
            tool_span.update(level="ERROR", status_message="ToolError: the tool reported a failure")

    logger.info(
        "Executed %s (success=%s): %s",
        tool_call.tool.value,
        result.success,
        tool_input[:80],
    )

    new_results = list(state.tool_results) + [result]
    return {
        "tool_results": new_results,
        "current_tool_index": state.current_tool_index + 1,
    }


def _execute_tool(tool_name: ToolName, tool_input: str, db_path: str | None = None) -> ToolResult:
    """Dispatch to the appropriate tool function."""
    try:
        if tool_name == ToolName.SQL_QUERY:
            output = sql_query(tool_input) if db_path is None else sql_query(tool_input, db_path)
            return ToolResult(
                tool=tool_name.value,
                input=tool_input,
                output=output.model_dump(),
                success=output.success,
            )
        elif tool_name == ToolName.DOC_SEARCH:
            doc_output = doc_search(tool_input)
            return ToolResult(
                tool=tool_name.value,
                input=tool_input,
                output=doc_output.model_dump(),
                success=doc_output.success,
            )
        elif tool_name == ToolName.CALCULATOR:
            calc_output = calculate(tool_input)
            return ToolResult(
                tool=tool_name.value,
                input=tool_input,
                output=calc_output.model_dump(),
                success=calc_output.success,
            )
        else:
            return ToolResult(
                tool=tool_name.value,
                input=tool_input,
                output=None,
                success=False,
            )
    except Exception as e:
        logger.exception("Tool execution failed: %s", tool_name.value)
        return ToolResult(
            tool=tool_name.value,
            input=tool_input,
            output={"error": str(e)},
            success=False,
        )


_EXPRESSION = re.compile(r"^[\d\s.+\-*/()]+$")


def _fill_placeholder(query: str, tool_call: ToolCall, results: list[ToolResult]) -> str | None:
    """Ask the model for the calculator expression a ``PLACEHOLDER`` stands for.

    Replaces a fixed growth-rate formula that was applied to every question: the 2026-11-17
    failure analysis found it computed an unrelated number for 5 of the 6 question types that
    reached it, and left the placeholder unfilled for the rest (ADR-011). Returns None when
    there are no earlier results or the model says the numbers are missing.
    """
    if not results:
        return None
    prompt = CALC_FILL_PROMPT.format(
        query=query,
        reason=tool_call.reason or "the calculation",
        tool_results=_format_tool_results(results),
    )
    text = get_client().complete(None, prompt, tag="fill").text.strip()
    expression = text.strip("`").strip().splitlines()[0].strip() if text.strip("`").strip() else ""
    if not expression or expression.upper() == "NONE" or not _EXPRESSION.match(expression):
        return None
    return expression


@trace_node("validate")
def validate_node(state: AgentState) -> dict[str, Any]:
    """Route back to ``execute`` while planned tools remain, else on to ``respond``.

    It used to ask the model to fill a ``PLACEHOLDER`` calculator input here, but ``execute``
    runs every planned tool before this node, so that path never ran; the placeholder is now
    filled where the tool runs (``_fill_placeholder``, ADR-011).
    """
    return {"needs_more_tools": state.current_tool_index < len(state.planned_tools)}


@trace_node("respond")
def respond_node(state: AgentState) -> dict[str, Any]:
    """Generate the final response using tool results.

    Sends all collected tool results to the LLM along with the
    original query to produce a natural language answer.
    """
    if state.skip_respond:
        return {"response": ""}

    neutral = state.profile is not ToolProfile.DEFAULT

    if state.error:
        if neutral:
            return {
                "response": f"Sorry, there was an error processing your question: {state.error}"
            }
        return {"response": f"Lo siento, hubo un error procesando tu pregunta: {state.error}"}

    if state.direct_answer:
        prompt = DIRECT_RESPONSE_PROMPT.format(query=state.query)
        return {"response": get_client().complete(None, prompt, tag="respond").text}

    if not state.tool_results:
        if neutral:
            return {"response": "I could not find relevant information to answer your question."}
        return {"response": "No pude encontrar información relevante para responder tu pregunta."}

    prompt = build_response_prompt(
        state.query, _format_tool_results(state.tool_results), state.profile
    )

    response_text = get_client().complete(None, prompt, tag="respond").text

    tools_used = list({r.tool for r in state.tool_results})
    response_with_trace = response_text.strip()

    logger.info("Response generated using tools: %s", tools_used)

    return {"response": response_with_trace}
