"""Graph nodes for the agent: reasoning, execution, validation, and response."""

import json
import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.prompts import (
    VALIDATION_PROMPT,
    build_response_prompt,
    build_tool_selection_prompt,
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


def _format_tool_results(results: list[ToolResult]) -> str:
    """Format tool results for inclusion in prompts."""
    parts = []
    for i, r in enumerate(results, 1):
        status = "SUCCESS" if r.success else "FAILED"
        output_str = json.dumps(r.output, ensure_ascii=False, default=str)
        if len(output_str) > 1500:
            output_str = output_str[:1500] + "..."
        parts.append(f"[{i}] {r.tool}(input={r.input!r}) → {status}\n{output_str}")
    return "\n\n".join(parts)


@trace_node("reason")
def reason_node(state: AgentState) -> dict[str, Any]:
    """Analyze the user query and plan which tools to use.

    Sends the query to the LLM with tool descriptions and expects
    a JSON array of tool calls in response.
    """
    prompt = build_tool_selection_prompt(state.query, state.profile, state.schema_description)
    response_text = (
        get_client().complete("You output only valid JSON arrays.", prompt, tag="reason").text
    )

    try:
        tool_calls_raw = _extract_json(response_text)
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

    Handles PLACEHOLDER inputs by substituting values from previous
    tool results (e.g., calculator expressions that depend on SQL data).
    """
    if state.current_tool_index >= len(state.planned_tools):
        return {}

    tool_call = state.planned_tools[state.current_tool_index]
    tool_input = tool_call.input

    if "PLACEHOLDER" in tool_input and state.tool_results:
        tool_input = _resolve_placeholder(tool_input, state.tool_results)

    result = _execute_tool(tool_call.tool, tool_input, state.db_path)

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


def _resolve_placeholder(tool_input: str, results: list[ToolResult]) -> str:
    """Attempt to replace PLACEHOLDER with data from previous results.

    For calculator expressions, extracts numeric values from the most
    recent SQL query results and builds the expression.
    """
    sql_results = [r for r in results if r.tool == "sql_query" and r.success]
    if not sql_results:
        return tool_input

    last_sql = sql_results[-1]
    output = last_sql.output
    if isinstance(output, dict):
        rows = output.get("rows", [])
        if len(rows) >= 2:
            val1 = _extract_number(rows[0])
            val2 = _extract_number(rows[1])
            if val1 is not None and val2 is not None:
                return f"({val2} - {val1}) / {val1} * 100"
        elif len(rows) == 1 and len(rows[0]) >= 2:
            val1 = _extract_number_from_value(rows[0][0])
            val2 = _extract_number_from_value(rows[0][1])
            if val1 is not None and val2 is not None:
                return f"({val2} - {val1}) / {val1} * 100"

    return tool_input


def _extract_number(row: list[Any]) -> float | None:
    """Extract the first numeric value from a row."""
    for val in row:
        result = _extract_number_from_value(val)
        if result is not None:
            return result
    return None


def _extract_number_from_value(val: Any) -> float | None:
    """Try to convert a value to a float."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val.replace(",", ""))
        except ValueError:
            return None
    return None


@trace_node("validate")
def validate_node(state: AgentState) -> dict[str, Any]:
    """Check if all tools have been executed and results are sufficient.

    If there are PLACEHOLDER calculator inputs remaining, uses the LLM
    to determine what expression to compute with the actual data.
    """
    has_placeholder = any(
        tc.input == "PLACEHOLDER" and state.current_tool_index <= i
        for i, tc in enumerate(state.planned_tools)
    )

    if has_placeholder and state.tool_results:
        prompt = VALIDATION_PROMPT.format(
            query=state.query,
            planned_tools=json.dumps(
                [tc.model_dump() for tc in state.planned_tools], ensure_ascii=False
            ),
            tool_results=_format_tool_results(state.tool_results),
        )
        response_text = get_client().complete(None, prompt, tag="validate").text

        try:
            validation = _extract_json(response_text)
            if not validation.get("is_complete", True) and validation.get("additional_tools"):
                additional = []
                for tc in validation["additional_tools"]:
                    try:
                        additional.append(
                            ToolCall(
                                tool=ToolName(tc["tool"]),
                                input=tc.get("input", ""),
                                reason=tc.get("reason", ""),
                            )
                        )
                    except (ValueError, KeyError):
                        continue

                if additional:
                    return {
                        "planned_tools": list(state.planned_tools) + additional,
                        "needs_more_tools": True,
                    }
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse validation response: %s", response_text[:200])

    all_executed = state.current_tool_index >= len(state.planned_tools)
    return {"needs_more_tools": not all_executed}


@trace_node("respond")
def respond_node(state: AgentState) -> dict[str, Any]:
    """Generate the final response using tool results.

    Sends all collected tool results to the LLM along with the
    original query to produce a natural language answer.
    """
    neutral = state.profile is not ToolProfile.DEFAULT

    if state.error:
        if neutral:
            return {
                "response": f"Sorry, there was an error processing your question: {state.error}"
            }
        return {"response": f"Lo siento, hubo un error procesando tu pregunta: {state.error}"}

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
