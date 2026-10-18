"""Tests for agent routing logic and conditional edges."""

from src.agent.graph import (
    _should_continue_execution,
    _should_loop_or_respond,
    _should_reason_or_error,
)
from src.agent.state import AgentState, ToolCall, ToolName


class TestReasonRouting:
    def test_routes_to_execute_on_valid_plan(self) -> None:
        state = AgentState(
            query="test",
            planned_tools=[
                ToolCall(tool=ToolName.SQL_QUERY, input="SELECT 1", reason="test"),
            ],
        )
        assert _should_reason_or_error(state) == "execute"

    def test_routes_to_respond_on_error(self) -> None:
        state = AgentState(query="test", error="parse failed")
        assert _should_reason_or_error(state) == "respond"

    def test_routes_to_respond_on_empty_plan(self) -> None:
        state = AgentState(query="test", planned_tools=[])
        assert _should_reason_or_error(state) == "respond"


class TestExecutionRouting:
    def test_continues_when_tools_remain(self) -> None:
        state = AgentState(
            query="test",
            planned_tools=[
                ToolCall(tool=ToolName.SQL_QUERY, input="q1", reason="first"),
                ToolCall(tool=ToolName.CALCULATOR, input="1+1", reason="second"),
            ],
            current_tool_index=1,
        )
        assert _should_continue_execution(state) == "execute"

    def test_validates_when_all_done(self) -> None:
        state = AgentState(
            query="test",
            planned_tools=[
                ToolCall(tool=ToolName.SQL_QUERY, input="q1", reason="only"),
            ],
            current_tool_index=1,
        )
        assert _should_continue_execution(state) == "validate"


class TestValidationRouting:
    def test_responds_when_complete(self) -> None:
        state = AgentState(query="test", needs_more_tools=False)
        assert _should_loop_or_respond(state) == "respond"

    def test_loops_when_more_tools_needed(self) -> None:
        state = AgentState(query="test", needs_more_tools=True)
        assert _should_loop_or_respond(state) == "execute"


class TestEndToEndRouting:
    """Verify the full routing sequence for different query types."""

    def test_single_tool_sequence(self) -> None:
        state = AgentState(
            query="¿Cuánto vendimos en Q3?",
            planned_tools=[
                ToolCall(
                    tool=ToolName.SQL_QUERY, input="SELECT SUM(amount) FROM sales", reason="sales"
                ),
            ],
            current_tool_index=0,
        )
        assert _should_reason_or_error(state) == "execute"

        state.current_tool_index = 1
        assert _should_continue_execution(state) == "validate"

        state.needs_more_tools = False
        assert _should_loop_or_respond(state) == "respond"

    def test_multi_tool_sequence(self) -> None:
        state = AgentState(
            query="¿Crecimiento Q3 vs Q2?",
            planned_tools=[
                ToolCall(tool=ToolName.SQL_QUERY, input="SELECT ...", reason="get data"),
                ToolCall(tool=ToolName.CALCULATOR, input="PLACEHOLDER", reason="calc"),
            ],
            current_tool_index=0,
        )
        assert _should_reason_or_error(state) == "execute"

        state.current_tool_index = 1
        assert _should_continue_execution(state) == "execute"

        state.current_tool_index = 2
        assert _should_continue_execution(state) == "validate"

        state.needs_more_tools = False
        assert _should_loop_or_respond(state) == "respond"
