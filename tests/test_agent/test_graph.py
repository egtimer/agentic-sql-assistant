"""Tests for the LangGraph agent graph structure and execution."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agent.graph import build_graph
from src.agent.nodes import (
    _execute_tool,
    _extract_json,
    _format_tool_results,
    _resolve_placeholder,
    execute_node,
    reason_node,
    respond_node,
    validate_node,
)
from src.agent.state import AgentState, ToolCall, ToolName, ToolResult
from src.llm import CallUsage, Completion


def _mock_client(*texts: str) -> MagicMock:
    """Client whose ``complete`` returns the given texts in order (the last one repeats)."""
    client = MagicMock()
    completions = [Completion(text=t, usage=CallUsage(tag="test")) for t in texts]
    client.complete.side_effect = completions + [completions[-1]] * 10
    return client


class TestGraphStructure:
    def test_graph_compiles(self) -> None:
        graph = build_graph()
        assert graph is not None

    def test_graph_has_all_nodes(self) -> None:
        graph = build_graph()
        node_names = set(graph.get_graph().nodes.keys())
        assert "reason" in node_names
        assert "execute" in node_names
        assert "validate" in node_names
        assert "respond" in node_names


class TestExtractJson:
    def test_plain_json_array(self) -> None:
        text = '[{"tool": "sql_query", "input": "SELECT 1", "reason": "test"}]'
        result = _extract_json(text)
        assert isinstance(result, list)
        assert result[0]["tool"] == "sql_query"

    def test_json_in_markdown_block(self) -> None:
        text = '```json\n[{"tool": "calculator", "input": "2+2", "reason": "math"}]\n```'
        result = _extract_json(text)
        assert isinstance(result, list)
        assert result[0]["tool"] == "calculator"

    def test_json_with_surrounding_text(self) -> None:
        text = (
            "Here is the plan:\n"
            '[{"tool": "doc_search", "input": "policy", "reason": "search"}]\n'
            "Done."
        )
        result = _extract_json(text)
        assert isinstance(result, list)

    def test_json_object(self) -> None:
        text = '{"is_complete": true, "reason": "done", "additional_tools": []}'
        result = _extract_json(text)
        assert isinstance(result, dict)
        assert result["is_complete"] is True

    def test_invalid_json_raises(self) -> None:
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _extract_json("not json at all")


class TestFormatToolResults:
    def test_format_single_result(self) -> None:
        results = [
            ToolResult(
                tool="sql_query",
                input="SELECT SUM(amount) FROM sales",
                output={"rows": [[234500]], "columns": ["sum"]},
                success=True,
            )
        ]
        formatted = _format_tool_results(results)
        assert "sql_query" in formatted
        assert "SUCCESS" in formatted
        assert "234500" in formatted

    def test_format_failed_result(self) -> None:
        results = [
            ToolResult(
                tool="calculator", input="1/0", output={"error": "div by zero"}, success=False
            )
        ]
        formatted = _format_tool_results(results)
        assert "FAILED" in formatted

    def test_format_truncates_long_output(self) -> None:
        results = [
            ToolResult(
                tool="sql_query", input="SELECT *", output={"data": "x" * 2000}, success=True
            )
        ]
        formatted = _format_tool_results(results)
        assert "..." in formatted


class TestResolvePlaceholder:
    def test_replaces_with_two_row_values(self) -> None:
        results = [
            ToolResult(
                tool="sql_query",
                input="SELECT quarter, SUM(amount) FROM sales GROUP BY quarter",
                output={"rows": [["Q2", 198000], ["Q3", 234500]], "columns": ["quarter", "sum"]},
                success=True,
            )
        ]
        resolved = _resolve_placeholder("PLACEHOLDER", results)
        assert "198000" in resolved
        assert "234500" in resolved
        assert "/" in resolved

    def test_keeps_placeholder_without_sql_results(self) -> None:
        resolved = _resolve_placeholder("PLACEHOLDER", [])
        assert resolved == "PLACEHOLDER"

    def test_keeps_placeholder_with_failed_sql(self) -> None:
        results = [ToolResult(tool="sql_query", input="bad query", output={}, success=False)]
        resolved = _resolve_placeholder("PLACEHOLDER", results)
        assert resolved == "PLACEHOLDER"


class TestExecuteTool:
    def test_execute_calculator(self) -> None:
        result = _execute_tool(ToolName.CALCULATOR, "2 + 3")
        assert result.success is True
        assert result.output["result"] == 5

    def test_execute_sql_query(self, tmp_path) -> None:
        import sqlite3

        db = tmp_path / "test.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'hello')")
        conn.commit()
        conn.close()

        with patch("src.agent.nodes.sql_query") as mock_sql:
            from src.tools.sql_query import SQLQueryResult

            mock_sql.return_value = SQLQueryResult(
                query="SELECT * FROM t",
                columns=["id", "val"],
                rows=[[1, "hello"]],
                row_count=1,
                success=True,
            )
            result = _execute_tool(ToolName.SQL_QUERY, "SELECT * FROM t")
            assert result.success is True

    def test_execute_doc_search_error(self) -> None:
        with patch("src.agent.nodes.doc_search") as mock_ds:
            from src.tools.doc_search import DocSearchResult

            mock_ds.return_value = DocSearchResult(
                query="test",
                chunks=[],
                chunk_count=0,
                success=False,
                error="Connection refused",
            )
            result = _execute_tool(ToolName.DOC_SEARCH, "test query")
            assert result.success is False


class TestExecuteNode:
    def test_executes_planned_tool(self) -> None:
        state = AgentState(
            query="test",
            planned_tools=[
                ToolCall(tool=ToolName.CALCULATOR, input="10 + 20", reason="add numbers"),
            ],
            current_tool_index=0,
        )
        updates = execute_node(state)
        assert len(updates["tool_results"]) == 1
        assert updates["tool_results"][0].success is True
        assert updates["current_tool_index"] == 1

    def test_skips_when_index_past_end(self) -> None:
        state = AgentState(
            query="test",
            planned_tools=[
                ToolCall(tool=ToolName.CALCULATOR, input="1+1", reason="done"),
            ],
            current_tool_index=1,
        )
        updates = execute_node(state)
        assert updates == {}


class TestReasonNode:
    @patch("src.agent.nodes.get_client")
    def test_plans_single_sql_tool(self, mock_get_client: MagicMock) -> None:
        mock_get_client.return_value = _mock_client(
            '[{"tool": "sql_query", '
            '"input": "SELECT SUM(amount) FROM sales WHERE quarter = \'Q3\'", '
            '"reason": "Get Q3 sales"}]'
        )

        state = AgentState(query="¿Cuánto vendimos en Q3?")
        updates = reason_node(state)

        assert len(updates["planned_tools"]) == 1
        assert updates["planned_tools"][0].tool == ToolName.SQL_QUERY
        system, user = mock_get_client.return_value.complete.call_args.args
        assert system == "You output only valid JSON arrays."
        assert "¿Cuánto vendimos en Q3?" in user
        assert mock_get_client.return_value.complete.call_args.kwargs == {"tag": "reason"}
        assert updates["messages"][0].content == "¿Cuánto vendimos en Q3?"
        assert updates["messages"][1].type == "ai"

    @patch("src.agent.nodes.get_client")
    def test_plans_multiple_tools(self, mock_get_client: MagicMock) -> None:
        mock_get_client.return_value = _mock_client(
            '[{"tool": "sql_query", "input": "SELECT quarter, SUM(amount) FROM sales '
            'WHERE quarter IN (\'Q2\',\'Q3\') GROUP BY quarter", "reason": "get totals"}, '
            '{"tool": "calculator", "input": "PLACEHOLDER", "reason": "calculate growth"}]'
        )

        state = AgentState(query="¿Crecimiento Q3 vs Q2?")
        updates = reason_node(state)

        assert len(updates["planned_tools"]) == 2
        assert updates["planned_tools"][0].tool == ToolName.SQL_QUERY
        assert updates["planned_tools"][1].tool == ToolName.CALCULATOR

    @patch("src.agent.nodes.get_client")
    def test_handles_invalid_json(self, mock_get_client: MagicMock) -> None:
        mock_get_client.return_value = _mock_client("I don't know what to do")

        state = AgentState(query="test")
        updates = reason_node(state)

        assert "error" in updates


class TestRespondNode:
    @patch("src.agent.nodes.get_client")
    def test_generates_response(self, mock_get_client: MagicMock) -> None:
        mock_get_client.return_value = _mock_client("Las ventas de Q3 fueron €234,500.")

        state = AgentState(
            query="¿Cuánto vendimos en Q3?",
            tool_results=[
                ToolResult(
                    tool="sql_query",
                    input="SELECT SUM(amount) FROM sales",
                    output={"rows": [[234500]]},
                    success=True,
                ),
            ],
        )
        updates = respond_node(state)
        assert "234,500" in updates["response"]

    def test_responds_with_error_message(self) -> None:
        state = AgentState(
            query="test",
            error="Something went wrong",
        )
        updates = respond_node(state)
        assert "error" in updates["response"].lower()

    def test_responds_when_no_results(self) -> None:
        state = AgentState(query="test")
        updates = respond_node(state)
        assert "no pude" in updates["response"].lower()


class TestValidateNode:
    def test_complete_when_all_executed(self) -> None:
        state = AgentState(
            query="test",
            planned_tools=[
                ToolCall(tool=ToolName.CALCULATOR, input="2+2", reason="test"),
            ],
            tool_results=[
                ToolResult(tool="calculator", input="2+2", output={"result": 4}, success=True),
            ],
            current_tool_index=1,
        )
        updates = validate_node(state)
        assert updates["needs_more_tools"] is False

    def test_needs_more_when_tools_remain(self) -> None:
        state = AgentState(
            query="test",
            planned_tools=[
                ToolCall(tool=ToolName.SQL_QUERY, input="SELECT 1", reason="first"),
                ToolCall(tool=ToolName.CALCULATOR, input="2+2", reason="second"),
            ],
            tool_results=[
                ToolResult(
                    tool="sql_query", input="SELECT 1", output={"rows": [[1]]}, success=True
                ),
            ],
            current_tool_index=1,
        )
        updates = validate_node(state)
        assert updates["needs_more_tools"] is True
