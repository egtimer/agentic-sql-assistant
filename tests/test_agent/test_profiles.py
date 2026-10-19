"""Tests for per-run schema, database and tool-set configuration."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.agent.graph import run_agent
from src.agent.nodes import execute_node, reason_node, respond_node
from src.agent.prompts import (
    RESPONSE_PROMPT,
    TOOL_SELECTION_PROMPT,
    build_response_prompt,
    build_tool_selection_parts,
    build_tool_selection_prompt,
)
from src.agent.state import AgentState, ToolCall, ToolName, ToolProfile, ToolResult
from src.llm import CallUsage, Completion

SNAPSHOTS = Path(__file__).parent / "snapshots"
SALES_QUERY = "¿Cuánto vendimos en Q3 2024?"
SCHEMA = "CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)"


def _client(*texts: str) -> MagicMock:
    client = MagicMock()
    completions = [Completion(text=t, usage=CallUsage(tag="test")) for t in texts]
    client.complete.side_effect = completions + [completions[-1]] * 10
    return client


class TestDefaultPromptsUnchanged:
    def test_tool_selection_matches_snapshot(self) -> None:
        expected = (SNAPSHOTS / "tool_selection_default.txt").read_text(encoding="utf-8")
        assert build_tool_selection_prompt(SALES_QUERY) == expected
        assert TOOL_SELECTION_PROMPT.format(query=SALES_QUERY) == expected

    def test_default_ignores_schema_argument(self) -> None:
        expected = (SNAPSHOTS / "tool_selection_default.txt").read_text(encoding="utf-8")
        assert build_tool_selection_prompt(SALES_QUERY, ToolProfile.DEFAULT, SCHEMA) == expected

    def test_response_matches_snapshot(self) -> None:
        expected = (SNAPSHOTS / "response_default.txt").read_text(encoding="utf-8")
        tool_results = "[1] sql_query(input='SELECT 1') → SUCCESS\n{}"
        assert build_response_prompt(SALES_QUERY, tool_results) == expected
        assert RESPONSE_PROMPT.format(query=SALES_QUERY, tool_results=tool_results) == expected


class TestSqlOnlyPrompt:
    def _prompt(self, query: str = "How many rows?\nExternal knowledge: none") -> str:
        return build_tool_selection_prompt(query, ToolProfile.SQL_ONLY, SCHEMA)

    def test_lists_only_sql_and_calculator(self) -> None:
        prompt = self._prompt()
        assert "**sql_query**" in prompt
        assert "**calculator**" in prompt
        assert "doc_search" not in prompt
        assert 'one of "sql_query", "calculator"' in prompt

    def test_no_sales_or_spanish_content(self) -> None:
        prompt = self._prompt()
        for word in ("sales", "customers", "Spanish", "política", "Cuánto", "Q3", "revenue"):
            assert word not in prompt

    def test_keeps_json_contract_and_placeholder_rule(self) -> None:
        prompt = self._prompt()
        assert "Respond ONLY with a JSON array" in prompt
        assert '"PLACEHOLDER"' in prompt
        assert '"reason"' in prompt

    def test_stable_content_first_and_question_last(self) -> None:
        prompt = self._prompt("Q-MARKER\nExternal knowledge: E-MARKER")
        assert prompt.index(SCHEMA) < prompt.index("Q-MARKER")
        assert prompt.endswith("User question: Q-MARKER\nExternal knowledge: E-MARKER")
        assert prompt.count("Q-MARKER") == 1

    def test_prefix_is_identical_across_questions(self) -> None:
        a = build_tool_selection_prompt("question A", ToolProfile.SQL_ONLY, SCHEMA)
        b = build_tool_selection_prompt("question B", ToolProfile.SQL_ONLY, SCHEMA)
        common = a[: a.index("question A")]
        assert b.startswith(common)

    def test_braces_in_schema_and_query_are_safe(self) -> None:
        prompt = build_tool_selection_prompt("q {x}", ToolProfile.SQL_ONLY, "CREATE TABLE a{b}")
        assert "CREATE TABLE a{b}" in prompt
        assert "q {x}" in prompt

    def test_parts_concatenate_to_prompt(self) -> None:
        for query in ("How many rows?", "q {x}", "User question: tricky\nUser question: again"):
            prefix, user = build_tool_selection_parts(query, ToolProfile.SQL_ONLY, SCHEMA)
            assert prefix is not None
            assert prefix + user == build_tool_selection_prompt(query, ToolProfile.SQL_ONLY, SCHEMA)
            assert prefix.endswith("\n\n")
            assert user.startswith("User question: ")
            assert SCHEMA in prefix

    def test_prefix_does_not_depend_on_query(self) -> None:
        a, _ = build_tool_selection_parts("question A", ToolProfile.SQL_ONLY, SCHEMA)
        b, _ = build_tool_selection_parts("question B", ToolProfile.SQL_ONLY, SCHEMA)
        assert a == b

    def test_default_profile_has_no_prefix(self) -> None:
        prefix, user = build_tool_selection_parts(SALES_QUERY)
        assert prefix is None
        assert user == build_tool_selection_prompt(SALES_QUERY)

    def test_neutral_response_prompt(self) -> None:
        prompt = build_response_prompt("q", "results", ToolProfile.SQL_ONLY)
        assert "€" not in prompt
        assert "según" not in prompt


class TestReasonNodeProfiles:
    @patch("src.agent.nodes.get_client")
    def test_sql_only_prompt_sent_with_schema(self, mock_get_client: MagicMock) -> None:
        mock_get_client.return_value = _client(
            '[{"tool": "sql_query", "input": "SELECT 1", "reason": "r"}]'
        )
        state = AgentState(query="q", profile=ToolProfile.SQL_ONLY, schema_description=SCHEMA)
        updates = reason_node(state)
        call = mock_get_client.return_value.complete.call_args
        assert SCHEMA in call.kwargs["prefix"]
        assert call.kwargs["prefix"] + call.args[1] == build_tool_selection_prompt(
            "q", ToolProfile.SQL_ONLY, SCHEMA
        )
        assert len(updates["planned_tools"]) == 1

    @patch("src.agent.nodes.get_client")
    def test_sql_only_drops_disallowed_tools(self, mock_get_client: MagicMock) -> None:
        mock_get_client.return_value = _client(
            '[{"tool": "doc_search", "input": "x", "reason": "r"},'
            ' {"tool": "sql_query", "input": "SELECT 1", "reason": "r"}]'
        )
        state = AgentState(query="q", profile=ToolProfile.SQL_ONLY, schema_description=SCHEMA)
        updates = reason_node(state)
        assert [t.tool for t in updates["planned_tools"]] == [ToolName.SQL_QUERY]

    @patch("src.agent.nodes.get_client")
    def test_default_keeps_all_tools(self, mock_get_client: MagicMock) -> None:
        mock_get_client.return_value = _client(
            '[{"tool": "doc_search", "input": "x", "reason": "r"}]'
        )
        updates = reason_node(AgentState(query="q"))
        assert [t.tool for t in updates["planned_tools"]] == [ToolName.DOC_SEARCH]
        call = mock_get_client.return_value.complete.call_args
        assert call.args[1] == TOOL_SELECTION_PROMPT.format(query="q")
        assert "prefix" not in call.kwargs


class TestExecuteNodeDbPath:
    def test_passes_db_path_when_set(self) -> None:
        state = AgentState(
            query="q",
            db_path="/some/db.sqlite",
            planned_tools=[ToolCall(tool=ToolName.SQL_QUERY, input="SELECT 1", reason="r")],
        )
        with patch("src.agent.nodes.sql_query") as mock_sql:
            mock_sql.return_value.model_dump.return_value = {}
            mock_sql.return_value.success = True
            execute_node(state)
        mock_sql.assert_called_once_with("SELECT 1", "/some/db.sqlite")

    def test_default_call_unchanged(self) -> None:
        state = AgentState(
            query="q",
            planned_tools=[ToolCall(tool=ToolName.SQL_QUERY, input="SELECT 1", reason="r")],
        )
        with patch("src.agent.nodes.sql_query") as mock_sql:
            mock_sql.return_value.model_dump.return_value = {}
            mock_sql.return_value.success = True
            execute_node(state)
        mock_sql.assert_called_once_with("SELECT 1")


class TestRespondNodeProfiles:
    def test_default_messages_stay_in_spanish(self) -> None:
        assert "Lo siento" in respond_node(AgentState(query="q", error="boom"))["response"]
        assert "No pude" in respond_node(AgentState(query="q"))["response"]

    def test_neutral_messages_for_sql_only(self) -> None:
        err = respond_node(AgentState(query="q", error="boom", profile=ToolProfile.SQL_ONLY))
        none = respond_node(AgentState(query="q", profile=ToolProfile.SQL_ONLY))
        assert "Sorry" in err["response"]
        assert "could not find" in none["response"]

    @patch("src.agent.nodes.get_client")
    def test_neutral_prompt_for_sql_only(self, mock_get_client: MagicMock) -> None:
        mock_get_client.return_value = _client("done")
        state = AgentState(
            query="q",
            profile=ToolProfile.SQL_ONLY,
            tool_results=[ToolResult(tool="sql_query", input="s", output={}, success=True)],
        )
        respond_node(state)
        assert "€" not in mock_get_client.return_value.complete.call_args.args[1]


class TestRunAgentEndToEnd:
    @patch("src.agent.nodes.get_client")
    def test_sql_only_run_on_given_database(
        self, mock_get_client: MagicMock, tmp_path: Path
    ) -> None:
        db = tmp_path / "other.sqlite"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'hello')")
        conn.commit()
        conn.close()

        mock_get_client.return_value = _client(
            '[{"tool": "sql_query", "input": "SELECT val FROM t", "reason": "r"}]',
            "The value is hello.",
        )
        result = run_agent(
            "What is the value?",
            schema_description=SCHEMA,
            db_path=str(db),
            profile=ToolProfile.SQL_ONLY,
        )
        assert result["tools_used"] == ["sql_query"]
        tool_result = result["tool_results"][0]
        assert tool_result["success"] is True
        assert tool_result["input"] == "SELECT val FROM t"
        assert tool_result["output"]["rows"] == [["hello"]]
        assert result["response"] == "The value is hello."
