"""State schema for the LangGraph agent."""

from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class ToolName(str, Enum):
    """Available tools the agent can invoke."""

    SQL_QUERY = "sql_query"
    DOC_SEARCH = "doc_search"
    CALCULATOR = "calculator"


class ToolCall(BaseModel):
    """A single tool invocation planned by the agent."""

    tool: ToolName = Field(description="Which tool to use")
    input: str = Field(description="Input argument for the tool")
    reason: str = Field(description="Why this tool was selected")


class ToolResult(BaseModel):
    """Result from executing a single tool."""

    tool: str = Field(description="Name of the tool that was executed")
    input: str = Field(description="Input that was passed to the tool")
    output: Any = Field(description="Raw output from the tool")
    success: bool = Field(description="Whether the tool executed successfully")


class AgentState(BaseModel):
    """Complete state for a single agent invocation.

    This state flows through all nodes in the graph, accumulating
    tool calls, results, and the final response.
    """

    query: str = Field(description="Original user question")
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    planned_tools: list[ToolCall] = Field(
        default_factory=list,
        description="Tools the agent plans to execute",
    )
    tool_results: list[ToolResult] = Field(
        default_factory=list,
        description="Results from executed tools",
    )
    current_tool_index: int = Field(
        default=0,
        description="Index of the next tool to execute",
    )
    response: str = Field(
        default="",
        description="Final response to the user",
    )
    needs_more_tools: bool = Field(
        default=False,
        description="Whether the agent needs additional tool calls after validation",
    )
    error: str | None = Field(
        default=None,
        description="Error message if the agent failed",
    )
