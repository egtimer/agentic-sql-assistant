"""Pydantic models for API request and response validation."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for the /agent/query endpoint."""

    query: str = Field(
        min_length=1,
        max_length=1000,
        description="Natural language question to the agent",
        examples=["¿Cuánto vendimos en Q3 2024?"],
    )
    session_id: str | None = Field(
        default=None,
        max_length=200,
        description="Optional session id: groups the traces of one conversation in Langfuse",
    )
    user_id: str | None = Field(
        default=None,
        max_length=200,
        description="Optional user id recorded on the Langfuse trace",
    )


class ToolDetail(BaseModel):
    """Details of a single tool execution."""

    tool: str = Field(description="Tool name that was executed")
    input: str = Field(description="Input passed to the tool")
    success: bool = Field(description="Whether the tool succeeded")


class QueryResponse(BaseModel):
    """Response body from the /agent/query endpoint."""

    query: str = Field(description="Original query")
    response: str = Field(description="Agent's natural language response")
    tools_used: list[str] = Field(description="List of tool names used")
    tool_details: list[ToolDetail] = Field(
        default_factory=list,
        description="Execution details for each tool call",
    )
    trace_id: str | None = Field(
        default=None,
        description="Langfuse trace id of this request (null when tracing is not configured)",
    )


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str = Field(description="Service status", examples=["healthy"])
    version: str = Field(description="Application version")
    services: dict[str, str] = Field(
        default_factory=dict,
        description="Status of dependent services",
    )


class ErrorResponse(BaseModel):
    """Standard error response body."""

    detail: str = Field(description="Error description")
