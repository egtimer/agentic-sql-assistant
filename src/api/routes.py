"""API route definitions."""

import logging
import time

from fastapi import APIRouter, HTTPException

from src.agent.graph import run_agent
from src.api.schemas import (
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ToolDetail,
)
from src.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns service status and dependency health.",
)
async def health_check() -> HealthResponse:
    """Check application health and connectivity to dependencies."""
    services: dict[str, str] = {}

    try:
        import httpx

        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            services["ollama"] = "healthy" if resp.status_code == 200 else "unhealthy"
    except Exception:
        services["ollama"] = "unavailable"

    try:
        import httpx

        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{settings.qdrant_url}/collections")
            services["qdrant"] = "healthy" if resp.status_code == 200 else "unhealthy"
    except Exception:
        services["qdrant"] = "unavailable"

    from pathlib import Path

    db_path = Path(settings.sqlite_db_path)
    services["sqlite"] = "healthy" if db_path.exists() else "missing"

    return HealthResponse(
        status="healthy",
        version="0.1.0",
        services=services,
    )


@router.post(
    "/agent/query",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Query the agent",
    description=(
        "Submit a natural language question. The agent reasons about which tools "
        "to use and returns a traced response."
    ),
)
async def agent_query(request: QueryRequest) -> QueryResponse:
    """Process a user query through the agent pipeline."""
    start_time = time.perf_counter()
    logger.info("Received query: %s", request.query[:100])

    try:
        result = run_agent(
            query=request.query,
            trace_metadata={"source": "api"},
        )

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Query processed in %.0fms, tools: %s",
            duration_ms,
            result.get("tools_used", []),
        )

        tool_details = []
        for tr in result.get("tool_results", []):
            if isinstance(tr, dict):
                tool_details.append(
                    ToolDetail(
                        tool=tr.get("tool", "unknown"),
                        input=tr.get("input", ""),
                        success=tr.get("success", False),
                    )
                )
            else:
                tool_details.append(
                    ToolDetail(
                        tool=tr.tool,
                        input=tr.input,
                        success=tr.success,
                    )
                )

        return QueryResponse(
            query=request.query,
            response=result.get("response", ""),
            tools_used=result.get("tools_used", []),
            tool_details=tool_details,
        )

    except Exception as e:
        logger.exception("Agent execution failed for query: %s", request.query[:100])
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {e!s}") from e
