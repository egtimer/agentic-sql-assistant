"""In-process MCP client: agent v1's doc_search and calculator tools (no Qdrant, no LLM)."""

from pathlib import Path
from typing import Any

import pytest
from mcp import Client

from src.mcp_server.catalog import Catalog
from src.mcp_server.server import create_server
from src.mcp_server.service import Service
from src.tools.doc_search import DocSearchResult, DocumentChunk

from .test_tools_basic import text_of


def fake_search(query: str, top_k: int = 5) -> DocSearchResult:
    chunks = [
        DocumentChunk(content="30 días naturales", source="politica_devoluciones.md", score=0.9),
        DocumentChunk(
            content="15 días para electrónica", source="politica_devoluciones.md", score=0.8
        ),
    ][:top_k]
    return DocSearchResult(query=query, chunks=chunks, chunk_count=len(chunks), success=True)


def failing_search(query: str, top_k: int = 5) -> DocSearchResult:
    return DocSearchResult(
        query=query, chunks=[], chunk_count=0, success=False,
        error="collection company_docs was built with another runtime; re-ingest",
    )  # fmt: skip


def server_with(directory: Path, search: Any) -> Any:
    return create_server(Service(Catalog.scan(directory), doc_search_fn=search))


async def test_tools_are_listed_read_only(databases_dir: Path):
    async with Client(server_with(databases_dir, fake_search)) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    for name in ("doc_search", "calculator"):
        assert tools[name].description and tools[name].output_schema is not None
        assert tools[name].annotations is not None and tools[name].annotations.read_only_hint
    assert set(tools["doc_search"].input_schema["required"]) == {"query"}


async def test_calculator(databases_dir: Path):
    async with Client(server_with(databases_dir, fake_search)) as client:
        ok = await client.call_tool("calculator", {"expression": "1299 * (1 - 0.12)"})
        bad = await client.call_tool("calculator", {"expression": "__import__('os')"})
    assert not ok.is_error and ok.structured_content["result"] == pytest.approx(1143.12)
    assert "= 1,143.12" in text_of(ok)
    assert bad.structured_content["success"] is False
    assert text_of(bad).startswith("Calculation failed")


async def test_doc_search_returns_passages_with_sources(databases_dir: Path):
    async with Client(server_with(databases_dir, fake_search)) as client:
        result = await client.call_tool("doc_search", {"query": "plazo de devolución", "top_k": 1})
    assert not result.is_error
    assert result.structured_content["chunk_count"] == 1
    assert "[politica_devoluciones.md, score 0.900]" in text_of(result)


async def test_doc_search_reports_a_refused_index(databases_dir: Path):
    async with Client(server_with(databases_dir, failing_search)) as client:
        result = await client.call_tool("doc_search", {"query": "garantía"})
    assert result.structured_content["success"] is False
    assert "another runtime" in text_of(result)
