"""Tool for searching company documents using vector similarity (Qdrant)."""

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DocumentChunk(BaseModel):
    """A relevant chunk retrieved from the document store."""

    content: str = Field(description="Text content of the chunk")
    source: str = Field(description="Source document filename")
    score: float = Field(description="Relevance score (0-1)")


class DocSearchResult(BaseModel):
    """Result of a document search operation."""

    query: str = Field(description="The search query")
    chunks: list[DocumentChunk] = Field(description="Retrieved document chunks")
    chunk_count: int = Field(description="Number of chunks returned")
    success: bool = Field(description="Whether the search executed successfully")
    error: str | None = Field(default=None, description="Error message if search failed")


def doc_search(
    query: str,
    qdrant_url: str = "http://localhost:6333",
    collection_name: str = "company_docs",
    top_k: int = 5,
    score_threshold: float = 0.3,
    _client: Any = None,
    _embedder: Any = None,
) -> DocSearchResult:
    """Search company documents for relevant information.

    Uses vector similarity search against documents indexed in Qdrant.

    Args:
        query: Natural language search query.
        qdrant_url: Qdrant server URL.
        collection_name: Name of the Qdrant collection to search.
        top_k: Maximum number of chunks to return.
        score_threshold: Minimum similarity score to include a result.
        _client: Injected Qdrant client for testing.
        _embedder: Injected embedding model for testing.

    Returns:
        DocSearchResult with relevant document chunks.
    """
    if not query.strip():
        return DocSearchResult(
            query=query,
            chunks=[],
            chunk_count=0,
            success=False,
            error="Empty search query",
        )

    try:
        if _client is None:
            from qdrant_client import QdrantClient

            client = QdrantClient(url=qdrant_url)
        else:
            client = _client

        if _embedder is None:
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            embedder = _embedder

        query_vector = embedder.encode(query).tolist()

        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
        )

        chunks = [
            DocumentChunk(
                content=hit.payload.get("content", ""),
                source=hit.payload.get("source", "unknown"),
                score=round(hit.score, 4),
            )
            for hit in results
        ]

        logger.info("Doc search returned %d chunks for: %s", len(chunks), query[:100])

        return DocSearchResult(
            query=query,
            chunks=chunks,
            chunk_count=len(chunks),
            success=True,
        )

    except Exception as e:
        logger.warning("Doc search failed for '%s': %s", query[:100], str(e))
        return DocSearchResult(
            query=query,
            chunks=[],
            chunk_count=0,
            success=False,
            error=str(e),
        )
