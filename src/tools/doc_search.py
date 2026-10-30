"""Tool for searching company documents using hybrid retrieval."""

import logging
from typing import Any

from pydantic import BaseModel, Field

from src.config import settings
from src.rag.embedding import EmbeddingRuntimeMismatch, check_index_runtime, load_embedder

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
    qdrant_url: str | None = None,
    collection_name: str | None = None,
    top_k: int = 5,
    score_threshold: float = 0.0,
    _client: Any = None,
    _embedder: Any = None,
) -> DocSearchResult:
    """Search company documents for relevant information.

    Uses hybrid retrieval (dense + BM25 with RRF fusion) against
    documents indexed in Qdrant. Falls back to direct vector search
    if the hybrid retriever is unavailable.

    Args:
        query: Natural language search query.
        qdrant_url: Qdrant server URL.
        collection_name: Name of the Qdrant collection to search.
        top_k: Maximum number of chunks to return.
        score_threshold: Minimum score to include a result.
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

    qdrant_url = qdrant_url or settings.qdrant_url
    collection_name = collection_name or settings.qdrant_collection

    try:
        from src.rag.retriever import retrieve

        retrieved = retrieve(
            query=query,
            top_k=top_k,
            qdrant_url=qdrant_url,
            collection_name=collection_name,
            _client=_client,
            _embedder=_embedder,
        )

        chunks = [
            DocumentChunk(
                content=chunk.content,
                source=chunk.source,
                score=chunk.score,
            )
            for chunk in retrieved
            if chunk.score >= score_threshold
        ]

        logger.info("Doc search returned %d chunks for: %s", len(chunks), query[:100])

        return DocSearchResult(
            query=query,
            chunks=chunks,
            chunk_count=len(chunks),
            success=True,
        )

    except EmbeddingRuntimeMismatch as e:
        # the fallback would search the same mismatched vectors
        logger.error("Doc search refused: %s", e)
        return DocSearchResult(query=query, chunks=[], chunk_count=0, success=False, error=str(e))
    except Exception as e:
        logger.warning("Hybrid retrieval failed, attempting direct search: %s", str(e))
        return _fallback_search(query, qdrant_url, collection_name, top_k, _client, _embedder)


def _fallback_search(
    query: str,
    qdrant_url: str,
    collection_name: str,
    top_k: int,
    _client: Any = None,
    _embedder: Any = None,
) -> DocSearchResult:
    """Direct vector search fallback when hybrid retrieval fails."""
    try:
        if _client is None:
            from qdrant_client import QdrantClient

            client = QdrantClient(url=qdrant_url)
            check_index_runtime(client, collection_name, qdrant_url)
        else:
            client = _client

        embedder = load_embedder() if _embedder is None else _embedder

        query_vector = embedder.encode(query).tolist()

        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
        )

        chunks = [
            DocumentChunk(
                content=(hit.payload or {}).get("content", ""),
                source=(hit.payload or {}).get("source", "unknown"),
                score=round(hit.score, 4),
            )
            for hit in results
        ]

        return DocSearchResult(
            query=query,
            chunks=chunks,
            chunk_count=len(chunks),
            success=True,
        )

    except Exception as e:
        logger.error("Fallback search also failed: %s", str(e))
        return DocSearchResult(
            query=query,
            chunks=[],
            chunk_count=0,
            success=False,
            error=str(e),
        )
