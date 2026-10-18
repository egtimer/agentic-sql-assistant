"""Hybrid retrieval: dense vector search + BM25 keyword matching with RRF fusion."""

import logging
import math
import re
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.config import settings

logger = logging.getLogger(__name__)


class RetrievedChunk(BaseModel):
    """A document chunk with retrieval metadata."""

    content: str = Field(description="Text content of the chunk")
    source: str = Field(description="Source document filename")
    score: float = Field(description="Combined relevance score")
    dense_score: float = Field(default=0.0, description="Dense vector similarity score")
    bm25_score: float = Field(default=0.0, description="BM25 keyword score")


def _tokenize(text: str) -> list[str]:
    """Simple tokenization: lowercase, split on non-alphanumeric, filter short tokens."""
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if len(t) > 1]


def _compute_bm25_scores(
    query: str,
    documents: list[dict[str, Any]],
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[int, float]:
    """Compute BM25 scores for documents against a query.

    Args:
        query: Search query string.
        documents: List of dicts with 'content' key.
        k1: Term frequency saturation parameter.
        b: Length normalization parameter.

    Returns:
        Mapping of document index to BM25 score.
    """
    query_tokens = _tokenize(query)
    if not query_tokens or not documents:
        return {}

    doc_tokens = [_tokenize(doc["content"]) for doc in documents]
    avg_dl = sum(len(dt) for dt in doc_tokens) / len(doc_tokens) if doc_tokens else 1

    doc_freqs: dict[str, int] = defaultdict(int)
    for tokens in doc_tokens:
        seen = set(tokens)
        for token in seen:
            doc_freqs[token] += 1

    n = len(documents)
    scores: dict[int, float] = {}

    for idx, tokens in enumerate(doc_tokens):
        score = 0.0
        dl = len(tokens)
        tf_map: dict[str, int] = defaultdict(int)
        for t in tokens:
            tf_map[t] += 1

        for qt in query_tokens:
            if qt not in tf_map:
                continue

            tf = tf_map[qt]
            df = doc_freqs.get(qt, 0)
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
            score += idf * tf_norm

        if score > 0:
            scores[idx] = score

    return scores


def _reciprocal_rank_fusion(
    rankings: list[dict[int, float]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Combine multiple rankings using Reciprocal Rank Fusion.

    Args:
        rankings: List of {doc_index: score} dicts, one per retrieval method.
        k: RRF parameter (default 60 as per original paper).

    Returns:
        Sorted list of (doc_index, fused_score) tuples.
    """
    fused: dict[int, float] = defaultdict(float)

    for ranking in rankings:
        sorted_items = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
        for rank, (doc_idx, _) in enumerate(sorted_items, start=1):
            fused[doc_idx] += 1.0 / (k + rank)

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def retrieve(
    query: str,
    top_k: int = 5,
    qdrant_url: str | None = None,
    collection_name: str | None = None,
    _client: Any = None,
    _embedder: Any = None,
) -> list[RetrievedChunk]:
    """Perform hybrid retrieval combining dense vectors and BM25.

    1. Dense search: embed query and find nearest vectors in Qdrant.
    2. BM25 search: score retrieved candidates by keyword overlap.
    3. RRF fusion: combine both rankings into a single result set.

    Args:
        query: Natural language search query.
        top_k: Number of results to return.
        qdrant_url: Qdrant server URL.
        collection_name: Qdrant collection to search.
        _client: Injected Qdrant client (for testing).
        _embedder: Injected embedding model (for testing).

    Returns:
        List of RetrievedChunk ordered by relevance.
    """
    qdrant_url = qdrant_url or settings.qdrant_url
    collection_name = collection_name or settings.qdrant_collection

    client = QdrantClient(url=qdrant_url) if _client is None else _client
    embedder = SentenceTransformer(settings.embedding_model) if _embedder is None else _embedder

    query_vector = embedder.encode(query).tolist()

    candidate_limit = max(top_k * 3, 20)
    hits = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=candidate_limit,
    )

    if not hits:
        return []

    candidates = []
    dense_ranking: dict[int, float] = {}

    for idx, hit in enumerate(hits):
        payload = hit.payload or {}
        candidates.append(
            {
                "content": payload.get("content", ""),
                "source": payload.get("source", "unknown"),
            }
        )
        dense_ranking[idx] = hit.score

    bm25_ranking = _compute_bm25_scores(query, candidates)

    fused_results = _reciprocal_rank_fusion([dense_ranking, bm25_ranking])

    chunks = []
    for doc_idx, fused_score in fused_results[:top_k]:
        candidate = candidates[doc_idx]
        chunks.append(
            RetrievedChunk(
                content=candidate["content"],
                source=candidate["source"],
                score=round(fused_score, 4),
                dense_score=round(dense_ranking.get(doc_idx, 0.0), 4),
                bm25_score=round(bm25_ranking.get(doc_idx, 0.0), 4),
            )
        )

    logger.info("Retrieved %d chunks for query: %s", len(chunks), query[:80])
    return chunks
