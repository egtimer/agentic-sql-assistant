"""Tests for the document search tool."""

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from src.tools.doc_search import doc_search


@dataclass
class FakeSearchHit:
    """Simulates a Qdrant search result."""

    payload: dict[str, str]
    score: float


class FakeEmbedder:
    """Simulates a sentence-transformers embedding model."""

    def encode(self, text: str) -> np.ndarray:
        return np.random.default_rng(42).random(384).astype(np.float32)


class FakeQdrantClient:
    """Simulates a Qdrant client with pre-configured responses."""

    def __init__(self, results: list[FakeSearchHit] | None = None) -> None:
        self._results = results or []

    def search(self, **kwargs: Any) -> list[FakeSearchHit]:
        return self._results


class TestDocSearchBasic:
    def test_returns_matching_chunks(self) -> None:
        hits = [
            FakeSearchHit(
                payload={
                    "content": "Return policy is 30 days",
                    "source": "politica_devoluciones.md",
                },
                score=0.92,
            ),
            FakeSearchHit(
                payload={
                    "content": "Defective products get full refund",
                    "source": "politica_devoluciones.md",
                },
                score=0.78,
            ),
        ]
        client = FakeQdrantClient(results=hits)
        embedder = FakeEmbedder()

        result = doc_search("return policy", _client=client, _embedder=embedder)

        assert result.success is True
        assert result.chunk_count == 2
        assert result.chunks[0].source == "politica_devoluciones.md"
        assert result.chunks[0].score > 0

    def test_returns_empty_when_no_matches(self) -> None:
        client = FakeQdrantClient(results=[])
        embedder = FakeEmbedder()

        result = doc_search("something obscure", _client=client, _embedder=embedder)

        assert result.success is True
        assert result.chunk_count == 0
        assert result.chunks == []

    def test_preserves_query_in_result(self) -> None:
        client = FakeQdrantClient(results=[])
        embedder = FakeEmbedder()

        result = doc_search("discount policy for Gold tier", _client=client, _embedder=embedder)

        assert result.query == "discount policy for Gold tier"

    def test_multiple_sources(self) -> None:
        hits = [
            FakeSearchHit(
                payload={"content": "Volume discounts apply", "source": "politica_descuentos.md"},
                score=0.85,
            ),
            FakeSearchHit(
                payload={"content": "ProBook X1 is popular", "source": "faq_productos.md"},
                score=0.60,
            ),
        ]
        client = FakeQdrantClient(results=hits)
        embedder = FakeEmbedder()

        result = doc_search("product discounts", _client=client, _embedder=embedder)

        assert result.success is True
        sources = {chunk.source for chunk in result.chunks}
        assert "politica_descuentos.md" in sources
        assert "faq_productos.md" in sources


class TestDocSearchValidation:
    def test_empty_query(self) -> None:
        result = doc_search("")
        assert result.success is False
        assert "Empty" in result.error

    def test_whitespace_query(self) -> None:
        result = doc_search("   ")
        assert result.success is False
        assert "Empty" in result.error


class TestDocSearchErrors:
    def test_connection_error(self) -> None:
        client = MagicMock()
        client.search.side_effect = ConnectionError("Qdrant unavailable")
        embedder = FakeEmbedder()

        result = doc_search("test query", _client=client, _embedder=embedder)

        assert result.success is False
        assert "Qdrant unavailable" in result.error

    def test_missing_payload_fields(self) -> None:
        hits = [
            FakeSearchHit(payload={}, score=0.5),
        ]
        client = FakeQdrantClient(results=hits)
        embedder = FakeEmbedder()

        result = doc_search("test", _client=client, _embedder=embedder)

        assert result.success is True
        assert result.chunks[0].content == ""
        assert result.chunks[0].source == "unknown"
