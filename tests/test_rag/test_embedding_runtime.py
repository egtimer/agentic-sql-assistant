"""Tests for the pinned, recorded and checked embedding runtime of the document index."""

from dataclasses import dataclass, field
from typing import Any

import pytest

from src.rag import embedding
from src.rag.embedding import (
    EmbeddingRuntimeMismatch,
    check_index_runtime,
    embedding_runtime,
    index_runtime,
)
from src.tools.doc_search import doc_search


@dataclass
class FakePoint:
    payload: dict[str, Any]


@dataclass
class FakeClient:
    payloads: list[dict[str, Any]] = field(default_factory=list)
    scrolls: int = 0

    def scroll(self, **kwargs: Any) -> tuple[list[FakePoint], None]:
        self.scrolls += 1
        return [FakePoint(p) for p in self.payloads[:1]], None


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    embedding._checked.clear()


def test_runtime_names_a_pinned_revision() -> None:
    runtime = embedding_runtime()
    assert runtime["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert len(runtime["embedding_revision"]) == 40


def test_index_runtime_reads_the_first_point() -> None:
    client = FakeClient([{"content": "x", **embedding_runtime()}])
    assert index_runtime(client, "docs") == embedding_runtime()
    assert index_runtime(FakeClient([{"content": "x"}]), "docs") is None
    assert index_runtime(FakeClient([]), "docs") is None


def test_matching_index_passes_once_and_is_cached() -> None:
    client = FakeClient([{"content": "x", **embedding_runtime()}])
    check_index_runtime(client, "docs", "url")
    check_index_runtime(client, "docs", "url")
    assert client.scrolls == 1


def test_other_revision_is_refused() -> None:
    other = {**embedding_runtime(), "embedding_revision": "0" * 40}
    with pytest.raises(EmbeddingRuntimeMismatch, match="re-ingest"):
        check_index_runtime(FakeClient([{"content": "x", **other}]), "docs", "url")


def test_unrecorded_index_only_warns(caplog: pytest.LogCaptureFixture) -> None:
    check_index_runtime(FakeClient([{"content": "x"}]), "docs", "url")
    assert "does not record its embedding runtime" in caplog.text


def test_doc_search_refuses_instead_of_falling_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def mismatched(*args: Any, **kwargs: Any) -> None:
        raise EmbeddingRuntimeMismatch("collection docs was built with another runtime")

    monkeypatch.setattr("src.rag.retriever.retrieve", mismatched)
    result = doc_search("return policy")
    assert result.success is False
    assert result.error is not None and "another runtime" in result.error
