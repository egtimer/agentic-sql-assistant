"""The embedding runtime of the document index: one pinned model, recorded and checked.

The same model name can give different vectors when the weights behind a moving
Hugging Face branch change, so the model is loaded at a fixed revision. Ingestion stores
model and revision in every point; queries compare them with their own and refuse an
index built with another runtime, instead of silently searching vectors that do not
match (the fastapi-rag-lab ADR-007 failure).
"""

import logging
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)

RUNTIME_KEYS = ("embedding_model", "embedding_revision")


class EmbeddingRuntimeMismatch(RuntimeError):  # noqa: N818  (reads as a condition)
    """The index was built with another embedding model or revision."""


def embedding_runtime() -> dict[str, str]:
    """Model and revision the current settings embed with."""
    return {
        "embedding_model": settings.embedding_model,
        "embedding_revision": settings.embedding_revision,
    }


def load_embedder() -> Any:
    """The sentence-transformers model at the pinned revision."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model, revision=settings.embedding_revision)


def index_runtime(client: Any, collection_name: str) -> dict[str, str] | None:
    """Runtime recorded in the collection's first point; None if it records none."""
    points, _ = client.scroll(
        collection_name=collection_name, limit=1, with_payload=True, with_vectors=False
    )
    if not points:
        return None
    payload = points[0].payload or {}
    if not all(key in payload for key in RUNTIME_KEYS):
        return None
    return {key: str(payload[key]) for key in RUNTIME_KEYS}


_checked: set[tuple[str, str]] = set()


def check_index_runtime(client: Any, collection_name: str, cache_key: str) -> None:
    """Raise EmbeddingRuntimeMismatch if the index was built with another runtime.

    An index without a recorded runtime (built before it was recorded) only logs a
    warning. A passed check is remembered per ``(cache_key, collection)``.
    """
    key = (cache_key, collection_name)
    if key in _checked:
        return
    recorded = index_runtime(client, collection_name)
    current = embedding_runtime()
    if recorded is None:
        logger.warning(
            "Collection %s does not record its embedding runtime; re-ingest to record it",
            collection_name,
        )
    elif recorded != current:
        raise EmbeddingRuntimeMismatch(
            f"collection {collection_name} was built with {recorded}, "
            f"queries embed with {current}; re-ingest the documents"
        )
    _checked.add(key)
