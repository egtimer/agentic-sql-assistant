"""Document ingestion pipeline: chunk, embed, and store in Qdrant."""

import logging
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from src.config import settings

logger = logging.getLogger(__name__)

DOCUMENTS_DIR = Path(__file__).parent.parent.parent / "data" / "documents"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


class DocumentChunk(BaseModel):
    """A text chunk with metadata ready for indexing."""

    content: str = Field(description="Chunk text content")
    source: str = Field(description="Source filename")
    chunk_index: int = Field(description="Position of the chunk in the source document")


def chunk_text(
    text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP
) -> list[str]:
    """Split text into overlapping chunks by character count.

    Splits on paragraph boundaries when possible, falling back to
    sentence boundaries, then hard character splits.

    Args:
        text: Full document text.
        chunk_size: Target characters per chunk.
        overlap: Number of overlapping characters between chunks.

    Returns:
        List of text chunks.
    """
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
            current_chunk = f"{current_chunk}\n\n{paragraph}" if current_chunk else paragraph
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                overlap_text = (
                    current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                )
                current_chunk = f"{overlap_text}\n\n{paragraph}" if overlap_text else paragraph
            else:
                if len(paragraph) > chunk_size:
                    words = paragraph.split()
                    temp = ""
                    for word in words:
                        if len(temp) + len(word) + 1 <= chunk_size:
                            temp = f"{temp} {word}" if temp else word
                        else:
                            if temp:
                                chunks.append(temp.strip())
                                temp_overlap = temp[-overlap:] if len(temp) > overlap else temp
                                temp = f"{temp_overlap} {word}"
                            else:
                                chunks.append(word[:chunk_size])
                                temp = word[chunk_size:]
                    if temp:
                        current_chunk = temp
                else:
                    current_chunk = paragraph

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def load_documents(docs_dir: Path | None = None) -> list[DocumentChunk]:
    """Load and chunk all markdown documents from the documents directory.

    Args:
        docs_dir: Directory containing .md files. Defaults to data/documents/.

    Returns:
        List of DocumentChunk objects ready for embedding.
    """
    docs_dir = docs_dir or DOCUMENTS_DIR

    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    all_chunks: list[DocumentChunk] = []

    for md_file in sorted(docs_dir.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        for i, chunk_content in enumerate(chunks):
            all_chunks.append(
                DocumentChunk(
                    content=chunk_content,
                    source=md_file.name,
                    chunk_index=i,
                )
            )

        logger.info("Loaded %s: %d chunks", md_file.name, len(chunks))

    logger.info(
        "Total documents loaded: %d files, %d chunks",
        len(list(docs_dir.glob("*.md"))),
        len(all_chunks),
    )
    return all_chunks


def ingest_to_qdrant(
    chunks: list[DocumentChunk],
    qdrant_url: str | None = None,
    collection_name: str | None = None,
    embedding_model: str | None = None,
) -> int:
    """Embed document chunks and upload to Qdrant.

    Creates the collection if it doesn't exist, replacing any existing one
    with the same name to ensure fresh data.

    Args:
        chunks: Document chunks to ingest.
        qdrant_url: Qdrant server URL.
        collection_name: Target collection name.
        embedding_model: Sentence-transformers model name.

    Returns:
        Number of points uploaded.
    """
    qdrant_url = qdrant_url or settings.qdrant_url
    collection_name = collection_name or settings.qdrant_collection
    embedding_model = embedding_model or settings.embedding_model

    logger.info("Loading embedding model: %s", embedding_model)
    embedder = SentenceTransformer(embedding_model)

    texts = [chunk.content for chunk in chunks]
    logger.info("Generating embeddings for %d chunks...", len(texts))
    embeddings = embedder.encode(texts, show_progress_bar=True)
    vector_size = embeddings.shape[1]

    logger.info("Connecting to Qdrant at %s", qdrant_url)
    client = QdrantClient(url=qdrant_url)

    collections = [c.name for c in client.get_collections().collections]
    if collection_name in collections:
        client.delete_collection(collection_name)
        logger.info("Deleted existing collection: %s", collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    logger.info("Created collection: %s (vector_size=%d)", collection_name, vector_size)

    points = [
        PointStruct(
            id=str(uuid4()),
            vector=embedding.tolist(),
            payload={
                "content": chunk.content,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
            },
        )
        for chunk, embedding in zip(chunks, embeddings, strict=False)
    ]

    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)

    logger.info("Uploaded %d points to collection '%s'", len(points), collection_name)
    return len(points)


def run_ingestion() -> None:
    """Execute the full ingestion pipeline."""
    chunks = load_documents()
    if not chunks:
        logger.warning("No documents found to ingest")
        return

    count = ingest_to_qdrant(chunks)
    print(f"Ingestion complete: {count} chunks indexed in Qdrant")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_ingestion()
