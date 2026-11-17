"""``qdrant_client``: a server URL opens a remote client; ``file:<dir>`` a shared local one."""

from pathlib import Path

from src.rag.qdrant import qdrant_client, server_version


def test_file_url_opens_one_shared_local_client(tmp_path: Path) -> None:
    url = f"file:{tmp_path / 'q'}"
    first, second = qdrant_client(url), qdrant_client(url)
    assert first is second  # local mode allows one client per directory per process
    assert first.get_collections().collections == []
    assert server_version(first, url) == "embedded"


def test_server_url_is_not_opened_locally() -> None:
    client = qdrant_client("http://localhost:6333")
    assert type(client).__name__ == "QdrantClient"
    assert client._client.__class__.__name__ != "QdrantLocal"
