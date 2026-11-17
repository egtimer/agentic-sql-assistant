"""One place to open the Qdrant client.

``QDRANT_URL`` is normally a server URL (``http://localhost:6333``, the Docker service). A
value ``file:<dir>`` opens Qdrant's embedded local mode on that directory instead, for
machines that cannot run the server (no Docker); local mode allows one client per directory
per process, so that client is shared.
"""

from functools import lru_cache
from typing import Any

FILE_PREFIX = "file:"


@lru_cache(maxsize=4)
def _local(path: str) -> Any:
    from qdrant_client import QdrantClient

    return QdrantClient(path=path)


def qdrant_client(url: str) -> Any:
    """A client for ``url``: a server URL, or ``file:<dir>`` for embedded local mode."""
    if url.startswith(FILE_PREFIX):
        return _local(url[len(FILE_PREFIX) :])
    from qdrant_client import QdrantClient

    return QdrantClient(url=url)


def server_version(client: Any, url: str) -> str:
    """The server's version, or ``embedded`` in local mode."""
    if url.startswith(FILE_PREFIX):
        return "embedded"
    return str(client.info().version)
