"""Download and verify the Chinook demo database used by the MCP server.

Chinook is a sample database of a digital media store (artists, albums, tracks, invoices,
customers) by Luis Rocha: https://github.com/lerocha/chinook-database, MIT License,
Copyright (c) 2008-2024 Luis Rocha. The SQLite build of release v1.4.5 is fetched on demand
into ``data/external/chinook/`` (ignored by git) and checked against a pinned sha256; a file
that does not match is refused and removed.

Usage: ``python -m src.mcp_server.demo_data`` (or ``make chinook-data``).
"""

import hashlib
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
CHINOOK_DIR = REPO_ROOT / "data" / "external" / "chinook"
CHINOOK_URL = (
    "https://github.com/lerocha/chinook-database/releases/download/v1.4.5/Chinook_Sqlite.sqlite"
)
CHINOOK_SHA256 = "bdf635be69850bd3be09c9a2dbeef7ddfb80036bd3ef3381383cd03b61e4a61a"
CHINOOK_FILENAME = "Chinook.sqlite"
LICENSE_LINE = "Chinook database: MIT License, Copyright (c) 2008-2024 Luis Rocha"


class ChecksumError(RuntimeError):
    """The downloaded or existing file is not the pinned release."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_bytes(1 << 20):
                f.write(chunk)


def ensure_chinook(
    dest_dir: Path = CHINOOK_DIR,
    *,
    url: str = CHINOOK_URL,
    expected_sha256: str = CHINOOK_SHA256,
) -> Path:
    """Path of the verified Chinook file in ``dest_dir``, downloading it when missing.

    Raises :class:`ChecksumError` on a mismatch. A fresh download that does not match is
    deleted; an existing file that does not match is left for the user to inspect.
    """
    target = dest_dir / CHINOOK_FILENAME
    if target.exists():
        actual = sha256_of(target)
        if actual != expected_sha256:
            raise ChecksumError(
                f"sha256 mismatch for {target.name}: expected {expected_sha256}, got {actual}. "
                "Delete the file and run again."
            )
        return target
    part = target.with_suffix(".sqlite.part")
    print(f"Downloading {url}")
    try:
        download(url, part)
        actual = sha256_of(part)
        if actual != expected_sha256:
            raise ChecksumError(
                f"sha256 mismatch for the download: expected {expected_sha256}, got {actual}. "
                "The file was discarded."
            )
        part.replace(target)
    finally:
        part.unlink(missing_ok=True)
    return target


def main() -> int:
    try:
        path = ensure_chinook()
    except (ChecksumError, httpx.HTTPError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"OK {path.name} sha256 {CHINOOK_SHA256[:12]}... in {path.parent}")
    print(LICENSE_LINE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
