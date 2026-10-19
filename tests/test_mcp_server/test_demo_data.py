import hashlib
from pathlib import Path

import pytest

from src.mcp_server import demo_data
from src.mcp_server.demo_data import CHINOOK_FILENAME, ChecksumError, ensure_chinook

PAYLOAD = b"not really sqlite"
SHA = hashlib.sha256(PAYLOAD).hexdigest()


def fake_download(payload: bytes):  # type: ignore[no-untyped-def]
    def download(url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)

    return download


def test_download_verifies_and_keeps_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(demo_data, "download", fake_download(PAYLOAD))
    path = ensure_chinook(tmp_path / "c", url="http://x/y", expected_sha256=SHA)
    assert path == tmp_path / "c" / CHINOOK_FILENAME
    assert path.read_bytes() == PAYLOAD
    assert not list((tmp_path / "c").glob("*.part"))


def test_mismatching_download_is_refused_and_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(demo_data, "download", fake_download(b"tampered"))
    with pytest.raises(ChecksumError, match="sha256 mismatch"):
        ensure_chinook(tmp_path / "c", url="http://x/y", expected_sha256=SHA)
    assert list((tmp_path / "c").iterdir()) == []


def test_existing_file_is_verified_without_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def boom(url: str, dest: Path) -> None:
        raise AssertionError("must not download")

    monkeypatch.setattr(demo_data, "download", boom)
    (tmp_path / CHINOOK_FILENAME).write_bytes(PAYLOAD)
    assert ensure_chinook(tmp_path, expected_sha256=SHA) == tmp_path / CHINOOK_FILENAME
    (tmp_path / CHINOOK_FILENAME).write_bytes(b"other")
    with pytest.raises(ChecksumError):
        ensure_chinook(tmp_path, expected_sha256=SHA)


def test_pinned_values():
    assert demo_data.CHINOOK_SHA256 == (
        "bdf635be69850bd3be09c9a2dbeef7ddfb80036bd3ef3381383cd03b61e4a61a"
    )
    assert "v1.4.5" in demo_data.CHINOOK_URL
