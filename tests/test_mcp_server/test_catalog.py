from pathlib import Path

import pytest

from src.mcp_server.catalog import Catalog, UnknownDatabaseError
from tests.test_mcp_server.conftest import make_db


def test_scan_uses_file_stems_and_ignores_other_files(databases_dir: Path):
    catalog = Catalog.scan(databases_dir)
    assert catalog.names == ["other", "toy"]
    assert catalog.path("toy") == (databases_dir / "toy.sqlite").resolve()
    assert catalog.table_count("toy") == 2


def test_unknown_name_lists_valid_names(databases_dir: Path):
    catalog = Catalog.scan(databases_dir)
    with pytest.raises(UnknownDatabaseError, match="'other', 'toy'"):
        catalog.path("../secret")


def test_paths_are_never_names(databases_dir: Path):
    catalog = Catalog.scan(databases_dir)
    for attempt in (str(databases_dir / "toy.sqlite"), "toy.sqlite", "dbs/toy"):
        with pytest.raises(UnknownDatabaseError):
            catalog.path(attempt)


def test_duplicate_stems_are_refused(tmp_path: Path):
    make_db(tmp_path / "a.sqlite")
    make_db(tmp_path / "a.db")
    with pytest.raises(ValueError, match="share the name 'a'"):
        Catalog.scan(tmp_path)


def test_missing_directory_gives_an_empty_catalog(tmp_path: Path):
    assert Catalog.scan(tmp_path / "nope").names == []
