import sqlite3
from pathlib import Path

import pytest


def make_db(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE city (id INTEGER PRIMARY KEY, name TEXT, pop INTEGER)")
    conn.executemany(
        "INSERT INTO city VALUES (?, ?, ?)",
        [(1, "Oslo", 5), (2, "Rome", 9), (3, "x" * 500, 1)],
    )
    conn.execute("CREATE TABLE country (code TEXT PRIMARY KEY, name TEXT)")
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def databases_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "dbs"
    directory.mkdir()
    make_db(directory / "toy.sqlite")
    make_db(directory / "other.db")
    (directory / "notes.txt").write_text("not a database")
    return directory
