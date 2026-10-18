"""BIRD Mini-Dev data access: questions, splits, database paths and schema DDL."""

import json
import sqlite3
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DIR = REPO_ROOT / "data" / "external"
RAW_DIR = EXTERNAL_DIR / "bird_raw"
BIRD_DIR = EXTERNAL_DIR / "bird"
DATABASES_DIR = BIRD_DIR / "dev_databases"
QUESTIONS_PATH = RAW_DIR / "mini_dev_sqlite.json"
SPLITS_DIR = Path(__file__).parent / "splits"

Difficulty = Literal["simple", "moderate", "challenging"]
DIFFICULTIES: tuple[Difficulty, ...] = ("simple", "moderate", "challenging")


class Question(BaseModel):
    """One BIRD Mini-Dev item."""

    model_config = ConfigDict(populate_by_name=True)

    question_id: int
    db_id: str
    question: str
    evidence: str = ""
    sql: str = Field(alias="SQL")
    difficulty: Difficulty


def load_questions(path: Path | None = None) -> list[Question]:
    """Load all questions from the Hugging Face JSON file."""
    with open(path or QUESTIONS_PATH, encoding="utf-8") as f:
        return [Question.model_validate(item) for item in json.load(f)]


def load_split(name: str, splits_dir: Path | None = None) -> list[int]:
    """Return the sorted question ids of the ``dev`` or ``test`` split."""
    path = (splits_dir or SPLITS_DIR) / f"{name}.json"
    with open(path, encoding="utf-8") as f:
        ids: list[int] = json.load(f)
    return ids


def db_path(db_id: str, databases_dir: Path | None = None) -> Path:
    """Path of the SQLite file for ``db_id``."""
    return (databases_dir or DATABASES_DIR) / db_id / f"{db_id}.sqlite"


def connect_readonly(path: Path | str) -> sqlite3.Connection:
    """Open a SQLite file in read-only mode (``mode=ro`` URI)."""
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def schema_ddl(path: Path | str) -> str:
    """CREATE TABLE statements from ``sqlite_master`` joined by blank lines."""
    conn = connect_readonly(path)
    try:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND sql IS NOT NULL AND name NOT LIKE 'sqlite_%' "
            "ORDER BY rowid"
        ).fetchall()
    finally:
        conn.close()
    return "\n\n".join(row[0].strip() for row in rows)
