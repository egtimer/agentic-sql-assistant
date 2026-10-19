"""Case files, disposable database copies and the synthetic fixture for the guardrail evals."""

import json
import os
import re
import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evals.text2sql.data import DATABASES_DIR, QUESTIONS_PATH, db_path, load_questions
from src.db.seed import seed_database
from src.sql_guard import GuardPolicy

HERE = Path(__file__).parent
ATTACKS_PATH = HERE / "attacks.jsonl"
BENIGN_PATH = HERE / "benign.jsonl"
INJECTION_PATH = HERE / "injection.jsonl"
REPORTS_DIR = HERE.parent / "reports"
DB_NAME = "db.sqlite"

FIXTURE_SCHEMA = """
CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT, password_hash TEXT);
CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id),
                     total REAL, note TEXT);
CREATE TABLE api_keys (id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id), token TEXT);
CREATE VIEW v_users_all AS SELECT * FROM users;
CREATE VIEW v_keys AS SELECT id, token FROM api_keys;
"""
FIXTURE_POLICY = GuardPolicy(
    denied_tables={"api_keys"},
    denied_columns={("users", "password_hash")},
)
_REPEAT = re.compile(r"<<REPEAT (\d+) (.*?)>>")


def expand(sql: str) -> str:
    """Expand ``<<REPEAT n unit>>`` macros (keeps long attack payloads out of the JSONL)."""
    return _REPEAT.sub(lambda m: m.group(2) * int(m.group(1)), sql)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_fixture(path: Path) -> None:
    """Synthetic database for the policy cases (fake data, built by code)."""
    conn = sqlite3.connect(path)
    conn.executescript(FIXTURE_SCHEMA)
    conn.executemany(
        "INSERT INTO users VALUES (?, ?, ?, ?)",
        [(i, f"user{i}", f"user{i}@example.com", f"pbkdf2$fake{i:04d}") for i in range(1, 6)],
    )
    conn.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?)",
        [(i, (i % 5) + 1, 10.0 * i, f"order {i}") for i in range(1, 11)],
    )
    conn.executemany(
        "INSERT INTO api_keys VALUES (?, ?, ?)", [(i, i, f"tok-fake-{i:04d}") for i in range(1, 6)]
    )
    conn.commit()
    conn.close()


def policy_for(db: str) -> GuardPolicy:
    return FIXTURE_POLICY if db == "fixture" else GuardPolicy()


def make_db(db: str, target: Path) -> Path:
    """Create ``target`` as a disposable database: the fixture, or a copy of a BIRD database."""
    if db == "fixture":
        build_fixture(target)
    elif db == "sales":
        seed_database(target)
    else:
        source = db_path(db)
        if not source.exists():
            raise FileNotFoundError(f"BIRD database missing: {source} (run `make bird-data`)")
        shutil.copyfile(source, target)
    return target


@contextmanager
def disposable(db: str, tmp_root: Path | None = None) -> Iterator[Path]:
    """Temporary directory holding a copy of ``db``; the cwd is that directory meanwhile."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="guardrails-", dir=tmp_root) as tmp:
        tmp_dir = Path(tmp).resolve()
        target = make_db(db, tmp_dir / DB_NAME)
        previous = Path.cwd()
        os.chdir(tmp_dir)
        try:
            assert target.resolve().is_relative_to(tmp_dir)
            assert not target.resolve().is_relative_to(DATABASES_DIR.resolve())
            yield target
        finally:
            os.chdir(previous)


def gold_cases() -> list[dict[str, Any]] | None:
    """The 500 BIRD Mini-Dev gold queries as benign cases, or None when the data is missing."""
    if not QUESTIONS_PATH.exists():
        return None
    return [
        {"id": f"gold-{q.question_id}", "sql": q.sql, "db": q.db_id, "gold": True}
        for q in load_questions()
    ]
