"""Tests for the BIRD data helpers: split, extraction, schema DDL."""

import json
import sqlite3
import zipfile
from collections import Counter
from pathlib import Path

import pytest

from evals.text2sql.data import (
    Question,
    connect_readonly,
    db_path,
    load_questions,
    load_split,
    schema_ddl,
)
from evals.text2sql.prepare import (
    extract_databases,
    make_split,
    sha256_of,
    write_split,
)

SPLITS_DIR = Path(__file__).parents[2] / "evals" / "text2sql" / "splits"


def synthetic_questions() -> list[Question]:
    counts = {"simple": 30, "moderate": 50, "challenging": 20}
    questions = []
    qid = 0
    for difficulty, n in counts.items():
        for _ in range(n):
            qid += 1
            questions.append(
                Question(
                    question_id=qid * 3,
                    db_id="db",
                    question=f"q{qid}",
                    evidence="",
                    SQL="SELECT 1",
                    difficulty=difficulty,  # type: ignore[arg-type]
                )
            )
    return questions


class TestMakeSplit:
    def test_sizes_and_disjoint(self) -> None:
        dev, test = make_split(synthetic_questions(), n_dev=30)
        assert len(dev) == 30
        assert len(test) == 70
        assert not set(dev) & set(test)
        assert sorted(dev + test) == sorted(q.question_id for q in synthetic_questions())

    def test_lists_are_sorted(self) -> None:
        dev, test = make_split(synthetic_questions(), n_dev=30)
        assert dev == sorted(dev)
        assert test == sorted(test)

    def test_stratified(self) -> None:
        questions = synthetic_questions()
        difficulty = {q.question_id: q.difficulty for q in questions}
        dev, _ = make_split(questions, n_dev=30)
        counts = Counter(difficulty[i] for i in dev)
        assert counts == {"simple": 9, "moderate": 15, "challenging": 6}

    def test_largest_remainder_rounding(self) -> None:
        questions = synthetic_questions()
        difficulty = {q.question_id: q.difficulty for q in questions}
        # quotas 7 * (0.3, 0.5, 0.2) = 2.1, 3.5, 1.4 -> floors 2, 3, 1 -> one left over,
        # the largest remainder (.5) is moderate.
        dev, _ = make_split(questions, n_dev=7)
        counts = Counter(difficulty[i] for i in dev)
        assert counts == {"simple": 2, "moderate": 4, "challenging": 1}

    def test_deterministic(self) -> None:
        assert make_split(synthetic_questions(), 30) == make_split(synthetic_questions(), 30)

    def test_input_order_does_not_matter(self) -> None:
        questions = synthetic_questions()
        assert make_split(questions, 30) == make_split(list(reversed(questions)), 30)

    def test_seed_changes_split(self) -> None:
        assert make_split(synthetic_questions(), 30, seed=0) != make_split(
            synthetic_questions(), 30, seed=1
        )

    def test_invalid_n_dev(self) -> None:
        with pytest.raises(ValueError):
            make_split(synthetic_questions(), n_dev=1000)

    def test_write_split_is_idempotent(self, tmp_path: Path) -> None:
        questions = synthetic_questions()
        assert write_split(tmp_path, questions, n_dev=30) is True
        first = (tmp_path / "dev.json").read_text()
        assert write_split(tmp_path, questions, n_dev=30) is False
        assert (tmp_path / "dev.json").read_text() == first
        assert load_split("dev", tmp_path) == json.loads(first)


class TestCommittedSplit:
    def test_files_are_disjoint_and_sorted(self) -> None:
        dev = load_split("dev")
        test = load_split("test")
        assert len(dev) == 150
        assert len(test) == 350
        assert dev == sorted(dev)
        assert test == sorted(test)
        assert not set(dev) & set(test)


class TestExtractDatabases:
    def _make_zip(self, path: Path, names: dict[str, bytes]) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            for name, content in names.items():
                archive.writestr(name, content)

    def test_extracts_only_sqlite_files(self, tmp_path: Path) -> None:
        archive = tmp_path / "a.zip"
        self._make_zip(
            archive,
            {
                "minidev/MINIDEV/dev_databases/db1/db1.sqlite": b"data1",
                "minidev/MINIDEV/dev_databases/db1/database_description/t.csv": b"csv",
                "minidev/MINIDEV/mini_dev_sqlite.json": b"{}",
                "minidev/MINIDEV/dev_databases/db2/db2.sqlite": b"data2",
                "other/dev_databases/db3/db3.sqlite": b"no",
            },
        )
        dest = tmp_path / "out"
        extracted = extract_databases(archive, dest)
        assert sorted(p.name for p in extracted) == ["db1.sqlite", "db2.sqlite"]
        assert (dest / "dev_databases" / "db1" / "db1.sqlite").read_bytes() == b"data1"
        assert sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()) == [
            "dev_databases/db1/db1.sqlite",
            "dev_databases/db2/db2.sqlite",
        ]

    @pytest.mark.parametrize(
        "bad", ["/etc/passwd", "../evil.sqlite", "minidev/MINIDEV/dev_databases/../../x.sqlite"]
    )
    def test_rejects_unsafe_paths(self, tmp_path: Path, bad: str) -> None:
        archive = tmp_path / "a.zip"
        self._make_zip(archive, {"minidev/MINIDEV/dev_databases/db1/db1.sqlite": b"x", bad: b"y"})
        dest = tmp_path / "out"
        with pytest.raises(ValueError, match="Unsafe"):
            extract_databases(archive, dest)
        assert not dest.exists()
        assert not (tmp_path / "evil.sqlite").exists()

    def test_existing_files_are_kept(self, tmp_path: Path) -> None:
        archive = tmp_path / "a.zip"
        self._make_zip(archive, {"minidev/MINIDEV/dev_databases/db1/db1.sqlite": b"data1"})
        dest = tmp_path / "out"
        extract_databases(archive, dest)
        target = dest / "dev_databases" / "db1" / "db1.sqlite"
        mtime = target.stat().st_mtime_ns
        extract_databases(archive, dest)
        assert target.stat().st_mtime_ns == mtime

    def test_sha256_of(self, tmp_path: Path) -> None:
        f = tmp_path / "f"
        f.write_bytes(b"abc")
        assert sha256_of(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


class TestSchemaDdl:
    def _make_db(self, tmp_path: Path) -> Path:
        path = tmp_path / "x" / "x.sqlite"
        path.parent.mkdir()
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE a (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE b (id INTEGER, a_id INTEGER REFERENCES a(id))")
        conn.execute("CREATE INDEX idx ON b(a_id)")
        conn.commit()
        conn.close()
        return path

    def test_ddl_joins_tables_with_blank_lines(self, tmp_path: Path) -> None:
        ddl = schema_ddl(self._make_db(tmp_path))
        statements = ddl.split("\n\n")
        assert len(statements) == 2
        assert statements[0].startswith("CREATE TABLE a")
        assert statements[1].startswith("CREATE TABLE b")
        assert "INDEX" not in ddl

    def test_connection_is_read_only(self, tmp_path: Path) -> None:
        conn = connect_readonly(self._make_db(tmp_path))
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO a VALUES (1)")
        conn.close()

    def test_db_path(self, tmp_path: Path) -> None:
        assert db_path("foo", tmp_path) == tmp_path / "foo" / "foo.sqlite"


class TestLoadQuestions:
    def test_loads_sql_alias(self, tmp_path: Path) -> None:
        path = tmp_path / "q.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "question_id": 1,
                        "db_id": "d",
                        "question": "Q?",
                        "evidence": "E",
                        "SQL": "SELECT 1",
                        "difficulty": "simple",
                    }
                ]
            )
        )
        (q,) = load_questions(path)
        assert q.sql == "SELECT 1"
        assert q.difficulty == "simple"
