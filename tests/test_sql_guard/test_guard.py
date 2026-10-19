"""Layers 2 to 4 and the public API, checked against a real SQLite engine."""

import hashlib
import os
import sqlite3
import time
from pathlib import Path

import pytest

from src.sql_guard import GuardPolicy, guarded_execute, open_guarded
from src.sql_guard.authorizer import Denial, decide


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "t.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, password_hash TEXT);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, total REAL);
        CREATE VIEW v_users AS SELECT * FROM users;
        INSERT INTO users VALUES (1, 'ana', 'h1'), (2, 'bo', 'h2');
        INSERT INTO orders VALUES (1, 1, 9.5), (2, 2, 20.0);
        """
    )
    conn.commit()
    conn.close()
    return path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_select_works(db: Path) -> None:
    r = guarded_execute(db, "SELECT id, name FROM users ORDER BY id")
    assert r.ok and not r.blocked and r.columns == ["id", "name"]
    assert r.rows == [(1, "ana"), (2, "bo")] and not r.truncated and r.layer is None


def test_replace_function_works(db: Path) -> None:
    assert guarded_execute(db, "SELECT replace(name, 'a', 'A') FROM users").rows[0] == ("AnA",)


def test_sqlite_master_is_readable(db: Path) -> None:
    r = guarded_execute(db, "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
    assert r.rows == [("orders",), ("users",)]


def test_ordinary_sql_errors_are_not_blocks(db: Path) -> None:
    r = guarded_execute(db, "SELECT nope FROM users")
    assert r.error and "no such column" in r.error and not r.blocked and r.layer is None
    assert not guarded_execute(db, "SELECT FROM").blocked
    assert not guarded_execute(db / "missing.db", "SELECT 1").blocked


def test_row_cap_and_truncation(db: Path) -> None:
    sql = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 50) "
        "SELECT x FROM c"
    )
    r = guarded_execute(db, sql, GuardPolicy(max_rows=10))
    assert len(r.rows) == 10 and r.truncated
    r = guarded_execute(db, sql, GuardPolicy(max_rows=50))
    assert len(r.rows) == 50 and not r.truncated


def test_timeout_is_not_a_block(db: Path) -> None:
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
    r = guarded_execute(db, sql, GuardPolicy(timeout_s=0.3))
    assert r.error == "timeout" and r.timed_out and not r.blocked
    assert r.elapsed_s < 3


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO users VALUES (9, 'x', 'y')",
        "UPDATE users SET name = 'x'",
        "DELETE FROM users",
        "DROP TABLE users",
        "CREATE TABLE t (a)",
        "CREATE TEMP TABLE t (a)",
        "ATTACH DATABASE 'x.sqlite' AS x",
        "VACUUM INTO 'x.sqlite'",
        "PRAGMA user_version = 5",
        "PRAGMA database_list",
        "BEGIN",
        "WITH x AS (SELECT 1) DELETE FROM users",
        "WITH x AS (SELECT 1) INSERT INTO users SELECT 9, 'a', 'b'",
        "SELECT 1; DELETE FROM users",
    ],
)
def test_blocked_and_database_untouched(db: Path, sql: str, tmp_path: Path) -> None:
    before = digest(db)
    r = guarded_execute(db, sql)
    assert r.blocked and r.error and r.error.startswith("blocked: ")
    assert r.block_reason and r.layer in {"lexical", "authorizer", "engine"}
    assert digest(db) == before
    assert sorted(os.listdir(tmp_path)) == ["t.db"]


# Which layer stops each probe. Layer 1 is switched off to see what the engine does alone.
@pytest.mark.parametrize(
    ("sql", "authorizer_reason"),
    [
        ("VACUUM INTO 'stolen.sqlite'", "ATTACH is not allowed"),
        ("ATTACH DATABASE 'new.sqlite' AS x", "ATTACH is not allowed"),
        ("CREATE TEMP TABLE t (a)", "writing to sqlite_temp_master"),
        ("PRAGMA database_list", "PRAGMA is not allowed"),
        ("PRAGMA user_version = 3", "PRAGMA is not allowed"),
        # SQLite 3.40 reports the eponymous virtual table as a schema write first; newer
        # releases (as on the CI runner) report the read of the pragma_* table. Both are denied.
        (
            "SELECT * FROM pragma_database_list",
            ("writing to sqlite_master", "pragma_database_list"),
        ),
        (
            "SELECT * FROM pragma_table_info('users')",
            ("writing to sqlite_master", "pragma_table_info"),
        ),
        ("DELETE FROM users", "DELETE is not allowed"),
        ("WITH x AS (SELECT 1) DELETE FROM users", "DELETE is not allowed"),
        ("BEGIN", "transaction control"),
        ("SAVEPOINT a", "SAVEPOINT"),
        ("SELECT load_extension('x')", "function load_extension"),
    ],
)
def test_authorizer_alone_stops_probes(
    db: Path, sql: str, authorizer_reason: str | tuple[str, ...]
) -> None:
    r = guarded_execute(db, sql, lexical=False)
    accepted = (authorizer_reason,) if isinstance(authorizer_reason, str) else authorizer_reason
    assert r.blocked and r.layer == "authorizer"
    assert any(reason in (r.block_reason or "") for reason in accepted), r.block_reason
    assert sorted(p.name for p in db.parent.iterdir()) == ["t.db"]


def test_probes_that_authorizer_does_not_stop(db: Path) -> None:
    """EXPLAIN and a no-op REINDEX are allowed by the engine layer; layer 1 stops them."""
    assert guarded_execute(db, "REINDEX", lexical=False).ok
    assert guarded_execute(db, "EXPLAIN SELECT 1", lexical=False).ok
    assert guarded_execute(db, "REINDEX").blocked


def test_connection_alone_stops_writes_but_not_files(db: Path) -> None:
    """With the authorizer off, mode=ro + query_only still refuse writes; ATTACH has no room."""
    r = guarded_execute(db, "DELETE FROM users", authorizer=False, lexical=False)
    assert r.error and not r.blocked
    r = guarded_execute(db, "ATTACH DATABASE 'new.sqlite' AS x", authorizer=False, lexical=False)
    assert r.error is not None  # SQLITE_LIMIT_ATTACHED = 0
    assert not (db.parent / "new.sqlite").exists()


def test_connection_alone_stops_vacuum_into_but_not_metadata(db: Path) -> None:
    """VACUUM INTO attaches internally, so the ATTACHED limit stops it; PRAGMAs still leak."""
    r = guarded_execute(db, "VACUUM INTO 'stolen.sqlite'", authorizer=False, lexical=False)
    assert r.blocked and r.layer == "limit" and "attached" in (r.block_reason or "")
    assert not (db.parent / "stolen.sqlite").exists()
    assert guarded_execute(db, "PRAGMA database_list", authorizer=False, lexical=False).ok


def test_open_guarded_flags(db: Path) -> None:
    conn = open_guarded(db, authorizer=False)
    try:
        assert conn.execute("PRAGMA query_only").fetchone() == (1,)
        assert conn.getlimit(sqlite3.SQLITE_LIMIT_ATTACHED) == 0
        assert conn.getlimit(sqlite3.SQLITE_LIMIT_LENGTH) == 1_000_000
        assert conn.getlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH) == 100_000
    finally:
        conn.close()


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT length(randomblob(500000000))",
        "SELECT length(hex(zeroblob(2000000)))",
        "SELECT length(replace(hex(zeroblob(400000)), '0', '00000'))",
    ],
)
def test_value_size_limit(db: Path, sql: str) -> None:
    r = guarded_execute(db, sql)
    assert r.blocked and r.layer == "limit" and "too big" in (r.block_reason or "")


def test_oversized_printf_degrades_to_null(db: Path) -> None:
    r = guarded_execute(db, "SELECT printf('%.*c', 5000000, 'x')")
    assert r.rows == [(None,)] and not r.blocked


def test_sql_length_limit(db: Path) -> None:
    sql = "SELECT 1 WHERE 1 IN (" + ",".join(["1"] * 60_000) + ")"
    assert len(sql) > 100_000
    r = guarded_execute(db, sql)
    assert r.blocked and r.layer == "lexical" and "longer" in (r.block_reason or "")
    r = guarded_execute(db, sql, GuardPolicy(max_sql_length=500_000))
    assert r.ok


def test_expression_depth_is_bounded(db: Path) -> None:
    sql = "SELECT " + "(" * 3000 + "1" + ")" * 3000
    started = time.monotonic()
    r = guarded_execute(db, sql)
    assert r.error is not None and time.monotonic() - started < 5


def test_memory_limit_policy_is_applied(db: Path) -> None:
    r = guarded_execute(
        db,
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 3000000) "
        "SELECT group_concat(hex(zeroblob(50)) ) FROM c",
        GuardPolicy(max_memory_bytes=50_000_000, timeout_s=20),
    )
    assert r.blocked and r.layer == "limit"
    # the limit is process-wide: restore a generous one so later tests are unaffected
    guarded_execute(db, "SELECT 1", GuardPolicy(max_memory_bytes=2**40))


class TestPolicy:
    policy = GuardPolicy(
        denied_tables={"Orders"},
        denied_columns={("USERS", "Password_Hash")},
    )

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT password_hash FROM users",
            "SELECT PASSWORD_HASH FROM USERS",
            "SELECT * FROM users",
            "SELECT u.* FROM users AS u",
            "SELECT x.h FROM (SELECT password_hash AS h FROM users) x",
            "WITH c AS (SELECT password_hash FROM users) SELECT * FROM c",
            "SELECT * FROM v_users",
            "SELECT name FROM users WHERE password_hash = 'h1'",
            "SELECT length(password_hash) FROM main.users",
            "SELECT * FROM orders",
            "SELECT count(*) FROM orders",
            "SELECT name FROM users WHERE id IN (SELECT user_id FROM orders)",
            "SELECT 1 UNION SELECT total FROM ORDERS",
        ],
    )
    def test_denied(self, db: Path, sql: str) -> None:
        r = guarded_execute(db, sql, self.policy)
        assert (
            r.blocked and r.layer == "authorizer" and "denied by policy" in (r.block_reason or "")
        )

    @pytest.mark.parametrize(
        "sql",
        ["SELECT id, name FROM users", "SELECT count(*) FROM users", "SELECT name FROM users"],
    )
    def test_allowed(self, db: Path, sql: str) -> None:
        assert guarded_execute(db, sql, self.policy).ok

    def test_policy_off_by_default(self, db: Path) -> None:
        assert guarded_execute(db, "SELECT * FROM users").ok

    def test_names_are_lowercased(self) -> None:
        assert self.policy.denied_tables == {"orders"}
        assert self.policy.denied_columns == {("users", "password_hash")}


def test_decide_is_pure() -> None:
    policy = GuardPolicy()
    assert decide(policy, 20, "pragma_table_info", "cid")
    assert decide(policy, 20, "sqlite_dbpage", "data")
    assert decide(policy, 20, "dbstat", "name")
    assert decide(policy, 31, None, "LOAD_EXTENSION")
    assert decide(policy, 20, "sqlite_master", "sql") is None
    assert decide(policy, 31, None, "replace") is None
    assert Denial().reason is None
