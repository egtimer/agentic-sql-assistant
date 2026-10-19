"""Rendered ``(system, prefix, user)`` triples of generate/check/respond, frozen in a snapshot.

Agent v2 without history and without memories must send exactly these prompts (the frozen
prompts of ADR-004). The snapshot was recorded before the memory feature existed; regenerate
it only on purpose, with ``UPDATE_SNAPSHOTS=1``.
"""

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from src.llm import CallUsage, Completion, reset_client, set_client
from src.sql_agent.graph import run_sql_agent

SNAPSHOT = Path(__file__).parent / "snapshots" / "prompts_no_memory.json"


class Recorder:
    def __init__(self, **script: list[str]) -> None:
        self.script = {k: list(v) for k, v in script.items()}
        self.triples: list[dict[str, Any]] = []

    def complete(
        self, system: str | None, user: str, *, tag: str, prefix: str | None = None
    ) -> Completion:
        self.triples.append({"tag": tag, "system": system, "prefix": prefix, "user": user})
        queue = self.script.get(tag, [""])
        text = queue.pop(0) if len(queue) > 1 else queue[0]
        return Completion(text=text, usage=CallUsage(tag=tag))


def sql(q: str) -> str:
    return f"```sql\n{q}\n```"


def build_financial(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE account (account_id INTEGER PRIMARY KEY, frequency TEXT, date TEXT);
        CREATE TABLE loan (
            loan_id INTEGER PRIMARY KEY, account_id INTEGER REFERENCES account(account_id),
            amount INTEGER, duration INTEGER, status TEXT);
        INSERT INTO account VALUES (1, 'POPLATEK MESICNE', '1993-01-05'),
            (2, 'POPLATEK TYDNE', '1995-03-11'), (3, 'POPLATEK MESICNE', '1996-07-30');
        INSERT INTO loan VALUES (10, 1, 80952, 24, 'A'), (11, 2, 30276, 12, 'B'),
            (12, 3, 318480, 60, 'A');
        """
    )
    conn.commit()
    conn.close()


def build_schools(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE frpm (
            "CDSCode" TEXT PRIMARY KEY, "County Name" TEXT, "Enrollment (K-12)" REAL,
            "Free Meal Count (K-12)" REAL);
        INSERT INTO frpm VALUES ('01', 'Alameda', 1087.0, 565.0), ('02', 'Kern', 395.0, 186.0);
        """
    )
    conn.commit()
    conn.close()


FIXTURES: list[dict[str, Any]] = [
    {
        "name": "financial_with_evidence_fix_and_respond",
        "build": build_financial,
        "question": "How many accounts with a loan of more than 12 months have monthly issuance?",
        "evidence": "monthly issuance refers to frequency = 'POPLATEK MESICNE'; "
        "loan of more than 12 months refers to duration > 12",
        "script": {
            "generate": [sql("SELECT COUNT(*) FROM loan WHERE nope > 12")],
            "check": [
                f"No such column.\n{sql('SELECT COUNT(*) FROM loan WHERE duration > 12')}",
                "OK",
            ],
            "respond": ["There are 2."],
        },
        "skip_respond": False,
    },
    {
        "name": "schools_no_evidence_ok",
        "build": build_schools,
        "question": "What is the free meal rate for Alameda?",
        "evidence": "",
        "script": {
            "generate": [
                sql(
                    'SELECT "Free Meal Count (K-12)" / "Enrollment (K-12)" FROM frpm '
                    "WHERE \"County Name\" = 'Alameda'"
                )
            ],
            "check": ["Looks right.\n\nOK"],
        },
        "skip_respond": True,
    },
    {
        "name": "financial_blocked_then_rephrased",
        "build": build_financial,
        "question": "List the account ids.",
        "evidence": "  ",
        "script": {
            "generate": [sql("VACUUM INTO 'x.sqlite'")],
            "check": [f"Not allowed.\n{sql('SELECT account_id FROM account')}", "OK"],
            "respond": ["1, 2 and 3."],
        },
        "skip_respond": False,
    },
]


def render(tmp_path: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for fixture in FIXTURES:
        db = tmp_path / f"{fixture['name']}.sqlite"
        fixture["build"](db)
        recorder = Recorder(**fixture["script"])
        token = set_client(recorder)
        try:
            run_sql_agent(
                fixture["question"],
                fixture["evidence"],
                db,
                skip_respond=fixture["skip_respond"],
            )
        finally:
            reset_client(token)
        out[fixture["name"]] = recorder.triples
    return out


def test_prompts_without_memory_match_the_snapshot(tmp_path: Path) -> None:
    rendered = render(tmp_path)
    if os.environ.get("UPDATE_SNAPSHOTS") == "1":
        SNAPSHOT.write_text(json.dumps(rendered, indent=1, ensure_ascii=False) + "\n", "utf-8")
        pytest.skip("snapshot rewritten")
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert rendered == expected


def test_snapshot_covers_every_prompting_node() -> None:
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    tags = {t["tag"] for triples in expected.values() for t in triples}
    assert tags == {"generate", "check", "respond"}
