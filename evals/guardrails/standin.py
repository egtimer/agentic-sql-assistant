"""Synthetic stand-in for the BIRD ``student_club`` database (fake rows, built by code).

CI has no BIRD data, but 130 attacks and 38 benign cases are written against
``student_club``. This module builds a small database with the same table and column names
those cases reference, so the case SQL stays unchanged and the guard is exercised on the
same statements. The DDL and rows are written here; nothing is copied from BIRD.

The row counts are chosen so the resource cases behave as on the real data: six-way
self-join of ``expense`` and three-way join of ``zip_code`` are far beyond the guard's
limits, and the benign queries return rows. Only the guard's behaviour is comparable with
the real database, not result values.
"""

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE major (
    major_id TEXT PRIMARY KEY, major_name TEXT, department TEXT, college TEXT
);
CREATE TABLE zip_code (
    zip_code INTEGER PRIMARY KEY, type TEXT, city TEXT, county TEXT, state TEXT
);
CREATE TABLE member (
    member_id TEXT PRIMARY KEY, first_name TEXT, last_name TEXT, email TEXT, position TEXT,
    t_shirt_size TEXT, phone TEXT, zip INTEGER REFERENCES zip_code(zip_code),
    link_to_major TEXT REFERENCES major(major_id)
);
CREATE TABLE event (
    event_id TEXT PRIMARY KEY, event_name TEXT, event_date TEXT, type TEXT, notes TEXT,
    location TEXT, status TEXT
);
CREATE TABLE attendance (
    link_to_event TEXT REFERENCES event(event_id),
    link_to_member TEXT REFERENCES member(member_id)
);
CREATE TABLE budget (
    budget_id TEXT PRIMARY KEY, category TEXT, spent REAL, remaining REAL, amount INTEGER,
    event_status TEXT, link_to_event TEXT REFERENCES event(event_id)
);
CREATE TABLE expense (
    expense_id TEXT PRIMARY KEY, expense_description TEXT, expense_date TEXT, cost REAL,
    approved TEXT, link_to_member TEXT REFERENCES member(member_id),
    link_to_budget TEXT REFERENCES budget(budget_id)
);
CREATE TABLE income (
    income_id TEXT PRIMARY KEY, date_received TEXT, amount INTEGER, source TEXT, notes TEXT,
    link_to_member TEXT REFERENCES member(member_id)
);
"""
POSITIONS = ["Member", "Member", "Member", "Treasurer", "Secretary", "Vice President"]
FIRST = ["Ada", "Ben", "Cleo", "Dara", "Eli", "Faye", "Gus", "Hana", "Ivan", "Jade", "Kai", "Lena"]
LAST = ["Alder", "Birch", "Cedar", "Dale", "Elm", "Fern", "Grove", "Hale", "Iris", "Jones"]
N_MEMBERS = 12
N_EVENTS = 6
N_EXPENSES = 32
N_ZIP = 1000


def build_student_club(path: Path) -> None:
    """Create the stand-in at ``path`` (rollback journal, not WAL)."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode = DELETE")
    conn.executescript(SCHEMA)
    conn.executemany(
        "INSERT INTO major VALUES (?, ?, ?, ?)",
        [(f"maj{i}", f"Major {i}", f"Dept {i % 3}", f"College {i % 2}") for i in range(5)],
    )
    conn.executemany(
        "INSERT INTO zip_code VALUES (?, ?, ?, ?, ?)",
        [(10000 + i, "Standard", f"City {i % 40}", f"County {i % 9}", "XX") for i in range(N_ZIP)],
    )
    conn.executemany(
        "INSERT INTO member VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                f"mem{i:03d}",
                FIRST[i % len(FIRST)],
                LAST[i % len(LAST)],
                f"{FIRST[i % len(FIRST)].lower()}{i}@example.com",
                POSITIONS[i % len(POSITIONS)],
                ("S", "M", "L")[i % 3],
                f"555-01{i:02d}",
                10000 + i,
                f"maj{i % 5}",
            )
            for i in range(N_MEMBERS)
        ],
    )
    conn.executemany(
        "INSERT INTO event VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (f"ev{i}", f"Event {i}", f"2020-{i + 1:02d}-15T12:00:00", "Meeting", "", "Hall", "Open")
            for i in range(N_EVENTS)
        ],
    )
    conn.executemany(
        "INSERT INTO attendance VALUES (?, ?)",
        [(f"ev{i % N_EVENTS}", f"mem{i % N_MEMBERS:03d}") for i in range(20)],
    )
    conn.executemany(
        "INSERT INTO budget VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (f"bud{i}", "Food", 10.0 * i, 100.0 - 10.0 * i, 100, "Open", f"ev{i}")
            for i in range(N_EVENTS)
        ],
    )
    # expenses belong to the first 8 members only, so the last 4 members have none
    conn.executemany(
        "INSERT INTO expense VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                f"exp{i:03d}",
                f"Item {i}",
                f"2020-0{i % 9 + 1}-10",
                round(7.5 * (i + 1) % 120, 2),
                "true",
                f"mem{i % 8:03d}",
                f"bud{i % N_EVENTS}",
            )
            for i in range(N_EXPENSES)
        ],
    )
    conn.executemany(
        "INSERT INTO income VALUES (?, ?, ?, ?, ?, ?)",
        [
            (f"inc{i}", f"2020-0{i + 1}-01", 50 * (i + 1), "Dues", "", f"mem{i:03d}")
            for i in range(6)
        ],
    )
    conn.commit()
    conn.close()
