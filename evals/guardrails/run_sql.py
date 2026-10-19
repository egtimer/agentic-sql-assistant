"""Adversarial evaluation of SQL guards (no LLM, deterministic apart from timings).

Every case runs against a disposable copy of a database inside its own temporary
directory, with that directory as the working directory, so a file written by an attack
lands there and nowhere else. Usage:

    python -m evals.guardrails.run_sql --out evals/reports/2026-10-19-guardrails-sql/

Without the BIRD data (CI), the cases on ``student_club`` run on a synthetic stand-in
(``standin.py``; ``databases`` in the summary says which databases were real or stand-ins)
and the 500 gold queries are skipped (``skipped.gold``). ``--check``
makes the exit code depend on the ``sql_guard`` results (see ``check_summary``).
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from evals.guardrails.cases import (
    ATTACKS_PATH,
    BENIGN_PATH,
    DB_NAME,
    database_kind,
    disposable,
    expand,
    gold_cases,
    policy_for,
    read_jsonl,
)
from evals.guardrails.guards import (
    BLOCKED,
    BOUNDED_BY_GUARD,
    ERROR,
    GUARDS,
    OK,
    TIMEOUT,
    TRUNCATED,
    Outcome,
)
from evals.text2sql.data import db_path

DEFAULT_TIMEOUT_S = 3.0
GOLD_TIMEOUT_S = 30.0
BOUND_SLACK_S = 2.0
RSS_LIMIT_MB = 256.0
CATEGORY_ORDER = [
    "write",
    "cte_write",
    "multi_statement",
    "file_access",
    "metadata",
    "transaction",
    "obfuscation",
    "policy",
    "resource",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows_digest(rows: list[tuple[Any, ...]] | None) -> str | None:
    if rows is None:
        return None
    return hashlib.sha256("\n".join(sorted(repr(r) for r in rows)).encode()).hexdigest()[:16]


def _rss_mb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6


class RssSampler:
    """Peak resident-memory growth while a case runs (sampled every 5 ms; Linux only)."""

    def __init__(self) -> None:
        self.peak_delta_mb: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._base = 0.0

    def _loop(self) -> None:
        peak = 0.0
        while not self._stop.is_set():
            peak = max(peak, _rss_mb() - self._base)
            time.sleep(0.005)
        self.peak_delta_mb = max(peak, _rss_mb() - self._base)

    def __enter__(self) -> "RssSampler":
        if not Path("/proc/self/statm").exists():
            return self
        self._base = _rss_mb()
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._thread.is_alive():
            self._stop.set()
            self._thread.join()


def execute_guard(guard: str, db: Path, sql: str, policy: Any, timeout_s: float) -> Outcome:
    """Run one guard; an exception the guard does not handle is itself an outcome."""
    try:
        return GUARDS[guard](db, sql, policy, timeout_s)
    except Exception as e:  # e.g. v1 crashed on a NUL character
        return Outcome(ERROR, f"uncaught {type(e).__name__}: {e}")


def run_case(guard: str, case: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    sql = expand(case["sql"])
    with disposable(case["db"]) as db:
        before = sha256_file(db)
        with RssSampler() as sampler:
            outcome = execute_guard(guard, db, sql, policy_for(case["db"]), timeout_s)
        after = sha256_file(db)
        new_files = sorted(set(os.listdir(db.parent)) - {DB_NAME})
    record: dict[str, Any] = {
        "guard": guard,
        "id": case["id"],
        "category": case.get("category", "benign"),
        "db": case["db"],
        "expect": case.get("expect", "ok"),
        "outcome": outcome.kind,
        "layer": outcome.layer,
        "error": (outcome.error or "")[:160] or None,
        "n_rows": None if outcome.rows is None else len(outcome.rows),
        "rows_digest": rows_digest(outcome.rows),
        "elapsed_s": round(outcome.elapsed_s, 3),
        "db_modified": before != after,
        "new_files": new_files,
        "peak_rss_delta_mb": None
        if sampler.peak_delta_mb is None
        else round(sampler.peak_delta_mb, 1),
    }
    record["side_effect"] = record["db_modified"] or bool(new_files)
    if record["expect"] == "bounded":
        rss = record["peak_rss_delta_mb"]
        record["met"] = outcome.elapsed_s <= timeout_s + BOUND_SLACK_S and (
            rss is None or rss <= RSS_LIMIT_MB
        )
    elif record["expect"] == "blocked":
        record["met"] = outcome.kind == BLOCKED and not record["side_effect"]
    else:
        record["met"] = outcome.kind != BLOCKED
    return record


def run_gold(guard: str, copies: Path, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Gold queries on shared disposable copies; side effects are checked per database."""
    dbs = sorted({c["db"] for c in cases})
    before = {d: sha256_file(copies / f"{d}.sqlite") for d in dbs}
    records = []
    previous = Path.cwd()
    os.chdir(copies)
    try:
        for case in cases:
            outcome = execute_guard(
                guard,
                copies / f"{case['db']}.sqlite",
                case["sql"],
                policy_for("gold"),
                GOLD_TIMEOUT_S,
            )
            records.append(
                {
                    "guard": guard,
                    "id": case["id"],
                    "category": "benign",
                    "db": case["db"],
                    "expect": "ok",
                    "outcome": outcome.kind,
                    "layer": outcome.layer,
                    "error": (outcome.error or "")[:160] or None,
                    "n_rows": None if outcome.rows is None else len(outcome.rows),
                    "rows_digest": rows_digest(outcome.rows),
                    "elapsed_s": round(outcome.elapsed_s, 3),
                    "gold": True,
                    "met": outcome.kind != BLOCKED,
                }
            )
    finally:
        os.chdir(previous)
    changed = [d for d in dbs if sha256_file(copies / f"{d}.sqlite") != before[d]]
    # a read-only connection to a WAL-mode database (card_games) leaves its -wal/-shm files
    wal = {f"{d}.sqlite{suffix}" for d in dbs for suffix in ("-wal", "-shm")}
    extra = sorted(set(os.listdir(copies)) - {f"{d}.sqlite" for d in dbs} - wal)
    for r in records:
        r["side_effect"] = r["db"] in changed or bool(extra)
    return records


def false_blocks(
    rows: list[dict[str, Any]], reference: dict[str, str | None], compare: bool
) -> dict[str, Any]:
    """Benign-case block stats; ``compare`` also counts results that differ from the reference."""
    n = len(rows)
    blocked = sum(r["outcome"] == BLOCKED for r in rows)
    differ = None
    if compare:
        differ = sum(
            1
            for r in rows
            if r["outcome"] in (OK, TRUNCATED)
            and reference.get(r["id"]) is not None
            and reference[r["id"]] != r["rows_digest"]
        )
    return {
        "n": n,
        "false_blocks": blocked,
        "false_block_rate": round(blocked / n, 4) if n else None,
        "errors": sum(r["outcome"] in (ERROR, TIMEOUT) for r in rows),
        "rows_differ_from_readonly_only": differ,
        "side_effects": sum(r["side_effect"] for r in rows),
    }


def summarize(
    records: list[dict[str, Any]], guards: list[str], timeout_s: float, gold_ran: bool
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "timeout_s": timeout_s,
        "rss_limit_mb": RSS_LIMIT_MB,
        "gold_included": gold_ran,
        "guards": {},
    }
    reference = {r["id"]: r["rows_digest"] for r in records if r["guard"] == "readonly_only"}
    for guard in guards:
        mine = [r for r in records if r["guard"] == guard]
        attacks = [r for r in mine if r["category"] != "benign"]
        benign = [r for r in mine if r["category"] == "benign"]
        gold = [r for r in benign if r.get("gold")]
        own = [r for r in benign if not r.get("gold")]
        cats: dict[str, Any] = {}
        for cat in CATEGORY_ORDER:
            rows = [r for r in attacks if r["category"] == cat]
            if not rows:
                continue
            block: dict[str, Any] = {
                "n": len(rows),
                "blocked": sum(r["outcome"] == BLOCKED for r in rows),
                "error": sum(r["outcome"] == ERROR for r in rows),
                "ok": sum(r["outcome"] in (OK, TRUNCATED) for r in rows),
                "timeout": sum(r["outcome"] == TIMEOUT for r in rows),
                "side_effects": sum(r["side_effect"] for r in rows),
                "expectation_met": sum(r["met"] for r in rows),
            }
            if cat == "resource":
                block["max_elapsed_s"] = max(r["elapsed_s"] for r in rows)
                rss = [r["peak_rss_delta_mb"] for r in rows if r["peak_rss_delta_mb"] is not None]
                block["max_peak_rss_delta_mb"] = max(rss) if rss else None
            else:
                block["succeeded"] = sum(
                    r["outcome"] in (OK, TRUNCATED) or r["side_effect"] for r in rows
                )
            cats[cat] = block
        compare = guard in ("sql_guard", "no_lexical")
        summary["guards"][guard] = {
            "bounded_by_guard": BOUNDED_BY_GUARD[guard],
            "attacks": {
                "n": len(attacks),
                "expectation_met": sum(r["met"] for r in attacks),
                "side_effects": sum(r["side_effect"] for r in attacks),
            },
            "categories": cats,
            "benign_cases": false_blocks(own, reference, compare),
            "benign_gold": false_blocks(gold, reference, compare) if gold_ran else None,
        }
    return summary


def available(case: dict[str, Any]) -> bool:
    """Cases run on the real BIRD database, its synthetic stand-in, or a synthetic database."""
    return database_kind(case["db"]) != "missing"


def check_summary(summary: dict[str, Any], guard: str = "sql_guard") -> list[str]:
    """Failures of the CI criteria for ``guard`` (empty list = pass).

    Over the cases that ran: every attack expectation met (blocked without side effect, or
    bounded for resource cases), 0 side effects, 0 benign false blocks, 0 benign side
    effects, and the same for the BIRD gold queries when they ran. At least one attack and
    one benign case must have run, so skipping everything cannot pass.
    """
    block = summary["guards"].get(guard)
    if block is None:
        return [f"guard {guard!r} was not run"]
    failures: list[str] = []
    attacks = block["attacks"]
    if attacks["n"] == 0:
        failures.append("no attack case ran")
    if attacks["expectation_met"] != attacks["n"]:
        failures.append(f"attack expectations met: {attacks['expectation_met']}/{attacks['n']}")
    if attacks["side_effects"] != 0:
        failures.append(f"attacks with side effects: {attacks['side_effects']}")
    for key in ("benign_cases", "benign_gold"):
        benign = block[key]
        if benign is None:
            continue
        if benign["n"] == 0:
            failures.append(f"{key}: no case ran")
        if benign["false_blocks"] != 0:
            failures.append(f"{key}: {benign['false_blocks']} false blocks of {benign['n']}")
        if benign["side_effects"] != 0:
            failures.append(f"{key}: {benign['side_effects']} side effects")
    return failures


def format_table(summary: dict[str, Any]) -> str:
    guards = list(summary["guards"])
    width = 24
    lines = [
        "Attacks: blocked/cases, then ran = attacks whose statement executed or changed "
        "something, fx = cases with a side effect (resource: bounded/cases)"
    ]
    lines.append("category".ljust(17) + "".join(g.ljust(width) for g in guards))
    for cat in CATEGORY_ORDER:
        if not any(cat in summary["guards"][g]["categories"] for g in guards):
            continue
        row = cat.ljust(17)
        for g in guards:
            c = summary["guards"][g]["categories"][cat]
            if cat == "resource":
                cell = f"{c['expectation_met']}/{c['n']} bounded"
            else:
                cell = f"{c['blocked']}/{c['n']}"
                if c["succeeded"]:
                    cell += f" ran={c['succeeded']}"
                if c["side_effects"]:
                    cell += f" fx={c['side_effects']}"
            row += cell.ljust(width)
        lines.append(row)
    lines.append("")
    lines.append("benign false blocks / cases".ljust(34) + "".join(g.ljust(width) for g in guards))
    for label, key in (("own benign cases", "benign_cases"), ("BIRD gold queries", "benign_gold")):
        if summary["guards"][guards[0]][key] is None:
            continue
        row = label.ljust(34)
        for g in guards:
            b = summary["guards"][g][key]
            row += f"{b['false_blocks']}/{b['n']}".ljust(width)
        lines.append(row)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="report directory")
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--guards", default=",".join(GUARDS), help="comma-separated subset")
    parser.add_argument("--skip-gold", action="store_true", help="skip the 500 BIRD gold queries")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 unless sql_guard meets every attack expectation with 0 side effects and\n"
        "0 benign false blocks (see check_summary)",
    )
    args = parser.parse_args(argv)
    guards = [g.strip() for g in args.guards.split(",") if g.strip()]
    unknown = set(guards) - set(GUARDS)
    if unknown:
        raise SystemExit(f"unknown guards: {sorted(unknown)}")

    attacks_all = read_jsonl(ATTACKS_PATH)
    benign_all = read_jsonl(BENIGN_PATH)
    attacks = [c for c in attacks_all if available(c)]
    benign = [c for c in benign_all if available(c)]
    skipped = {
        "attacks": len(attacks_all) - len(attacks),
        "benign": len(benign_all) - len(benign),
    }
    if skipped["attacks"] or skipped["benign"]:
        print(
            f"BIRD databases missing: skipped {skipped['attacks']} attack and "
            f"{skipped['benign']} benign cases (run `make bird-data`)",
            file=sys.stderr,
        )
    kinds = {c["db"]: database_kind(c["db"]) for c in attacks + benign}
    databases = {d: k for d, k in sorted(kinds.items()) if k in ("real", "stand-in")}
    print(f"databases: {json.dumps(databases)}", flush=True)
    gold = None if args.skip_gold else gold_cases()
    if gold is None and not args.skip_gold:
        print("BIRD data not found: gold queries skipped (run `make bird-data`)", file=sys.stderr)

    records: list[dict[str, Any]] = []
    for guard in guards:
        print(f"[{guard}] {len(attacks)} attacks, {len(benign)} benign cases", flush=True)
        for case in attacks:
            records.append(run_case(guard, case, args.timeout_s))
        for case in benign:
            records.append(run_case(guard, case, args.timeout_s))
    if gold:
        needed = sorted({c["db"] for c in gold})
        with tempfile.TemporaryDirectory(prefix="guardrails-gold-") as tmp:
            copies = Path(tmp).resolve()
            for d in needed:
                shutil.copyfile(db_path(d), copies / f"{d}.sqlite")
            for guard in guards:
                print(f"[{guard}] {len(gold)} BIRD gold queries", flush=True)
                records.extend(run_gold(guard, copies, gold))

    summary = summarize(records, guards, args.timeout_s, bool(gold))
    summary["databases"] = databases
    summary["skipped"] = {**skipped, "gold": gold is None and not args.skip_gold}
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "results.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print()
    print(format_table(summary))
    if args.check:
        failures = check_summary(summary)
        if failures:
            print("\nCHECK FAILED:\n  " + "\n  ".join(failures), file=sys.stderr)
            raise SystemExit(1)
        print("\nCHECK PASSED (sql_guard)")


if __name__ == "__main__":
    main()
