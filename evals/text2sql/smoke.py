"""Offline end-to-end smoke test of the text-to-SQL runner (no network, no BIRD data).

Builds a tiny SQLite database and four questions, then runs ``baseline`` and ``agent_v2``
through the real runner (``run_experiment``: predictions, scorer, v2 graph with check and
fix loop, summary, ``run.json``) with a scripted client that replays fixed answers. The
client lives here, not in ``src/llm``, so the production provider list is untouched: it is
test code that only the CI job and the tests import. Exit code 1 when a number differs from
the expected one. Usage:

    python -m evals.text2sql.smoke --out /tmp/smoke-run
"""

import argparse
import json
import re
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

from evals.text2sql.data import Question
from evals.text2sql.run import run_experiment
from src.llm import CallUsage, Completion, LLMConfig, UsageTracker

DB_ID = "toy"
RIGHT = "SELECT v FROM t WHERE id = {n}"
WRONG = "SELECT v FROM t WHERE id = 999"
BAD_COLUMN = "SELECT nope FROM t WHERE id = {n}"


def fence(sql: str, tag: str = "sql") -> str:
    return f"```{tag}\n{sql}\n```"


# question id -> tag -> answers in call order (the last one repeats)
SCRIPT: dict[int, dict[str, list[str]]] = {
    1: {  # right first time; the reviewer approves with a bold OK
        "baseline": [fence(RIGHT.format(n=1))],
        "generate": [fence(RIGHT.format(n=1))],
        "check": ["The query matches the question.\n\n**OK**"],
    },
    2: {  # wrong first time; the reviewer fixes it (sqlite fence, then OK)
        "baseline": [fence(WRONG)],
        "generate": [fence(WRONG)],
        "check": [
            f"The filter is wrong.\n\n{fence(RIGHT.format(n=2), 'sqlite')}",
            "Now it is right.\n\nOK",
        ],
    },
    3: {  # SQL error first time; error recovery through the fix loop
        "baseline": [fence(RIGHT.format(n=3))],
        "generate": [fence(BAD_COLUMN.format(n=3))],
        "check": [f"No such column.\n{fence(RIGHT.format(n=3))}", "OK"],
    },
    4: {  # wrong and the reviewer does not notice (trailing OK)
        "baseline": [fence(WRONG)],
        "generate": [fence(WRONG)],
        "check": ["The result looks plausible and matches the evidence.\n\nOK"],
    },
}
EXPECTED = {
    "baseline": {"correct": 2, "llm_calls": 4},
    "agent_v2": {"correct": 3, "llm_calls": 10, "with_fix": 2, "unparsed": 0},
}


class ScriptedClient:
    """Replays ``SCRIPT``; the question is found through its ``[Q<id>]`` marker."""

    def __init__(self, tracker: UsageTracker) -> None:
        self.tracker = tracker
        self.calls: dict[tuple[int, str], int] = {}

    def complete(
        self, system: str | None, user: str, *, tag: str, prefix: str | None = None
    ) -> Completion:
        self.tracker.check()
        match = re.search(r"\[Q(\d+)\]", (prefix or "") + user)
        if match is None:
            raise AssertionError(f"no question marker in the {tag} prompt")
        qid = int(match.group(1))
        answers = SCRIPT[qid][tag]
        n = self.calls.get((qid, tag), 0)
        self.calls[qid, tag] = n + 1
        usage = CallUsage(tag=tag, input_tokens=100, output_tokens=20, stop_reason="end_turn")
        self.tracker.record(usage)
        return Completion(text=answers[min(n, len(answers) - 1)], usage=usage)


def build_fixture(root: Path) -> tuple[Path, list[Question]]:
    """Databases directory and the questions (gold = ``RIGHT`` for the question id)."""
    path = root / "dbs" / DB_ID / f"{DB_ID}.sqlite"
    path.parent.mkdir(parents=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)")
    conn.executemany("INSERT INTO t VALUES (?, ?)", [(i, i * 10) for i in range(1, 10)])
    conn.commit()
    conn.close()
    difficulties = ["simple", "moderate", "challenging", "simple"]
    questions = [
        Question(
            question_id=qid,
            db_id=DB_ID,
            question=f"[Q{qid}] What is the value for id {qid}?",
            evidence="",
            SQL=RIGHT.format(n=qid),
            difficulty=difficulties[qid - 1],  # type: ignore[arg-type]
        )
        for qid in SCRIPT
    ]
    return root / "dbs", questions


def verify(summary: dict[str, Any], run_dir: Path) -> list[str]:
    """Differences between the run and ``EXPECTED`` (empty = pass)."""
    problems: list[str] = []

    def expect(label: str, got: Any, want: Any) -> None:
        if got != want:
            problems.append(f"{label}: got {got!r}, expected {want!r}")

    systems = summary["systems"]
    for name, want in EXPECTED.items():
        s = systems[name]
        expect(f"{name} n", s["n"], len(SCRIPT))
        expect(f"{name} correct", s["correct"], want["correct"])
        expect(
            f"{name} llm calls", round(s["llm_calls_mean_per_question"] * s["n"]), want["llm_calls"]
        )
    v2 = systems["agent_v2"]["v2"]
    expect("agent_v2 questions with a fix", round(v2["frac_with_fix"] * v2["n"]), 2)
    expect("agent_v2 unparsed checks", v2["unparsed_checks"], 0)
    expect("aborted", summary["aborted"], None)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    expect("run.json status", run["status"], "completed")
    predictions = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    texts = [v["text"] for r in predictions for v in r.get("verdicts", [])]
    expect("verdict texts recorded", bool(texts) and all(texts), True)
    return problems


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, help="run directory (default: a temporary one)")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="t2s-smoke-") as tmp:
        databases_dir, questions = build_fixture(Path(tmp))
        run_dir = args.out.resolve() if args.out else Path(tmp) / "run"
        summary = run_experiment(
            config=LLMConfig(provider="ollama", model="scripted"),
            systems=["baseline", "agent_v2"],
            questions=questions,
            run_dir=run_dir,
            run_id="smoke",
            split="dev",
            client_factory=lambda config, tracker: ScriptedClient(tracker),
            databases_dir=databases_dir,
            skip_respond=True,
            bootstrap_resamples=200,
            launched_by="evals.text2sql.smoke",
        )
        problems = verify(summary, run_dir)
    if problems:
        print("SMOKE FAILED:\n  " + "\n  ".join(problems), file=sys.stderr)
        raise SystemExit(1)
    print("SMOKE PASSED: baseline 2/4, agent_v2 3/4, 2 fixes, 0 unparsed checks")


if __name__ == "__main__":
    main()
