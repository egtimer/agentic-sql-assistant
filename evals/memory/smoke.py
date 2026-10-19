"""Offline end-to-end smoke test of the two memory evaluations (no network, no BIRD data).

Builds a tiny SQLite database and seven questions, then runs the real long-term and follow-up
runners (teach, recall, the agent v2 graph with a checkpointer thread, the scorer, the
statistics, ``run.json`` and the budget) with a scripted client that behaves like a model that
can only answer correctly when the prompt carries the right information:

- long-term: a question is answered right when its definition is in the prompt (evidence or
  a recalled memory); the teaching of one question yields no memory and the oracle is wrong
  on one question, so the three conditions differ.
- follow-up: a follow-up is answered right only when the conversation block is in the prompt
  (one pair), never (one pair), or not needed (one pair).

The client lives here, not in ``src/llm``. Exit code 1 when a number differs from the
expected one. Usage:

    python -m evals.memory.smoke --out /tmp/memory-smoke
"""

import argparse
import json
import re
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

from evals.memory.run_followups import FollowUp, run_followups
from evals.memory.run_longterm import run_longterm
from evals.text2sql.data import Question
from src.llm import CallUsage, Completion, LLMConfig, Pricing, UsageTracker
from src.sql_agent.prompts import HISTORY_HEADER

DB_ID = "toy"
COST_PER_CALL = 0.001
CONFIG = LLMConfig(
    provider="ollama", model="scripted", pricing=Pricing(input_per_mtok=0.0, output_per_mtok=0.0)
)
WRONG = "SELECT COUNT(*) FROM t"
GOLD = {
    1: "SELECT COUNT(*) FROM t WHERE v >= 50",
    2: "SELECT COUNT(*) FROM t WHERE v <= 20",
    3: "SELECT SUM(v) FROM t WHERE v BETWEEN 30 AND 40",
    4: "SELECT COUNT(*) FROM t WHERE v = 15",
    5: "SELECT COUNT(*) FROM t WHERE kind = 'small'",
    6: "SELECT COUNT(*) FROM t WHERE kind = 'big'",
    7: "SELECT MAX(v) FROM t",
}
TEXT = {
    1: ("How many bigshots are there?", "bigshots refers to v >= 50"),
    2: ("How many tiny rows are there?", "tiny refers to v <= 20"),
    3: ("What is the sum of v for midsize rows?", "midsize refers to v BETWEEN 30 AND 40"),
    4: ("How many odd rows are there?", "odd refers to v = 15"),
    5: ("How many small rows are there?", "small refers to kind = 'small'"),
    6: ("How many big rows are there?", "big refers to kind = 'big'"),
    7: ("What is the highest value?", "highest refers to MAX(v)"),
}
DIFFICULTY = ["simple", "moderate", "challenging", "simple", "moderate", "simple", "challenging"]
NO_EXTRACTION = {"midsize"}  # teaching this term yields no memory
ORACLE_WRONG = {4}  # the model fails even with the evidence
DEFINITION_IN_PROMPT = {1: "v >= 50", 2: "v <= 20", 3: "v BETWEEN 30 AND 40", 4: "v = 15"}
FOLLOWUPS = [
    # history needed and enough
    FollowUp(
        id="fu-a",
        source_question_id=5,
        db_id=DB_ID,
        difficulty="moderate",
        turn1="[Q5] Show the rows of kind small.",
        turn2="[Q5] How many of those are there?",
    ),
    # history needed but the model fails anyway
    FollowUp(
        id="fu-b",
        source_question_id=6,
        db_id=DB_ID,
        difficulty="simple",
        turn1="[Q6] Show the rows of kind big.",
        turn2="[Q6] How many of those are there?",
    ),
    # turn 2 is self-contained
    FollowUp(
        id="fu-c",
        source_question_id=7,
        db_id=DB_ID,
        difficulty="challenging",
        turn1="[Q7] Show the value column.",
        turn2="[Q7] What is the highest value?",
    ),
]
EXPECTED_LONGTERM = {
    "correct": {"no_memory": 0, "memory": 2, "oracle": 3},
    "llm_calls": {"no_memory": 8, "memory": 8, "oracle": 8},
    "teaching_turns": 4,
    "memories_extracted": 3,
    "zero_memory_turns": 1,
    "gap_closed": 2 / 3,
    "retrieval_hits": 3,  # q1, q2 and q4; q3 has no memory of its own
    "total_calls": 4 + 24,
}
EXPECTED_FOLLOWUPS = {
    "correct": {"turn2_alone": 1, "with_history": 2, "original": 3},
    "llm_calls": {"turn2_alone": 6, "with_history": 12, "original": 6},
    "total_calls": 24,
}


def fence(sql: str) -> str:
    return f"```sql\n{sql}\n```"


class ScriptedClient:
    """Answers by reading the prompt, so the outcome depends on what the prompt contains."""

    def __init__(self, tracker: UsageTracker) -> None:
        self.tracker = tracker
        self.prompts: list[tuple[str, str]] = []

    def complete(
        self, system: str | None, user: str, *, tag: str, prefix: str | None = None
    ) -> Completion:
        self.tracker.check()
        self.prompts.append((tag, user))
        if tag == "memorize":
            text = self.memorize(user)
        elif tag == "generate":
            text = fence(self.generate(user))
        elif tag == "check":
            text = "OK"
        else:
            text = "done"
        usage = CallUsage(
            tag=tag,
            input_tokens=100,
            output_tokens=20,
            stop_reason="end_turn",
            cost_usd=COST_PER_CALL,
        )
        self.tracker.record(usage)
        return Completion(text=text, usage=usage)

    @staticmethod
    def memorize(user: str) -> str:
        match = re.search(r"database: (\w+) refers to (.+)$", user.strip())
        if match is None or match.group(1) in NO_EXTRACTION:
            return "[]"
        return json.dumps([{"term": match.group(1), "definition": match.group(2)}])

    @staticmethod
    def generate(user: str) -> str:
        qid = int(re.findall(r"\[Q(\d+)\]", user)[-1])
        question = user.strip().splitlines()[-1]
        if qid in DEFINITION_IN_PROMPT:
            known = DEFINITION_IN_PROMPT[qid] in user
            return GOLD[qid] if known and qid not in ORACLE_WRONG else WRONG
        if "of those" in question:  # an elliptical follow-up
            return GOLD[qid] if qid == 5 and HISTORY_HEADER in user else WRONG
        return GOLD.get(qid, WRONG)


def build_fixture(root: Path) -> tuple[Path, list[Question]]:
    path = root / "dbs" / DB_ID / f"{DB_ID}.sqlite"
    path.parent.mkdir(parents=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, kind TEXT)")
    conn.executemany(
        "INSERT INTO t VALUES (?, ?, ?)",
        [(i, i * 10, "big" if i * 10 >= 50 else "small") for i in range(1, 10)],
    )
    conn.commit()
    conn.close()
    questions = [
        Question(
            question_id=qid,
            db_id=DB_ID,
            question=f"[Q{qid}] {TEXT[qid][0]}",
            evidence=TEXT[qid][1],
            SQL=GOLD[qid],
            difficulty=DIFFICULTY[qid - 1],  # type: ignore[arg-type]
        )
        for qid in GOLD
    ]
    return root / "dbs", questions


def expect(problems: list[str], label: str, got: Any, want: Any) -> None:
    if got != want:
        problems.append(f"{label}: got {got!r}, expected {want!r}")


def read_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def verify_longterm(summary: dict[str, Any], run_dir: Path) -> list[str]:
    problems: list[str] = []
    want = EXPECTED_LONGTERM
    for name, s in summary["systems"].items():
        expect(problems, f"longterm {name} n", s["n"], 4)
        expect(problems, f"longterm {name} correct", s["correct"], want["correct"][name])
        calls = round(s["llm_calls_mean_per_question"] * s["n"])
        expect(problems, f"longterm {name} llm calls", calls, want["llm_calls"][name])
    t = summary["teaching"]
    expect(problems, "teaching turns", t["n_turns"], want["teaching_turns"])
    expect(problems, "memories extracted", t["memories_extracted"], want["memories_extracted"])
    expect(problems, "zero-memory turns", t["turns_with_zero_memories"], want["zero_memory_turns"])
    gap = summary["gap_closed"]
    expect(problems, "gap closed", round(gap["gap_closed"], 6), round(want["gap_closed"], 6))
    expect(problems, "retrieval recall", summary["retrieval"]["retrieval_recall"], 3 / 4)
    expect(
        problems,
        "retrieval given stored",
        summary["retrieval"]["retrieval_recall_given_stored"],
        1.0,
    )
    expect(problems, "total llm calls", summary["totals"]["n_llm_calls"], want["total_calls"])
    expect(problems, "total cost", round(summary["totals"]["cost_usd"], 6), 28 * COST_PER_CALL)
    expect(problems, "teaching cost", round(t["cost_usd"], 6), 4 * COST_PER_CALL)
    expect(problems, "aborted", summary["aborted"], None)
    expect(problems, "run.json status", read_run_status(run_dir), "completed")
    predictions = read_lines(run_dir / "predictions.jsonl")
    expect(problems, "gold sql written", any("gold_sql" in r for r in predictions), False)
    return problems


def verify_followups(summary: dict[str, Any], run_dir: Path) -> list[str]:
    problems: list[str] = []
    want = EXPECTED_FOLLOWUPS
    for name, s in summary["systems"].items():
        expect(problems, f"followups {name} n", s["n"], 3)
        expect(problems, f"followups {name} correct", s["correct"], want["correct"][name])
        calls = round(s["llm_calls_mean_per_question"] * s["n"])
        expect(problems, f"followups {name} llm calls", calls, want["llm_calls"][name])
    expect(problems, "total llm calls", summary["totals"]["n_llm_calls"], want["total_calls"])
    diff = summary["comparison"]["with_history_minus_turn2_alone"]["diff"]
    expect(problems, "with_history - turn2_alone", round(diff, 6), round(1 / 3, 6))
    expect(problems, "aborted", summary["aborted"], None)
    expect(problems, "run.json status", read_run_status(run_dir), "completed")
    return problems


def read_run_status(run_dir: Path) -> str:
    return str(json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["status"])


def run_all(out: Path) -> list[str]:
    """Run both evaluations on the fixture under ``out``; return the problems found."""
    with tempfile.TemporaryDirectory(prefix="memory-smoke-") as tmp:
        databases_dir, questions = build_fixture(Path(tmp))
        longterm = run_longterm(
            config=CONFIG,
            questions=[q for q in questions if q.question_id <= 4],
            run_dir=out / "longterm",
            run_id="smoke-longterm",
            split="dev",
            client_factory=lambda config, tracker: ScriptedClient(tracker),
            databases_dir=databases_dir,
            skip_respond=True,
            bootstrap_resamples=300,
            launched_by="evals.memory.smoke",
        )
        followups = run_followups(
            config=CONFIG,
            pairs=[
                (f, next(q for q in questions if q.question_id == f.source_question_id))
                for f in FOLLOWUPS
            ],
            run_dir=out / "followups",
            run_id="smoke-followups",
            client_factory=lambda config, tracker: ScriptedClient(tracker),
            databases_dir=databases_dir,
            skip_respond=True,
            bootstrap_resamples=300,
            launched_by="evals.memory.smoke",
        )
    return verify_longterm(longterm, out / "longterm") + verify_followups(
        followups, out / "followups"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, help="output directory (default: a temporary one)")
    args = parser.parse_args(argv)
    if args.out:
        problems = run_all(args.out.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="memory-smoke-out-") as tmp:
            problems = run_all(Path(tmp))
    if problems:
        print("SMOKE FAILED:\n  " + "\n  ".join(problems), file=sys.stderr)
        raise SystemExit(1)
    print(
        "SMOKE PASSED: long-term EX 0/4 -> 2/4 -> 3/4 (gap closed 66.7%), "
        "follow-ups 1/3 -> 2/3 -> 3/3"
    )


if __name__ == "__main__":
    main()
