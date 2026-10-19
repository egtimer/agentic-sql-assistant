"""Tests for the multi-system runner: v2 systems, flips, pairwise comparisons."""

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from evals.text2sql.data import Question
from evals.text2sql.run import (
    compare_systems,
    format_table,
    parse_args,
    parse_systems,
    read_predictions,
    run_experiment,
    summarize_v2,
    system_options,
)
from src.llm import CallUsage, Completion, LLMConfig, Pricing, UsageTracker, cost_of

PRICING = Pricing(input_per_mtok=1.0, output_per_mtok=2.0)
CONFIG = LLMConfig(provider="anthropic", model="fake", max_tokens=100, pricing=PRICING)
# questions 1 and 2: the first generated SQL is wrong (empty result), the check fixes it.
# question 3: first SQL is right, the check "fixes" it into a wrong one (right -> wrong).
# questions 4 and 5: right and left alone.
QUESTIONS = {1: "alpha", 2: "alpha", 3: "alpha", 4: "beta", 5: "beta"}
FIRST_WRONG = {1, 2}
SPOILED = {3}


class FakeV2LLM:
    def __init__(self, tracker: UsageTracker) -> None:
        self.tracker = tracker
        self.calls: list[tuple[str, str | None]] = []

    def complete(
        self, system: str | None, user: str, *, tag: str, prefix: str | None = None
    ) -> Completion:
        self.tracker.check()
        self.calls.append((tag, prefix))
        full = (prefix or "") + user
        match = re.search(r"question number (\d+)\?", full)
        qid = int(match.group(1)) if match else -1
        if tag in ("baseline", "baseline_ctx"):
            text = f"```sql\nSELECT v FROM t WHERE id = {qid}\n```"
        elif tag == "generate":
            value = 999 if qid in FIRST_WRONG else qid
            text = f"```sql\nSELECT v FROM t WHERE id = {value}\n```"
        elif tag == "check":
            if "The result is empty." in user:
                text = f"Wrong id.\n```sql\nSELECT v FROM t WHERE id = {qid}\n```"
            elif qid in SPOILED and "WHERE id = 3\n" in user:
                text = "```sql\nSELECT v FROM t WHERE id = 4\n```"
            else:
                text = "OK"
        else:
            text = "done"
        usage = CallUsage(tag=tag, input_tokens=1000, output_tokens=100, latency_ms=1.0)
        usage.cost_usd = cost_of(usage, PRICING)
        self.tracker.record(usage)
        return Completion(text=text, usage=usage)


@pytest.fixture
def databases(tmp_path: Path) -> Path:
    root = tmp_path / "dbs"
    for db_id in ("alpha", "beta"):
        path = root / db_id / f"{db_id}.sqlite"
        path.parent.mkdir(parents=True)
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, label TEXT)")
        conn.executemany("INSERT INTO t VALUES (?, ?, 'x')", [(i, i * 10) for i in range(1, 10)])
        if db_id == "beta":
            conn.execute("CREATE TABLE extra (z INTEGER)")
        conn.commit()
        conn.close()
    return root


@pytest.fixture
def questions() -> list[Question]:
    return [
        Question(
            question_id=qid,
            db_id=db_id,
            question=f"What is the value of question number {qid}?",
            evidence="",
            SQL=f"SELECT v FROM t WHERE id = {qid}",
            difficulty="simple",
        )
        for qid, db_id in QUESTIONS.items()
    ]


def run(
    tmp_path: Path, databases: Path, questions: list[Question], **kwargs: Any
) -> tuple[dict[str, Any], list[FakeV2LLM]]:
    holder: list[FakeV2LLM] = []

    def factory(config: LLMConfig, tracker: UsageTracker) -> FakeV2LLM:
        llm = FakeV2LLM(tracker)
        holder.append(llm)
        return llm

    params: dict[str, Any] = {
        "config": CONFIG,
        "systems": ["baseline", "baseline_ctx", "agent_v2_nocheck", "agent_v2"],
        "questions": questions,
        "run_dir": tmp_path / "run",
        "run_id": "r",
        "split": "dev",
        "client_factory": factory,
        "databases_dir": databases,
        "skip_respond": True,
    }
    params.update(kwargs)
    return run_experiment(**params), holder


class TestSystemArgs:
    def test_parse_systems(self) -> None:
        assert parse_systems("baseline,agent_v2") == ["baseline", "agent_v2"]
        assert parse_systems("agent_v2, baseline") == ["agent_v2", "baseline"]
        assert parse_systems("both") == ["baseline", "agent_v1"]
        assert parse_systems("agent") == ["agent_v1"]
        assert parse_systems("baseline,both") == ["baseline", "agent_v1"]
        with pytest.raises(ValueError, match="unknown system"):
            parse_systems("baseline,nope")

    def test_cli(self) -> None:
        args = parse_args(["--llm-config", "x.yaml", "--max-fixes", "3"])
        assert args.max_fixes == 3 and args.system == "both"
        assert parse_args(["--llm-config", "x.yaml"]).max_fixes == 2

    def test_system_options(self) -> None:
        assert system_options("baseline", skip_respond=True, max_fixes=2) == {}
        assert system_options("baseline_ctx", skip_respond=True, max_fixes=2) == {}
        assert system_options("agent_v1", skip_respond=True, max_fixes=2) == {"skip_respond": True}
        assert system_options("agent_v1", skip_respond=False, max_fixes=2) == {}
        assert system_options("agent_v2", skip_respond=True, max_fixes=1) == {
            "skip_respond": True,
            "max_fixes": 1,
        }


class TestEndToEnd:
    def test_all_systems(self, tmp_path: Path, databases: Path, questions: list[Question]) -> None:
        summary, _ = run(tmp_path, databases, questions)
        s = summary["systems"]
        assert s["baseline"]["correct"] == 5 and s["baseline_ctx"]["correct"] == 5
        assert s["agent_v2_nocheck"]["correct"] == 3  # errors only are rechecked: none here
        assert s["agent_v2"]["correct"] == 4
        assert summary["max_fixes"] == 2
        assert summary["systems_order"] == [
            "baseline",
            "baseline_ctx",
            "agent_v2_nocheck",
            "agent_v2",
        ]
        assert set(summary["comparison"]) == {
            "baseline_ctx_minus_baseline",
            "agent_v2_nocheck_minus_baseline",
            "agent_v2_minus_baseline",
            "agent_v2_nocheck_minus_baseline_ctx",
            "agent_v2_minus_baseline_ctx",
            "agent_v2_minus_agent_v2_nocheck",
        }
        assert summary["comparison"]["agent_v2_minus_baseline"]["diff"] == pytest.approx(-0.2)
        assert summary["comparison"]["agent_v2_minus_agent_v2_nocheck"]["diff"] == pytest.approx(
            0.2
        )

    def test_v2_records_and_flips(
        self, tmp_path: Path, databases: Path, questions: list[Question]
    ) -> None:
        summary, _ = run(tmp_path, databases, questions)
        records = read_predictions(tmp_path / "run" / "predictions.jsonl")
        r = records[1, "agent_v2"]
        assert r["first_sql_correct"] is False and r["correct"] is True
        assert r["fixes_used"] == 1 and r["n_llm_calls"] == 3
        assert [a["row_count"] for a in r["attempts"]] == ["0", "1"]
        assert r["attempts"][0]["n_columns"] == 1 and r["attempts"][0]["error"] is None
        assert [v["kind"] for v in r["verdicts"]] == ["fix", "ok"]
        assert all(isinstance(v["text"], str) and v["text"] for v in r["verdicts"])  # full text
        v2 = summary["systems"]["agent_v2"]["v2"]
        assert v2["flips"] == {
            "wrong_to_right": 2,
            "right_to_wrong": 1,
            "wrong_to_wrong": 0,
            "right_to_right": 0,
        }
        assert v2["n_final_differs_from_first"] == 3
        assert v2["frac_with_fix"] == pytest.approx(3 / 5)
        assert v2["first_sql_ex"] == pytest.approx(3 / 5)
        assert v2["unparsed_checks"] == 0
        assert v2["llm_calls_mean"] == pytest.approx((3 + 3 + 3 + 2 + 2) / 5)
        nocheck = summary["systems"]["agent_v2_nocheck"]["v2"]
        assert nocheck["frac_with_fix"] == 0 and nocheck["first_sql_ex"] == pytest.approx(3 / 5)
        assert "v2" not in summary["systems"]["baseline"]
        assert "attempts" not in records[1, "baseline"]

    def test_max_fixes_zero(
        self, tmp_path: Path, databases: Path, questions: list[Question]
    ) -> None:
        summary, holder = run(tmp_path, databases, questions, systems=["agent_v2"], max_fixes=0)
        assert summary["max_fixes"] == 0
        assert summary["systems"]["agent_v2"]["llm_calls_mean_per_question"] == 1
        assert {tag for tag, _ in holder[0].calls} == {"generate"}

    def test_system_order_and_alias_still_work(
        self, tmp_path: Path, databases: Path, questions: list[Question]
    ) -> None:
        run(tmp_path, databases, questions, systems=["agent_v2", "baseline"])
        lines = (tmp_path / "run" / "predictions.jsonl").read_text().splitlines()
        assert [json.loads(x)["system"] for x in lines] == ["agent_v2"] * 5 + ["baseline"] * 5

    def test_prefix_stable_per_database_and_node(
        self, tmp_path: Path, databases: Path, questions: list[Question]
    ) -> None:
        _, holder = run(tmp_path, databases, questions, systems=["agent_v2"])
        by_key: dict[tuple[str, str], set[str | None]] = {}
        db_of = {q.question_id: q.db_id for q in questions}
        assert len(holder[0].calls) == 3 + 3 + 3 + 2 + 2
        # calls are in question order within the pass: ids 1,2,3 (alpha) then 4,5 (beta)
        per_question = [3, 3, 3, 2, 2]
        index = 0
        for qid, n in zip(sorted(db_of), per_question, strict=True):
            for tag, prefix in holder[0].calls[index : index + n]:
                by_key.setdefault((db_of[qid], tag), set()).add(prefix)
            index += n
        assert all(len(prefixes) == 1 for prefixes in by_key.values())
        assert by_key["alpha", "generate"] == by_key["alpha", "check"]
        assert by_key["alpha", "generate"] != by_key["beta", "generate"]

    def test_baseline_ctx_uses_context_with_example_values(
        self, tmp_path: Path, databases: Path, questions: list[Question]
    ) -> None:
        _, holder = run(tmp_path, databases, questions, systems=["baseline", "baseline_ctx"])
        plain = {p for t, p in holder[0].calls if t == "baseline"}
        rich = {p for t, p in holder[0].calls if t == "baseline_ctx"}
        assert all(p and "Example values" not in p for p in plain)
        assert all(p and "Example values" in p and "'x'" in p for p in rich)
        assert all(p.startswith("Database schema (SQLite):\n\n") for p in rich if p)


class TestSummaries:
    def test_summarize_v2_none_without_v2_records(self) -> None:
        assert summarize_v2([{"correct": True}]) is None

    def test_wrong_to_wrong_and_unchanged(self) -> None:
        def rec(first: str, final: str, first_ok: bool, ok: bool) -> dict[str, Any]:
            return {
                "first_sql": first,
                "pred_sql": final,
                "first_sql_correct": first_ok,
                "correct": ok,
                "fixes_used": int(first != final),
                "n_llm_calls": 2,
                "verdicts": [{"kind": "unparsed", "unparsed": True}],
            }

        v2 = summarize_v2([rec("a", "b", False, False), rec("a", "a", True, True)])
        assert v2 is not None
        assert v2["flips"]["wrong_to_wrong"] == 1 and v2["n_final_differs_from_first"] == 1
        assert v2["unparsed_checks"] == 2

    def test_compare_single_system_is_none(self) -> None:
        assert compare_systems({"baseline": [{"question_id": 1, "correct": True}]}) is None

    def test_format_table_shows_pairs_and_flips(
        self, tmp_path: Path, databases: Path, questions: list[Question]
    ) -> None:
        summary, _ = run(tmp_path, databases, questions)
        table = format_table(summary)
        assert "agent_v2 - baseline EX:" in table
        assert "agent_v2 - agent_v2_nocheck EX:" in table
        assert "wrong->right 2, right->wrong 1" in table
