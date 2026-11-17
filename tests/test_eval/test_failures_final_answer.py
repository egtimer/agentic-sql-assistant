"""Final-answer pass with a scripted client on a tiny SQLite database (no BIRD, no API)."""

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from evals.failures import final_answer
from evals.text2sql.data import Question
from src.llm import CallUsage, Completion, LLMConfig, Pricing, ProviderError, UsageTracker

CONFIG = LLMConfig(provider="ollama", model="scripted", pricing=Pricing())


@pytest.fixture
def databases(tmp_path: Path) -> Path:
    root = tmp_path / "dbs"
    (root / "shop").mkdir(parents=True)
    conn = sqlite3.connect(root / "shop" / "shop.sqlite")
    conn.execute("CREATE TABLE t (v INTEGER)")
    conn.executemany("INSERT INTO t VALUES (?)", [(1,), (2,), (3,)])
    conn.commit()
    conn.close()
    return root


def question(qid: int, gold: str) -> Question:
    return Question(
        question_id=qid, db_id="shop", question=f"Q{qid}?", SQL=gold, difficulty="simple"
    )


def pred(qid: int, sql: str, correct: bool) -> dict[str, Any]:
    return {"question_id": qid, "system": "agent_v2", "pred_sql": sql, "correct": correct}


class Scripted:
    """respond: echoes the first result row; judge: correct when the answer says "3"."""

    def __init__(self, tracker: UsageTracker, fail_on: str | None = None) -> None:
        self.tracker = tracker
        self.fail_on = fail_on
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: Any, user: str, *, tag: str, prefix: Any = None) -> Completion:
        self.tracker.check()
        self.calls.append((tag, user))
        if self.fail_on and self.fail_on in user:
            raise ProviderError("anthropic", RuntimeError("credit balance is too low"))
        if tag == "respond":
            text = "The answer is 3." if "3" in user.split("Query result:")[1] else "No idea."
        else:
            answer = user.split("Answer:\n")[1]
            label = "correct" if "3" in answer else "incorrect"
            text = json.dumps({"rationale": "r", "label": label})
        usage = CallUsage(tag=tag, input_tokens=10, output_tokens=5, cost_usd=0.001)
        usage.stop_reason = "end_turn"
        self.tracker.record(usage)
        return Completion(text=text, usage=usage)


def run(tmp_path: Path, databases: Path, preds: list[dict[str, Any]], **kw: Any) -> Any:
    questions = {
        1: question(1, "SELECT MAX(v) FROM t"),
        2: question(2, "SELECT MAX(v) FROM t"),
        3: question(3, "SELECT nope FROM t"),
    }
    clients: list[Scripted] = []
    fail_on = kw.pop("fail_on", None)

    def factory(c: LLMConfig, t: UsageTracker) -> Scripted:
        clients.append(Scripted(t, fail_on))
        return clients[0]

    summary = final_answer.run_final_answer(
        config=CONFIG,
        predictions=preds,
        questions=questions,
        report="r",
        run_dir=tmp_path / "out",
        run_id="t",
        databases_dir=databases,
        client_factory=factory,
        launched_by="test",
        **kw,
    )
    return summary, clients[0]


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def test_answers_judged_and_public_file_has_no_gold(tmp_path: Path, databases: Path) -> None:
    preds = [
        pred(1, "SELECT MAX(v) FROM t", True),
        pred(2, "SELECT MIN(v) FROM t", False),
        pred(3, "SELECT MAX(v) FROM t", False),
    ]
    summary, client = run(tmp_path, databases, preds)
    records = {r["question_id"]: r for r in read(tmp_path / "out" / "answers.jsonl")}
    assert records[1]["answer"] == "The answer is 3."
    assert records[1]["judge_label"] == "correct"
    assert records[2]["judge_label"] == "incorrect"
    assert records[3]["judge_label"] == "no_reference"  # gold SQL fails: no judge call
    assert [tag for tag, _ in client.calls].count("judge") == 2
    # the respond prompt is the agent's: the question and the re-executed result
    respond_users = [u for tag, u in client.calls if tag == "respond"]
    assert respond_users[0].startswith("Question: Q1?\n\nQuery result:")
    public = read(tmp_path / "out" / "answers_public.jsonl")
    assert set(public[0]) == set(final_answer.PUBLIC_FIELDS)
    assert "gold" not in json.dumps(public)
    assert summary["ex_vs_judge"] == {
        "ex_ok/judge_correct": 1,
        "ex_wrong/judge_incorrect": 1,
        "ex_wrong/judge_no_reference": 1,
    }
    assert summary["totals"]["cost_usd"] == pytest.approx(0.005)


def test_provider_error_stops_the_run(tmp_path: Path, databases: Path) -> None:
    preds = [pred(1, "SELECT MAX(v) FROM t", True), pred(2, "SELECT MIN(v) FROM t", False)]
    summary, _ = run(tmp_path, databases, preds, fail_on="Q1?")
    assert summary["aborted"] == "provider_error"
    records = read(tmp_path / "out" / "answers.jsonl")
    assert len(records) == 1 and records[0]["failure_type"] == "exception"


def test_budget_stops_the_run(tmp_path: Path, databases: Path) -> None:
    preds = [pred(i, "SELECT MAX(v) FROM t", True) for i in (1, 2)]
    summary, _ = run(tmp_path, databases, preds, budget_usd=0.0025)
    assert summary["aborted"] == "budget"


def test_run_dir_must_be_new(tmp_path: Path, databases: Path) -> None:
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "x").write_text("")
    with pytest.raises(FileExistsError):
        run(tmp_path, databases, [pred(1, "SELECT 1", True)])
