"""Item-set builder: layout, gold stays out of the public file, budget abort, provenance."""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from evals.judge import build_set
from evals.judge.build_set import assign_ids
from evals.text2sql import smoke
from evals.text2sql.data import Question
from src.llm import CallUsage, Completion, LLMConfig, Pricing, UsageTracker

CONFIG = LLMConfig(provider="ollama", model="scripted", pricing=Pricing())
COST_PER_CALL = 0.001


class AnsweringClient:
    """The text-to-SQL smoke script plus a respond step that states the value for the id."""

    def __init__(self, tracker: UsageTracker, fail_respond_for: int | None = None) -> None:
        self.tracker = tracker
        self.fail_respond_for = fail_respond_for
        self.calls: dict[tuple[int, str], int] = {}

    def complete(self, system: Any, user: str, *, tag: str, prefix: Any = None) -> Completion:
        self.tracker.check()
        match = re.search(r"\[Q(\d+)\]", (prefix or "") + user)
        assert match is not None, f"no question marker in the {tag} prompt"
        qid = int(match.group(1))
        if tag == "respond":
            if qid == self.fail_respond_for:
                raise RuntimeError("boom")
            text = f"The value is {qid * 10}."
        else:
            answers = smoke.SCRIPT[qid][tag]
            n = self.calls.get((qid, tag), 0)
            self.calls[qid, tag] = n + 1
            text = answers[min(n, len(answers) - 1)]
        usage = CallUsage(tag=tag, input_tokens=100, output_tokens=20, cost_usd=COST_PER_CALL)
        usage.stop_reason = "end_turn"
        self.tracker.record(usage)
        return Completion(text=text, usage=usage)


@pytest.fixture
def fixture(tmp_path: Path) -> tuple[Path, list[Question]]:
    return smoke.build_fixture(tmp_path / "fixture")


def build(
    fixture: tuple[Path, list[Question]],
    run_dir: Path,
    fail_respond_for: int | None = None,
    **kw: Any,
) -> dict[str, Any]:
    databases_dir, questions = fixture
    return build_set.build_set(
        config=CONFIG,
        questions=questions,
        run_dir=run_dir,
        run_id="t",
        databases_dir=databases_dir,
        client_factory=lambda c, t: AnsweringClient(t, fail_respond_for),
        launched_by="test",
        **kw,
    )


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def read_run(run_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((run_dir / "run.json").read_text())
    return data


def test_ids_are_a_seeded_shuffle(fixture: tuple[Path, list[Question]]) -> None:
    _, questions = fixture
    ids = assign_ids(questions, 0)
    assert [i for i, _ in ids] == ["j-01", "j-02", "j-03", "j-04"]
    assert ids == assign_ids(list(reversed(questions)), 0)
    assert {q.question_id for _, q in ids} == {1, 2, 3, 4}


def test_run_layout_and_content(fixture: tuple[Path, list[Question]], tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    summary = build(fixture, run_dir)
    assert {p.name for p in run_dir.iterdir()} >= {
        "items.jsonl",
        "items_public.jsonl",
        "run.json",
        "llm_calls.jsonl",
        "summary.json",
    }
    items = read(run_dir / "items.jsonl")
    assert [i["item_id"] for i in items] == ["j-01", "j-02", "j-03", "j-04"]
    by_q = {i["question_id"]: i for i in items}
    # question 1 is right first time; 4 is wrong and the reviewer does not notice
    assert by_q[1]["ex_correct"] is True and by_q[4]["ex_correct"] is False
    assert by_q[1]["answer"] == "The value is 10."
    assert by_q[1]["result_preview"] == {"columns": ["v"], "rows": [["10"]], "row_count": "1"}
    assert by_q[1]["gold_preview"] == {"columns": ["v"], "rows": [["10"]], "row_count": "1"}
    assert by_q[4]["result_preview"] == {"columns": ["v"], "rows": [], "row_count": "0"}
    assert summary["n_items"] == 4 and summary["n_ex_correct"] == 3
    assert summary["totals"]["n_llm_calls"] == sum(i["n_llm_calls"] for i in items)
    calls = read(run_dir / "llm_calls.jsonl")
    assert {c["tag"] for c in calls} == {"generate", "check", "respond"}
    assert all(c["run_id"] == "t" and c["system"] == "judge_set" for c in calls)
    run = read_run(run_dir)
    assert run["status"] == "completed" and run["launched_by"] == "test"


def test_public_file_has_no_gold_and_no_ex(
    fixture: tuple[Path, list[Question]], tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    build(fixture, run_dir)
    public = read(run_dir / "items_public.jsonl")
    assert len(public) == 4
    assert all(set(p) == set(build_set.PUBLIC_FIELDS) for p in public)
    text = (run_dir / "items_public.jsonl").read_text()
    assert "gold" not in text and "ex_correct" not in text
    assert "WHERE id = 999" in text  # the agent's own (wrong) SQL is public


def test_gold_sql_text_is_never_written(
    fixture: tuple[Path, list[Question]], tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    build(fixture, run_dir)
    full = read(run_dir / "items.jsonl")
    assert all("gold_preview" in i and "ex_correct" in i for i in full)
    # the agent's SQL equals the gold SQL for question 1, so look in every other field
    for item in full:
        rest = json.dumps({k: v for k, v in item.items() if k != "final_sql"})
        assert "SELECT" not in rest


def test_new_run_dir_is_required(fixture: tuple[Path, list[Question]], tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    build(fixture, run_dir)
    with pytest.raises(FileExistsError):
        build(fixture, run_dir)


def test_budget_abort_saves_partial_items(
    fixture: tuple[Path, list[Question]], tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    summary = build(fixture, run_dir, budget_usd=COST_PER_CALL * 4.5)
    assert summary["aborted"] == "budget"
    assert 0 < summary["n_items"] < 4
    assert len(read(run_dir / "items_public.jsonl")) == summary["n_items"]
    assert read_run(run_dir)["status"] == "aborted:budget"


def test_agent_failure_is_recorded_not_fatal(
    fixture: tuple[Path, list[Question]], tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    summary = build(fixture, run_dir, fail_respond_for=2)
    assert summary["n_failed"] == 1 and summary["n_items"] == 4
    failed = [i for i in read(run_dir / "items.jsonl") if i["failure_type"]]
    assert failed[0]["question_id"] == 2 and failed[0]["answer"] == ""
