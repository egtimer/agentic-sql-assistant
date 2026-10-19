"""Tests for the text-to-SQL runner (temp sqlite databases, fake LLM client)."""

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from evals.text2sql.data import Question
from evals.text2sql.run import (
    PREDICTORS,
    classify,
    execution_order,
    format_table,
    load_llm_config,
    parse_args,
    read_predictions,
    run_experiment,
    select_questions,
)
from src.llm import CallUsage, Completion, LLMConfig, Pricing, UsageTracker, cost_of

PRICING = Pricing(input_per_mtok=1.0, output_per_mtok=2.0)
CONFIG = LLMConfig(provider="anthropic", model="fake", max_tokens=100, pricing=PRICING)
CONFIGS_DIR = Path(__file__).parents[2] / "configs" / "llm"

# question_id -> (db_id, correct SQL). Questions 1, 2 and 9 get a wrong prediction.
QUESTIONS = {5: "beta", 3: "beta", 9: "beta", 1: "alpha", 4: "alpha", 2: "alpha"}
WRONG = {1, 2, 9}


class FakeLLM:
    """Mimics the real clients: checks and records on the tracker, answers per question."""

    def __init__(
        self,
        tracker: UsageTracker,
        stop_reason: dict[int, str] | None = None,
        raise_for: set[int] | None = None,
        input_tokens: int = 1000,
    ) -> None:
        self.tracker = tracker
        self.stop_reason = stop_reason or {}
        self.raise_for = raise_for or set()
        self.input_tokens = input_tokens
        self.n_calls = 0

    def _question_id(self, user: str) -> int:
        for qid in QUESTIONS:
            if f"question number {qid}?" in user:
                return qid
        raise AssertionError(f"no question id in prompt: {user[-200:]}")

    def complete(self, system: str | None, user: str, *, tag: str) -> Completion:
        self.tracker.check()
        self.n_calls += 1
        if tag == "respond":
            text = "done"
            qid = -1
        else:
            qid = self._question_id(user)
            if qid in self.raise_for:
                raise RuntimeError("api down")
            value = 999 if qid in WRONG else qid
            sql = f"SELECT v FROM t WHERE id = {value}"
            text = (
                f"```sql\n{sql}\n```"
                if tag == "baseline"
                else json.dumps([{"tool": "sql_query", "input": sql, "reason": "r"}])
            )
        usage = CallUsage(
            tag=tag,
            input_tokens=self.input_tokens,
            output_tokens=100,
            latency_ms=5.0,
            stop_reason=self.stop_reason.get(qid, "end_turn"),
        )
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
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)")
        conn.executemany("INSERT INTO t VALUES (?, ?)", [(i, i * 10) for i in range(1, 10)])
        conn.commit()
        conn.close()
    return root


@pytest.fixture
def questions() -> list[Question]:
    difficulties = ["simple", "moderate", "challenging"]
    return [
        Question(
            question_id=qid,
            db_id=db_id,
            question=f"What is the value of question number {qid}?",
            evidence="",
            SQL=f"SELECT v FROM t WHERE id = {qid}",
            difficulty=difficulties[qid % 3],  # type: ignore[arg-type]
        )
        for qid, db_id in QUESTIONS.items()
    ]


def run(
    tmp_path: Path,
    databases: Path,
    questions: list[Question],
    holder: list[FakeLLM] | None = None,
    llm_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    def factory(config: LLMConfig, tracker: UsageTracker) -> FakeLLM:
        llm = FakeLLM(tracker, **(llm_kwargs or {}))
        if holder is not None:
            holder.append(llm)
        return llm

    params: dict[str, Any] = {
        "config": CONFIG,
        "systems": ["baseline", "agent"],
        "questions": questions,
        "run_dir": tmp_path / "run",
        "run_id": "test-run",
        "split": "dev",
        "client_factory": factory,
        "databases_dir": databases,
    }
    params.update(kwargs)
    return run_experiment(**params)


class TestEndToEnd:
    def test_both_systems(self, tmp_path: Path, databases: Path, questions: list[Question]) -> None:
        summary = run(tmp_path, databases, questions)
        run_dir = tmp_path / "run"

        lines = [
            json.loads(line)
            for line in (run_dir / "predictions.jsonl").read_text().split("\n")
            if line
        ]
        assert len(lines) == 12
        # systems run in separate passes, each ordered by (db_id, question_id)
        assert [r["system"] for r in lines] == ["baseline"] * 6 + ["agent"] * 6
        expected_order = [1, 2, 4, 3, 5, 9]
        assert [r["question_id"] for r in lines[:6]] == expected_order
        assert [r["question_id"] for r in lines[6:]] == expected_order

        first = lines[0]
        for key in (
            "question_id", "db_id", "difficulty", "system", "pred_sql", "gold_sql", "correct",
            "error", "stop_reasons", "n_llm_calls", "input_tokens", "output_tokens",
            "cache_read_tokens", "cache_write_tokens", "cost_usd", "latency_ms",
        ):  # fmt: skip
            assert key in first
        by_key = {(r["system"], r["question_id"]): r for r in lines}
        assert by_key["baseline", 3]["correct"] is True
        assert by_key["baseline", 1]["correct"] is False
        assert by_key["baseline", 1]["failure_type"] == "wrong_result"
        assert by_key["baseline", 3]["n_llm_calls"] == 1
        assert by_key["agent", 3]["n_llm_calls"] == 2
        assert by_key["agent", 3]["input_tokens"] == 2000
        assert by_key["agent", 3]["cost_usd"] == pytest.approx(2 * (1000 * 1.0 + 100 * 2.0) / 1e6)

        for system in ("baseline", "agent"):
            s = summary["systems"][system]
            assert s["n"] == 6
            assert s["correct"] == 3
            assert s["ex"] == 0.5
            assert sum(d["n"] for d in s["by_difficulty"].values()) == 6
            assert s["failures"] == {"wrong_result": 3}
            assert s["aborted"] is None
        assert summary["systems"]["baseline"]["tokens"]["input_mean_per_question"] == 1000
        assert summary["systems"]["agent"]["tokens"]["input_mean_per_question"] == 2000
        assert summary["aborted"] is None
        assert summary["split"] == "dev"
        assert summary["llm_config"]["model"] == "fake"
        assert {"commit", "dirty"} <= set(summary["git"])
        assert summary["started_at"] <= summary["finished_at"]
        assert json.loads((run_dir / "summary.json").read_text())["run_id"] == "test-run"

        calls = [
            json.loads(line)
            for line in (run_dir / "llm_calls.jsonl").read_text().split("\n")
            if line
        ]
        assert len(calls) == 6 + 12
        assert {c["run_id"] for c in calls} == {"test-run"}
        assert {c["system"] for c in calls} == {"baseline", "agent"}
        assert {c["tag"] for c in calls} == {"baseline", "reason", "respond"}
        assert all("question_id" in c for c in calls)
        assert "EX" in format_table(summary)

    def test_workers(self, tmp_path: Path, databases: Path, questions: list[Question]) -> None:
        summary = run(tmp_path, databases, questions, workers=3)
        assert summary["systems"]["baseline"]["n"] == 6
        assert summary["systems"]["agent"]["correct"] == 3
        records = read_predictions(tmp_path / "run" / "predictions.jsonl")
        assert len(records) == 12

    def test_existing_results_need_resume(
        self, tmp_path: Path, databases: Path, questions: list[Question]
    ) -> None:
        run(tmp_path, databases, questions, systems=["baseline"])
        with pytest.raises(FileExistsError):
            run(tmp_path, databases, questions, systems=["baseline"])


class TestBudget:
    def test_abort_saves_partial_results(
        self, tmp_path: Path, databases: Path, questions: list[Question]
    ) -> None:
        # each call costs 0.0012 USD; the cap is reached after the 3rd baseline call
        summary = run(tmp_path, databases, questions, budget_usd=0.003)
        run_dir = tmp_path / "run"
        records = read_predictions(run_dir / "predictions.jsonl")
        assert len(records) == 3
        assert {s for (_, s) in records} == {"baseline"}
        assert summary["aborted"] == "budget"
        assert summary["systems"]["baseline"]["aborted"] == "budget"
        assert summary["systems"]["baseline"]["n"] == 3
        assert summary["systems"]["agent"]["n"] == 0
        assert summary["systems"]["agent"]["aborted"] == "budget"
        assert (run_dir / "llm_calls.jsonl").read_text().count("\n") == 3
        assert json.loads((run_dir / "summary.json").read_text())["aborted"] == "budget"

    def test_overshoot_is_at_most_one_call(
        self, tmp_path: Path, databases: Path, questions: list[Question]
    ) -> None:
        summary = run(tmp_path, databases, questions, budget_usd=0.0001)
        assert summary["tracker"]["cost_usd_this_invocation"] <= 0.0001 + 0.0012


class TestResume:
    def test_skips_done_pairs(
        self, tmp_path: Path, databases: Path, questions: list[Question]
    ) -> None:
        first: list[FakeLLM] = []
        run(tmp_path, databases, questions[:3], first)
        assert first[0].n_calls == 3 + 6
        second: list[FakeLLM] = []
        summary = run(tmp_path, databases, questions, second, resume=True)
        # 3 pairs per system were done; only the other 3 questions per system run
        assert second[0].n_calls == 3 + 6
        records = read_predictions(tmp_path / "run" / "predictions.jsonl")
        assert len(records) == 12
        lines = (tmp_path / "run" / "predictions.jsonl").read_text().count("\n")
        assert lines == 12
        assert summary["systems"]["baseline"]["n"] == 6
        assert summary["resumed"] is True

    def test_resume_after_budget_abort_completes(
        self, tmp_path: Path, databases: Path, questions: list[Question]
    ) -> None:
        run(tmp_path, databases, questions, budget_usd=0.003)
        summary = run(tmp_path, databases, questions, resume=True)
        assert summary["aborted"] is None
        assert summary["systems"]["baseline"]["n"] == 6
        assert summary["systems"]["agent"]["n"] == 6
        assert len(read_predictions(tmp_path / "run" / "predictions.jsonl")) == 12


class TestFailures:
    @pytest.mark.parametrize("reason", ["refusal", "max_tokens"])
    def test_failure_stop_reason_marks_question_failed(
        self, tmp_path: Path, databases: Path, questions: list[Question], reason: str
    ) -> None:
        # question 3 would be answered correctly, but the call ended abnormally
        summary = run(
            tmp_path, databases, questions, systems=["baseline"],
            llm_kwargs={"stop_reason": {3: reason}},
        )  # fmt: skip
        record = read_predictions(tmp_path / "run" / "predictions.jsonl")[(3, "baseline")]
        assert record["correct"] is False
        assert record["failure_type"] == reason
        assert record["stop_reasons"] == [reason]
        assert summary["systems"]["baseline"]["failures"][reason] == 1
        assert summary["systems"]["baseline"]["n"] == 6

    def test_exception_is_recorded_and_run_continues(
        self, tmp_path: Path, databases: Path, questions: list[Question]
    ) -> None:
        summary = run(
            tmp_path, databases, questions, systems=["baseline"], llm_kwargs={"raise_for": {3}}
        )
        record = read_predictions(tmp_path / "run" / "predictions.jsonl")[(3, "baseline")]
        assert record["failure_type"] == "exception"
        assert "api down" in record["error"]
        assert summary["systems"]["baseline"]["n"] == 6
        assert summary["aborted"] is None

    def test_repeated_exceptions_abort(
        self, tmp_path: Path, databases: Path, questions: list[Question]
    ) -> None:
        summary = run(
            tmp_path, databases, questions, systems=["baseline"],
            llm_kwargs={"raise_for": set(QUESTIONS)},
        )  # fmt: skip
        assert summary["aborted"] == "errors"
        assert summary["systems"]["baseline"]["n"] == 3

    def test_exceptions_are_retried_on_resume(
        self, tmp_path: Path, databases: Path, questions: list[Question]
    ) -> None:
        run(tmp_path, databases, questions, systems=["baseline"], llm_kwargs={"raise_for": {3}})
        summary = run(tmp_path, databases, questions, systems=["baseline"], resume=True)
        record = read_predictions(tmp_path / "run" / "predictions.jsonl")[(3, "baseline")]
        assert record["failure_type"] is None
        assert summary["systems"]["baseline"]["failures"] == {"wrong_result": 3}


class TestHelpers:
    def test_classify(self) -> None:
        assert classify(True, None, ["end_turn"]) is None
        assert classify(True, None, ["refusal"]) == "refusal"
        assert classify(False, "no sql", []) == "no_sql"
        assert classify(False, "timeout", []) == "timeout"
        assert classify(False, "no such column", []) == "sql_error"
        assert classify(False, None, []) == "wrong_result"
        assert classify(False, "x", ["max_tokens"]) == "max_tokens"

    def test_select_questions(self, questions: list[Question]) -> None:
        split = [1, 2, 3, 4, 5]
        assert [q.question_id for q in select_questions(questions, split)] == [1, 2, 3, 4, 5]
        limited = select_questions(questions, split, limit=2)
        assert len(limited) == 2
        assert limited == select_questions(questions, split, limit=2)
        assert [
            q.question_id for q in select_questions(questions, split, limit=2, question_ids=[4])
        ] == [4]
        assert [q.question_id for q in select_questions(questions, split, question_ids=[4, 2])] == [
            2,
            4,
        ]
        with pytest.raises(ValueError):
            select_questions(questions, split, question_ids=[9])

    def test_execution_order(self, questions: list[Question]) -> None:
        assert [q.question_id for q in execution_order(questions)] == [1, 2, 4, 3, 5, 9]

    def test_predictors_registered(self) -> None:
        assert set(PREDICTORS) == {"baseline", "agent"}

    def test_parse_args_defaults(self) -> None:
        args = parse_args(["--llm-config", "x.yaml"])
        assert args.system == "both"
        assert args.split == "dev"
        assert args.workers == 1
        assert args.resume is False
        assert args.budget_usd is None

    def test_committed_configs_load(self) -> None:
        haiku = load_llm_config(CONFIGS_DIR / "claude-haiku-5-5.yaml")
        assert haiku.provider == "anthropic"
        assert haiku.model == "claude-haiku-5-5"
        assert haiku.max_tokens == 16000
        assert haiku.effort == "low"
        assert haiku.temperature is None
        p = haiku.pricing
        assert (p.input_per_mtok, p.output_per_mtok) == (0.10, 0.50)
        assert (p.cache_read_per_mtok, p.cache_write_per_mtok) == (0.01, 0.125)
        assert p.long_context_threshold == 100000
        assert (p.long_input_per_mtok, p.long_output_per_mtok) == (0.50, 2.50)

        qwen = load_llm_config(CONFIGS_DIR / "ollama-qwen2.5-3b.yaml")
        assert qwen.provider == "ollama"
        assert qwen.model == "qwen2.5:3b"
        assert (qwen.temperature, qwen.num_ctx, qwen.max_tokens) == (0, 8192, 1024)
        assert qwen.base_url == "http://localhost:11434"
        assert qwen.pricing == Pricing()


def test_sampling_meta(questions: list[Question]) -> None:
    from evals.text2sql.run import sampling_meta

    meta = sampling_meta(questions, 3, None, 7)
    assert meta["method"] == "stratified"
    assert meta["seed"] == 7
    assert meta["n_sampled"] == len(questions)
    assert sum(meta["per_stratum"].values()) == len(questions)
    assert sampling_meta(questions, 3, [1], 7)["method"] == "question_ids"
    assert sampling_meta(questions, None, None, 7)["seed"] is None


def test_parse_args_seed_default() -> None:
    assert parse_args(["--llm-config", "x.yaml"]).seed == 0
    assert parse_args(["--llm-config", "x.yaml", "--seed", "3"]).seed == 3
