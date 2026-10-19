"""Tests for the text-to-SQL runner (temp sqlite databases, fake LLM client)."""

import json
import os
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

    def complete(
        self, system: str | None, user: str, *, tag: str, prefix: str | None = None
    ) -> Completion:
        self.tracker.check()
        user = (prefix or "") + user
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


class TestProvenance:
    def run_json(self, tmp_path: Path) -> dict[str, Any]:
        data: dict[str, Any] = json.loads((tmp_path / "run" / "run.json").read_text())
        return data

    def test_run_json_is_written_before_the_first_call_and_completed(
        self, tmp_path: Path, databases: Path, questions: list[Question]
    ) -> None:
        seen: list[dict[str, Any]] = []

        class Spy(FakeLLM):
            def complete(self, system, user, *, tag, prefix=None):  # type: ignore[no-untyped-def]
                seen.append(json.loads((tmp_path / "run" / "run.json").read_text()))
                return super().complete(system, user, tag=tag, prefix=prefix)

        def factory(config: LLMConfig, tracker: UsageTracker) -> Spy:
            return Spy(tracker)

        run(
            tmp_path,
            databases,
            questions,
            client_factory=factory,
            argv=["run.py", "--x"],
            launched_by="me",
            llm_config_name="configs/x.yaml",
            budget_usd=5.0,
        )
        first = seen[0]
        assert first["status"] == "running" and first["finished_at"] is None
        assert first["argv"] == ["run.py", "--x"] and first["launched_by"] == "me"
        assert first["llm_config"] == {
            "path": "configs/x.yaml",
            "provider": "anthropic",
            "model": "fake",
            "budget_usd": 5.0,
        }
        assert first["run_id"] == "test-run" and first["resumes"] == []
        final = self.run_json(tmp_path)
        assert final["status"] == "completed" and final["finished_at"]

    def test_existing_run_dir_is_refused_without_resume(
        self, tmp_path: Path, databases: Path, questions: list[Question]
    ) -> None:
        (tmp_path / "run").mkdir()
        (tmp_path / "run" / "something").write_text("x")
        with pytest.raises(FileExistsError):
            run(tmp_path, databases, questions)
        assert not (tmp_path / "run" / "run.json").exists()

    def test_resume_appends_an_entry(
        self, tmp_path: Path, databases: Path, questions: list[Question]
    ) -> None:
        run(tmp_path, databases, questions, budget_usd=0.003, argv=["first"])
        assert self.run_json(tmp_path)["status"] == "aborted:budget"
        started = self.run_json(tmp_path)["started_at"]
        run(tmp_path, databases, questions, resume=True, argv=["second", "--resume"])
        data = self.run_json(tmp_path)
        assert data["argv"] == ["first"] and data["started_at"] == started
        assert len(data["resumes"]) == 1
        assert data["resumes"][0]["argv"] == ["second", "--resume"]
        assert data["resumes"][0]["pid"] == os.getpid() and data["resumes"][0]["resumed_at"]
        assert data["status"] == "completed"

    def test_keyboard_interrupt_is_recorded(
        self, tmp_path: Path, databases: Path, questions: list[Question]
    ) -> None:
        class Boom(FakeLLM):
            def complete(self, system, user, *, tag, prefix=None):  # type: ignore[no-untyped-def]
                raise KeyboardInterrupt

        def factory(config: LLMConfig, tracker: UsageTracker) -> Boom:
            return Boom(tracker)

        summary = run(tmp_path, databases, questions, client_factory=factory)
        assert summary["aborted"] == "interrupted"
        assert self.run_json(tmp_path)["status"] == "interrupted"

    def test_launched_by_falls_back_to_the_environment(
        self,
        tmp_path: Path,
        databases: Path,
        questions: list[Question],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("EVAL_LAUNCHED_BY", "github-actions:wf#1")
        run(tmp_path, databases, questions)
        assert self.run_json(tmp_path)["launched_by"] == "github-actions:wf#1"


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
        assert set(PREDICTORS) == {
            "baseline",
            "baseline_ctx",
            "agent",
            "agent_v1",
            "agent_v2_nocheck",
            "agent_v2",
        }

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


def test_summarize_system_cache_totals_and_hit_ratio() -> None:
    from evals.text2sql.run import summarize_system

    def rec(i: int, cr: int, cw: int) -> dict[str, Any]:
        return {
            "correct": True, "difficulty": "simple", "failure_type": None,
            "input_tokens": i, "output_tokens": 1, "cache_read_tokens": cr,
            "cache_write_tokens": cw, "cost_usd": 0.0, "n_llm_calls": 1, "latency_ms": 1.0,
        }  # fmt: skip

    tokens = summarize_system([rec(10, 0, 90), rec(10, 90, 0)], 2, None)["tokens"]
    assert tokens["cache_read_total"] == 90
    assert tokens["cache_write_total"] == 90
    assert tokens["cache_hit_ratio"] == pytest.approx(90 / 200)
    assert summarize_system([], 0, None)["tokens"]["cache_hit_ratio"] is None


def test_counting_client_forwards_prefix() -> None:
    from evals.text2sql.baseline import CountingClient

    class Spy:
        def complete(
            self, system: str | None, user: str, *, tag: str, prefix: str | None = None
        ) -> Completion:
            self.seen = (system, user, tag, prefix)
            return Completion(text="", usage=CallUsage(tag=tag))

    spy = Spy()
    counting = CountingClient(spy)
    counting.complete("s", "u", tag="t", prefix="p")
    assert spy.seen == ("s", "u", "t", "p")
    assert counting.n_calls == 1


class TestSkipRespond:
    def test_default_runs_respond_and_records_flag(
        self, tmp_path: Path, databases: Path, questions: list[Question]
    ) -> None:
        summary = run(tmp_path, databases, questions)
        assert summary["skip_respond"] is False
        records = read_predictions(tmp_path / "run" / "predictions.jsonl")
        assert records[3, "agent"]["n_llm_calls"] == 2

    def test_skip_respond_drops_the_respond_call_for_the_agent_only(
        self, tmp_path: Path, databases: Path, questions: list[Question]
    ) -> None:
        holder: list[FakeLLM] = []
        summary = run(tmp_path, databases, questions, holder=holder, skip_respond=True)
        assert summary["skip_respond"] is True
        records = read_predictions(tmp_path / "run" / "predictions.jsonl")
        assert records[3, "agent"]["n_llm_calls"] == 1
        assert records[3, "baseline"]["n_llm_calls"] == 1
        assert summary["systems"]["agent"]["ex"] == 0.5
        assert holder[0].n_calls == 12  # 6 baseline + 6 agent reason, no respond

    def test_flag_parsed(self) -> None:
        assert parse_args(["--llm-config", "x.yaml"]).skip_respond is False
        assert parse_args(["--llm-config", "x.yaml", "--skip-respond"]).skip_respond is True


class TestBootstrapInSummary:
    def test_cis_comparison_and_meta(
        self, tmp_path: Path, databases: Path, questions: list[Question]
    ) -> None:
        summary = run(tmp_path, databases, questions)
        for system in ("baseline", "agent"):
            s = summary["systems"][system]
            low, high = s["ex_ci95"]
            assert low <= s["ex"] <= high
            assert all("ex_ci95" in d for d in s["by_difficulty"].values())
        paired = summary["comparison"]["agent_minus_baseline"]
        assert paired["n"] == 6 and paired["diff"] == 0.0
        assert summary["bootstrap"]["B"] == 10000 and summary["bootstrap"]["seed"] == 0
        on_disk = json.loads((tmp_path / "run" / "summary.json").read_text())
        assert on_disk["bootstrap"] == summary["bootstrap"]

    def test_no_comparison_for_one_system(
        self, tmp_path: Path, databases: Path, questions: list[Question]
    ) -> None:
        summary = run(tmp_path, databases, questions, systems=["baseline"])
        assert summary["comparison"] is None
