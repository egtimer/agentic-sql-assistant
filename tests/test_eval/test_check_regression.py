"""Regression gate: pass and fail cases on synthetic predictions."""

import json
from pathlib import Path

import pytest

from evals.text2sql import check_regression
from evals.text2sql.check_regression import evaluate, load_records, main, render


def write_run(path: Path, correct: list[bool], cost: float, system: str = "agent_v2") -> Path:
    path.mkdir(parents=True)
    rows = [
        {"system": system, "question_id": i, "correct": ok, "cost_usd": cost}
        for i, ok in enumerate(correct)
    ]
    rows.append({"system": "baseline", "question_id": 0, "correct": True, "cost_usd": 9.0})
    (path / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def verdict(
    ref: list[bool], cur: list[bool], ref_cost: float = 0.01, cur_cost: float = 0.01, **kw: float
) -> dict[str, object]:
    reference = {i: {"correct": ok, "cost_usd": ref_cost} for i, ok in enumerate(ref)}
    current = {i: {"correct": ok, "cost_usd": cur_cost} for i, ok in enumerate(cur)}
    return evaluate(reference, current, resamples=2000, **kw)


class TestEvaluate:
    def test_same_results_pass(self) -> None:
        ref = [True] * 20 + [False] * 10
        r = verdict(ref, list(ref))
        assert r["passed"] and r["diff_pts"] == 0 and r["cost_ratio"] == pytest.approx(1.0)

    def test_big_significant_drop_fails(self) -> None:
        # 30 questions: 24 right in the reference, the same 24 wrong now -> -80 pts, CI < 0
        ref = [True] * 24 + [False] * 6
        r = verdict(ref, [False] * 30)
        assert r["ex_regression"] and not r["passed"]
        assert r["diff_pts"] == pytest.approx(-80.0) and r["ci_high_pts"] < 0

    def test_drop_beyond_the_limit_but_within_noise_passes(self) -> None:
        # 6 questions lost and 2 gained out of 30: -13.3 pts, but the interval reaches above 0
        ref = [True] * 6 + [False] * 2 + [True] * 11 + [False] * 11
        cur = [False] * 6 + [True] * 2 + [True] * 11 + [False] * 11
        r = verdict(ref, cur)
        assert r["diff_pts"] == pytest.approx(-100 * 4 / 30)
        assert r["ci_high_pts"] >= 0
        assert not r["ex_regression"] and r["passed"]

    def test_small_sample_loss_with_interval_below_zero_fails(self) -> None:
        # 4 questions lost, none gained: -13.3 pts and the interval stays below 0
        ref = [True] * 4 + [False] * 26
        r = verdict(ref, [False] * 30)
        assert r["diff_pts"] == pytest.approx(-100 * 4 / 30)
        assert r["ci_high_pts"] < 0 and r["ex_regression"]

    def test_significant_but_small_drop_passes(self) -> None:
        # 100 questions, 5 flips down: significant? CI upper bound is 0 or below, drop is 5 pts
        ref = [True] * 60 + [False] * 40
        cur = [False] * 5 + [True] * 55 + [False] * 40
        r = verdict(ref, cur)
        assert r["diff_pts"] == pytest.approx(-5.0)
        assert not r["ex_regression"] and r["passed"]

    def test_threshold_is_a_parameter(self) -> None:
        ref = [True] * 60 + [False] * 40
        cur = [False] * 20 + [True] * 40 + [False] * 40  # -20 pts
        assert verdict(ref, cur)["ex_regression"]
        assert not verdict(ref, cur, max_drop_pts=25)["ex_regression"]

    def test_improvement_passes(self) -> None:
        ref = [False] * 30
        assert verdict(ref, [True] * 30)["passed"]

    def test_cost_over_the_ratio_fails(self) -> None:
        ref = [True] * 30
        r = verdict(ref, list(ref), ref_cost=0.01, cur_cost=0.021)
        assert r["cost_regression"] and not r["passed"] and not r["ex_regression"]

    def test_cost_at_the_ratio_passes(self) -> None:
        ref = [True] * 30
        assert verdict(ref, list(ref), ref_cost=0.01, cur_cost=0.02)["passed"]
        assert not verdict(ref, list(ref), ref_cost=0.01, cur_cost=0.015, max_cost_ratio=1.2)[
            "passed"
        ]

    def test_zero_reference_cost_skips_the_cost_check(self) -> None:
        ref = [True] * 30
        r = verdict(ref, list(ref), ref_cost=0.0, cur_cost=1.0)
        assert r["cost_ratio"] is None and r["passed"]

    def test_only_shared_questions_count(self) -> None:
        reference = {i: {"correct": True, "cost_usd": 0.01} for i in range(10)}
        current = {i: {"correct": True, "cost_usd": 0.01} for i in range(5, 20)}
        r = evaluate(reference, current, resamples=100)
        assert (r["n"], r["n_reference"], r["n_current"]) == (5, 10, 15)

    def test_no_shared_question_is_an_error(self) -> None:
        with pytest.raises(SystemExit, match="share no question id"):
            evaluate({1: {"correct": True, "cost_usd": 0}}, {2: {"correct": True, "cost_usd": 0}})


def test_load_records_filters_the_system(tmp_path: Path) -> None:
    run = write_run(tmp_path / "r", [True, False], 0.01)
    assert set(load_records(run, "agent_v2")) == {0, 1}
    assert set(load_records(run, "baseline")) == {0}
    with pytest.raises(SystemExit, match="no predictions"):
        load_records(run, "agent_v1")
    with pytest.raises(SystemExit, match="not found"):
        load_records(tmp_path / "missing", "agent_v2")


class TestMain:
    def test_pass_prints_a_table_and_appends_to_the_step_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ref = write_run(tmp_path / "ref", [True] * 20 + [False] * 10, 0.01)
        cur = write_run(tmp_path / "cur", [True] * 20 + [False] * 10, 0.012)
        summary = tmp_path / "summary.md"
        summary.write_text("earlier\n")
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
        code = main(["--reference", str(ref), "--current", str(cur)])
        out = capsys.readouterr().out
        assert code == 0 and "| EX |" in out and "**Result: pass**" in out
        assert summary.read_text().startswith("earlier\n### Regression check: `agent_v2`")

    def test_regression_exits_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        ref = write_run(tmp_path / "ref", [True] * 24 + [False] * 6, 0.01)
        cur = write_run(tmp_path / "cur", [False] * 30, 0.01)
        assert main(["--reference", str(ref), "--current", str(cur)]) == 1
        assert "**Result: regression**" in capsys.readouterr().out

    def test_system_and_limits_are_options(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        ref = write_run(tmp_path / "ref", [True] * 30, 0.01, system="agent_v1")
        cur = write_run(tmp_path / "cur", [True] * 30, 0.05, system="agent_v1")
        args = ["--reference", str(ref), "--current", str(cur), "--system", "agent_v1"]
        assert main(args) == 1  # 5x the cost
        assert main([*args, "--max-cost-ratio", "6"]) == 0


def test_render_names_the_failing_row() -> None:
    r = verdict([True] * 30, [False] * 30)
    text = render(r, system="agent_v2", reference="ref", current="cur")
    assert "| FAIL |" in text and text.count("FAIL") == 1
    assert check_regression.DEFAULT_SYSTEM == "agent_v2"
