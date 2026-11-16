"""Tests for the run-to-run noise statistics and the analysis of repeated runs."""

import json
from pathlib import Path

import numpy as np
import pytest

from evals.noise import analyze, stats


def test_dispersion_and_flips_of_a_known_matrix() -> None:
    m = stats.as_matrix([[True, True, True], [False, True, False], [False, False, False]])
    d = stats.dispersion(m)
    assert d["per_run"] == pytest.approx([1 / 3, 2 / 3, 1 / 3])
    assert d["mean"] == pytest.approx(4 / 9)
    assert d["sd"] == pytest.approx(np.std([1 / 3, 2 / 3, 1 / 3], ddof=1))
    assert (d["min"], d["max"]) == pytest.approx((1 / 3, 2 / 3))
    f = stats.flips(m, ["a", "b", "c"])
    assert f["changing_ids"] == ["b"]
    assert (f["n_always_right"], f["n_always_wrong"]) == (1, 1)


def test_identical_runs_have_no_within_item_variance() -> None:
    m = stats.as_matrix([[True] * 4, [False] * 4, [True] * 4, [False] * 4])
    v = stats.variance_components(m)
    assert v["within_item"] == 0.0 and v["icc"] == pytest.approx(1.0)
    ci = stats.two_stage_ci(m, resamples=500)
    # with no run noise the two-stage interval is the item-only one
    assert ci["ci_low"] == pytest.approx(ci["single_run"]["ci_low"])
    assert ci["ci_high"] == pytest.approx(ci["single_run"]["ci_high"])


def test_run_noise_widens_the_interval() -> None:
    rng = np.random.default_rng(1)
    m = stats.as_matrix((rng.random((200, 5)) < 0.5).tolist())
    ci = stats.two_stage_ci(m, resamples=2000)
    single = ci["single_run"]
    assert ci["ci_high"] - ci["ci_low"] < single["ci_high"] - single["ci_low"]
    v = stats.variance_components(m)
    # pure coin flips: nearly all the variance is within items
    assert v["icc"] < 0.1


def test_paired_difference_is_exact_when_systems_differ_on_fixed_items() -> None:
    a = stats.as_matrix([[True, True], [True, True], [False, False], [True, True]])
    b = stats.as_matrix([[True, True], [False, False], [False, False], [True, True]])
    p = stats.paired_two_stage(a, b, resamples=1000)
    assert p["diff"] == pytest.approx(0.25)
    assert p["per_run_diff"] == pytest.approx([0.25, 0.25])
    assert 0.0 <= p["ci_low"] <= 0.25 <= p["ci_high"]
    with pytest.raises(ValueError):
        stats.paired_two_stage(a, b[:3])


def test_items_needed_matches_the_textbook_formula() -> None:
    # (1.96 + 0.8416)^2 * 0.3^2 / 0.03^2 = 784.9 -> 785
    assert stats.items_needed(0.3, 0.03) == 785


def test_power_analysis_more_runs_need_fewer_items() -> None:
    rng = np.random.default_rng(2)
    p = rng.random(300)[:, None]
    a = stats.as_matrix((rng.random((300, 4)) < p).tolist())
    b = stats.as_matrix((rng.random((300, 4)) < p).tolist())
    pw = stats.power_analysis(a, b, delta=0.05, runs_options=(1, 4), simulations=400)
    one, four = pw["by_runs"]["1"], pw["by_runs"]["4"]
    assert four["items_needed"] < one["items_needed"]
    # the simulation agrees with the formula, also when runs per item > 1
    assert 0.7 <= one["simulated_power_at_n"] <= 0.9
    assert 0.7 <= four["simulated_power_at_n"] <= 0.9


def _write_run(root: Path, name: str, rows: list[dict[str, object]]) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def test_analyze_tools_runs_drops_unscored_items_and_checks_the_reference(tmp_path: Path) -> None:
    def rows(correct_q2: bool) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for system, base in (("agent_v1", True), ("no_tools", False)):
            out += [
                {"id": "q1", "system": system, "answer_correct": base, "exact": base},
                {
                    "id": "q2",
                    "system": system,
                    "answer_correct": correct_q2 and base,
                    "exact": base,
                },
                {"id": "q3", "system": system, "answer_correct": None, "exact": base},
            ]
        return out

    runs = [_write_run(tmp_path, f"r{i}", rows(i % 2 == 0)) for i in range(4)]
    ref = _write_run(tmp_path, "ref", rows(True))
    report = analyze.analyze(
        "tools",
        runs,
        "answer_correct",
        [("agent_v1", "no_tools")],
        reference=ref,
        resamples=300,
    )
    s = report["systems"]["agent_v1"]
    assert s["dropped_items"] == 0 and s["dispersion"]["n_items"] == 2
    assert s["flips"]["changing_ids"] == ["q2"]
    assert s["dispersion"]["per_run"] == pytest.approx([1.0, 0.5, 1.0, 0.5])
    assert s["reference"]["accuracy"] == 1.0 and s["reference"]["within_repetition_range"]
    paired = report["comparisons"]["agent_v1-no_tools"]["paired"]
    assert paired["diff"] == pytest.approx(0.75)
    assert "agent_v1-no_tools" in analyze.format_report(report)


def test_analyze_refuses_a_metric_of_the_other_eval(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "r1", [{"question_id": 1, "system": "baseline", "correct": True}])
    with pytest.raises(ValueError):
        analyze.analyze("text2sql", [run], "answer_correct", [])
    report = analyze.analyze("text2sql", [run], "correct", [], resamples=50)
    assert report["systems"]["baseline"]["dispersion"]["mean"] == 1.0
    assert "variance_components" not in report["systems"]["baseline"]


def test_api_failures_are_missing_not_wrong(tmp_path: Path) -> None:
    credit = "BadRequestError: Error code: 400 - credit balance is too low"
    good = [
        {"question_id": i, "system": "baseline", "correct": True, "error": None} for i in range(4)
    ]
    bad = [
        *good[:2],
        *[
            {"question_id": i, "system": "baseline", "correct": False, "error": credit}
            for i in (2, 3)
        ],
    ]
    runs = [_write_run(tmp_path, "r1", good), _write_run(tmp_path, "r2", bad)]
    report = analyze.analyze("text2sql", runs, "correct", [], resamples=50)
    s = report["systems"]["baseline"]
    # the two failed items are dropped (not in every run), not counted as wrong
    assert s["dispersion"]["per_run"] == [1.0, 1.0]
    assert s["dropped_items"] == 2
    assert s["provider_failures_left_out"] == [0, 2]
