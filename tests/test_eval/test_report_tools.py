"""Tests for the report tools: cross-report comparison and the cost vs EX chart."""

import json
from pathlib import Path

import pytest

from evals.text2sql import compare_runs, plot_cost_ex


def _write_report(root: Path, name: str, rows: list[tuple[str, int, bool]]) -> None:
    report = root / name
    report.mkdir(parents=True)
    with open(report / "predictions.jsonl", "w", encoding="utf-8") as f:
        for system, qid, correct in rows:
            f.write(json.dumps({"system": system, "question_id": qid, "correct": correct}) + "\n")


class TestCompareRuns:
    def test_load_outcomes_filters_by_system(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_report(tmp_path, "r1", [("a", 1, True), ("b", 1, False), ("a", 2, False)])
        monkeypatch.setattr(compare_runs, "REPORTS", tmp_path)
        assert compare_runs.load_outcomes("r1:a") == {1: True, 2: False}

    def test_missing_system_exits(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _write_report(tmp_path, "r1", [("a", 1, True)])
        monkeypatch.setattr(compare_runs, "REPORTS", tmp_path)
        with pytest.raises(SystemExit):
            compare_runs.load_outcomes("r1:zzz")

    def test_main_reports_b_minus_a_on_common_questions(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _write_report(tmp_path, "ra", [("s", q, q <= 2) for q in range(1, 6)])
        _write_report(tmp_path, "rb", [("s", q, q <= 4) for q in range(1, 7)])
        monkeypatch.setattr(compare_runs, "REPORTS", tmp_path)
        monkeypatch.setattr("sys.argv", ["compare_runs", "ra:s", "rb:s"])
        compare_runs.main()
        out = capsys.readouterr().out
        assert "n = 5 common questions" in out
        assert "B - A: +40.0 pts" in out


class TestPlot:
    def _points(self) -> list[plot_cost_ex.Point]:
        return [
            plot_cost_ex.Point("Haiku 5.5", 0, "baseline", 0.0002, 0.60, 0.55, 0.65, 350),
            plot_cost_ex.Point("Haiku 5.5", 0, "agent_v2", 0.0004, 0.64, 0.59, 0.69, 350),
            plot_cost_ex.Point("Sonnet 5.5", 1, "baseline", 0.002, 0.66, 0.61, 0.71, 350),
            plot_cost_ex.Point("Sonnet 5.5", 1, "agent_v2", 0.005, 0.70, 0.65, 0.75, 350),
        ]

    @pytest.mark.parametrize("theme", ["light", "dark"])
    def test_render_is_svg_with_one_label_per_point(self, theme: str) -> None:
        svg = plot_cost_ex.render(self._points(), plot_cost_ex.THEMES[theme])
        assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
        assert plot_cost_ex.THEMES[theme]["surface"] in svg
        for label in ("Haiku · single shot 60.0%", "Sonnet · agent v2 70.0%"):
            assert label in svg

    def test_description_lists_every_point(self) -> None:
        svg = plot_cost_ex.render(self._points(), plot_cost_ex.THEMES["light"])
        desc = svg.split("<desc id='d'>")[1].split("</desc>")[0]
        assert desc.count("EX ") == 4
