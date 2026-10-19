"""Kappa, accuracy, bootstrap and the agreement report, on hand-computed examples."""

import json
from pathlib import Path

import pytest

from evals.judge import agreement
from evals.judge.agreement import (
    accuracy,
    agreement_report,
    bootstrap_metric,
    cohens_kappa,
    compare,
    confusion,
)

C, N = "correct", "incorrect"


def pairs(both_c: int, c_i: int, i_c: int, both_i: int) -> tuple[list[str], list[str]]:
    """Two raters from a 2x2 table (first rater, second rater)."""
    a = [C] * both_c + [C] * c_i + [N] * i_c + [N] * both_i
    b = [C] * both_c + [N] * c_i + [C] * i_c + [N] * both_i
    return a, b


class TestKappa:
    def test_perfect_agreement_is_one(self) -> None:
        a, b = pairs(6, 0, 0, 4)
        assert cohens_kappa(a, b) == pytest.approx(1.0)
        assert accuracy(a, b) == 1.0

    def test_chance_level_is_zero(self) -> None:
        # po = 0.5 and pe = 0.5 (both raters 50/50)
        a, b = pairs(1, 1, 1, 1)
        assert accuracy(a, b) == 0.5
        assert cohens_kappa(a, b) == pytest.approx(0.0)

    def test_textbook_two_by_two(self) -> None:
        # 50 items: both yes 20, A yes/B no 5, A no/B yes 10, both no 15.
        # po = 35/50 = 0.7; A says yes 25/50, B says yes 30/50; pe = .5*.6 + .5*.4 = 0.5
        # kappa = (0.7 - 0.5) / (1 - 0.5) = 0.4
        a, b = pairs(20, 5, 10, 15)
        assert accuracy(a, b) == pytest.approx(0.7)
        assert cohens_kappa(a, b) == pytest.approx(0.4)

    def test_systematic_disagreement_is_negative(self) -> None:
        a, b = pairs(0, 5, 5, 0)
        assert cohens_kappa(a, b) == pytest.approx(-1.0)

    def test_undefined_cases(self) -> None:
        assert cohens_kappa([], []) is None
        assert cohens_kappa([C, C], [C, C]) is None  # pe = 1
        assert accuracy([], []) is None

    def test_labels_other_than_two_are_handled(self) -> None:
        # po = 2/3; pe = (1*1 + 1*2 + 1*0) / 9 = 1/3; kappa = (1/3) / (2/3) = 0.5
        assert cohens_kappa(["x", "y", "z"], ["x", "y", "y"]) == pytest.approx(0.5)


def test_confusion_matrix_rows_first_rater() -> None:
    a, b = pairs(20, 5, 10, 15)
    assert confusion(a, b, [C, N])["matrix"] == [[20, 5], [10, 15]]


class TestBootstrap:
    def test_interval_contains_the_estimate_and_is_deterministic(self) -> None:
        a, b = pairs(20, 5, 10, 15)
        one = bootstrap_metric(a, b, cohens_kappa, resamples=500, seed=0)
        two = bootstrap_metric(a, b, cohens_kappa, resamples=500, seed=0)
        assert one == two
        assert one["ci_low"] < one["value"] < one["ci_high"]
        acc = bootstrap_metric(a, b, accuracy, resamples=500, seed=0)
        assert acc["ci_low"] < 0.7 < acc["ci_high"] and acc["frac_undefined"] == 0.0

    def test_perfect_agreement_has_a_degenerate_interval(self) -> None:
        a, b = pairs(10, 0, 0, 10)
        out = bootstrap_metric(a, b, cohens_kappa, resamples=300, seed=1)
        assert out["value"] == pytest.approx(1.0)
        assert out["ci_low"] == out["ci_high"] == pytest.approx(1.0)

    def test_undefined_resamples_are_counted_not_averaged(self) -> None:
        # one incorrect item among 4: resamples without it have a single label (pe = 1)
        a = [C, C, C, N]
        out = bootstrap_metric(a, list(a), cohens_kappa, resamples=2000, seed=0)
        assert 0.25 < out["frac_undefined"] < 0.40  # (3/4) ** 4 = 0.316
        assert out["ci_low"] == out["ci_high"] == pytest.approx(1.0)

    def test_no_items(self) -> None:
        out = bootstrap_metric([], [], accuracy, resamples=10)
        assert out["value"] is None and out["ci_low"] is None


def test_compare_uses_shared_ids_only() -> None:
    first = {"j-01": C, "j-02": N, "j-03": C}
    second = {"j-02": N, "j-03": N, "j-04": C}
    out = compare(first, second, resamples=50)
    assert out["n"] == 2
    assert out["accuracy"]["value"] == 0.5
    assert out["confusion"]["matrix"] == [[0, 1], [0, 1]]


class TestReport:
    items = [
        {"item_id": f"j-0{n}", "ex_correct": ex}
        for n, ex in zip(range(1, 7), [True, True, False, False, True, False], strict=True)
    ]
    labels = [
        {"item_id": "j-01", "label": C},
        {"item_id": "j-02", "label": C},
        {"item_id": "j-03", "label": N},
        {"item_id": "j-04", "label": "unsure"},
        {"item_id": "j-05", "label": N},
        {"item_id": "j-06", "label": N},
    ]
    judgements = [
        {"item_id": "j-01", "label": C},
        {"item_id": "j-02", "label": N},
        {"item_id": "j-03", "label": N},
        {"item_id": "j-04", "label": C},
        {"item_id": "j-05", "label": "unparsed"},
        {"item_id": "j-06", "label": N},
    ]

    def test_counts_and_exclusions(self) -> None:
        r = agreement_report(self.items, self.judgements, self.labels, resamples=50)
        assert r["counts"]["human_unsure"] == 1 and r["counts"]["judge_unparsed"] == 1
        # human without j-04, judge without j-05: they share j-01, j-02, j-03, j-06
        assert r["judge_vs_human"]["n"] == 4
        assert r["judge_vs_human"]["accuracy"]["value"] == 0.75
        # human without j-04 against EX: agree on 01, 02, 03, 06 of 5
        assert r["human_vs_ex"]["n"] == 5
        assert r["human_vs_ex"]["accuracy"]["value"] == pytest.approx(0.8)
        # judge without j-05 against EX: agree on 01, 03, 06 of 5
        assert r["judge_vs_ex"]["n"] == 5
        assert r["judge_vs_ex"]["accuracy"]["value"] == pytest.approx(0.6)
        assert r["bootstrap"]["B"] == 50

    def test_main_writes_agreement_json(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        items_dir, judge_dir = tmp_path / "items", tmp_path / "judge"
        items_dir.mkdir()
        judge_dir.mkdir()
        write = lambda path, rows: path.write_text(  # noqa: E731
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
        write(items_dir / "items.jsonl", self.items)
        write(items_dir / "labels.jsonl", [{**x, "labeler": "me"} for x in self.labels])
        write(judge_dir / "judgements.jsonl", self.judgements)
        agreement.main(
            ["--items", str(items_dir), "--judgements", str(judge_dir), "--resamples", "50"]
        )
        report = json.loads((judge_dir / "agreement.json").read_text())
        assert report["labeler"] == "me" and report["judge_vs_human"]["n"] == 4
        out = capsys.readouterr().out
        assert "judge vs human" in out and "kappa" in out
