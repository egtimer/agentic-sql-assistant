"""Tests for the cost and latency summary of a committed run."""

import json
from pathlib import Path

import pytest

from evals.text2sql import cost_latency


def _row(system: str, qid: int, cost: float, latency_ms: float, calls: int = 1) -> dict:
    return {
        "system": system,
        "question_id": qid,
        "cost_usd": cost,
        "latency_ms": latency_ms,
        "n_llm_calls": calls,
    }


class TestSummarize:
    def test_means_quantiles_and_ci_per_system(self) -> None:
        rows = [_row("a", i, 0.001 * i, 1000.0 * i) for i in range(1, 6)]
        rows += [_row("b", 1, 0.01, 500.0, calls=2)]
        out = cost_latency.summarize(rows)
        a = out["a"]
        assert a["n"] == 5
        assert a["cost_usd"]["mean"] == pytest.approx(0.003)
        assert a["cost_usd"]["ci_low"] <= 0.003 <= a["cost_usd"]["ci_high"]
        assert a["latency_s"]["mean"] == pytest.approx(3.0)
        assert a["latency_s_median"] == pytest.approx(3.0)
        assert a["latency_s_p95"] == pytest.approx(4.8)
        assert out["b"]["llm_calls"]["mean"] == 2.0

    def test_single_question_has_a_degenerate_interval(self) -> None:
        ci = cost_latency.mean_ci([2.5])
        assert ci == {"mean": 2.5, "ci_low": 2.5, "ci_high": 2.5}
        assert cost_latency.mean_ci([]) is None


def test_main_writes_cost_latency_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = tmp_path / "run"
    report.mkdir()
    with open(report / "predictions.jsonl", "w", encoding="utf-8") as f:
        for i in range(3):
            f.write(json.dumps(_row("baseline", i, 0.0002, 2000.0)) + "\n")
    (report / "summary.json").write_text(json.dumps({"workers": 4}))
    monkeypatch.setattr(cost_latency, "REPORTS", tmp_path)
    cost_latency.main(["run"])
    out = json.loads((report / "cost_latency.json").read_text())
    assert out["workers"] == 4
    assert out["systems"]["baseline"]["cost_usd"]["mean"] == pytest.approx(0.0002)
    assert len(out["input_sha256"]) == 64
