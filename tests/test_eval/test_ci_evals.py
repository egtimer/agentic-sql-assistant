"""Offline eval checks used by CI: the guard check criteria and the runner smoke test."""

import copy
import json
from pathlib import Path

import pytest

from evals.guardrails import run_sql
from evals.text2sql import smoke


def passing_summary() -> dict[str, object]:
    return {
        "guards": {
            "sql_guard": {
                "attacks": {"n": 22, "expectation_met": 22, "side_effects": 0},
                "benign_cases": {"n": 2, "false_blocks": 0, "side_effects": 0},
                "benign_gold": None,
            }
        }
    }


class TestCheckSummary:
    def test_pass(self) -> None:
        assert run_sql.check_summary(passing_summary()) == []

    @pytest.mark.parametrize(
        ("path", "value", "fragment"),
        [
            (("attacks", "expectation_met"), 21, "expectations met: 21/22"),
            (("attacks", "side_effects"), 1, "side effects"),
            (("attacks", "n"), 0, "no attack case ran"),
            (("benign_cases", "false_blocks"), 1, "1 false blocks"),
            (("benign_cases", "side_effects"), 2, "2 side effects"),
            (("benign_cases", "n"), 0, "no case ran"),
        ],
    )
    def test_failures(self, path: tuple[str, str], value: int, fragment: str) -> None:
        summary = copy.deepcopy(passing_summary())
        summary["guards"]["sql_guard"][path[0]][path[1]] = value  # type: ignore[index]
        failures = run_sql.check_summary(summary)
        assert any(fragment in f for f in failures), failures

    def test_gold_is_checked_when_present(self) -> None:
        summary = copy.deepcopy(passing_summary())
        summary["guards"]["sql_guard"]["benign_gold"] = {  # type: ignore[index]
            "n": 500,
            "false_blocks": 3,
            "side_effects": 0,
        }
        assert any("benign_gold" in f for f in run_sql.check_summary(summary))

    def test_missing_guard(self) -> None:
        assert run_sql.check_summary({"guards": {}}) == ["guard 'sql_guard' was not run"]


def test_run_sql_uses_the_stand_in_when_bird_data_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import evals.guardrails.cases as cases

    monkeypatch.setattr(cases, "QUESTIONS_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(cases, "db_path", lambda db, *a, **k: tmp_path / "missing" / db)
    run_sql.main(["--out", str(tmp_path / "out"), "--guards", "sql_guard", "--check"])
    captured = capsys.readouterr()
    assert 'databases: {"student_club": "stand-in"}' in captured.out
    assert "CHECK PASSED" in captured.out and "gold queries skipped" in captured.err
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["databases"] == {"student_club": "stand-in"}
    assert summary["skipped"] == {"attacks": 0, "benign": 0, "gold": True}
    block = summary["guards"]["sql_guard"]
    assert block["attacks"]["n"] == len(cases.read_jsonl(cases.ATTACKS_PATH))
    assert block["attacks"]["expectation_met"] == block["attacks"]["n"]
    assert block["benign_cases"]["n"] == len(cases.read_jsonl(cases.BENIGN_PATH))
    assert block["benign_cases"]["false_blocks"] == 0


def test_smoke_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    smoke.main(["--out", str(tmp_path / "run")])
    assert "SMOKE PASSED" in capsys.readouterr().out
    assert (tmp_path / "run" / "run.json").exists()


def test_smoke_fails_when_a_number_differs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(smoke.EXPECTED["agent_v2"], "correct", 4)
    with pytest.raises(SystemExit) as exc:
        smoke.main(["--out", str(tmp_path / "run")])
    assert exc.value.code == 1
