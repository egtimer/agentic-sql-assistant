"""Tests for the repetition driver (no subprocess is started for real)."""

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from evals.noise import repeat


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(repeat, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(repeat, "NOISE_ROOT", tmp_path / "noise")
    monkeypatch.setitem(
        repeat.EVALS, "tools", repeat.EvalKind("evals.tools.run", tmp_path / "out", "stopped")
    )
    return tmp_path


def _write_summary(run_dir: Path, **fields: Any) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(fields))


def test_run_ids_are_numbered_from_one() -> None:
    assert repeat.run_ids("x", 3) == ["x-r01", "x-r02", "x-r03"]


def test_status_and_cost_of_both_runners(tmp_path: Path) -> None:
    tools = repeat.EVALS["tools"]
    t2s = repeat.EVALS["text2sql"]
    assert repeat.status_of(tools, tmp_path / "missing") == "new"
    (tmp_path / "half").mkdir()
    assert repeat.status_of(tools, tmp_path / "half") == "partial"
    _write_summary(tmp_path / "t", stopped=None, cost_usd_total=0.02)
    assert repeat.status_of(tools, tmp_path / "t") == "done"
    assert repeat.summary_cost(repeat.read_summary(tmp_path / "t") or {}) == 0.02
    _write_summary(
        tmp_path / "s",
        aborted="budget",
        systems={
            "baseline": {"cost_usd": {"total": 0.1}},
            "agent_v2": {"cost_usd": {"total": 0.2}},
        },
    )
    assert repeat.status_of(t2s, tmp_path / "s") == "stopped"
    assert repeat.summary_cost(repeat.read_summary(tmp_path / "s") or {}) == pytest.approx(0.3)


def test_command_sets_run_id_budget_and_resume() -> None:
    kind = repeat.EVALS["tools"]
    cmd = repeat.command(kind, "x-r01", 0.03, ["--split", "test"], resume=True)
    assert cmd[1:] == [
        "-m",
        "evals.tools.run",
        "--split",
        "test",
        "--run-id",
        "x-r01",
        "--budget-usd",
        "0.03",
        "--resume",
    ]


def test_runner_owned_flags_are_refused(roots: Path) -> None:
    with pytest.raises(SystemExit):
        repeat.main(
            [
                "--eval",
                "tools",
                "--n",
                "2",
                "--prefix",
                "x",
                "--budget-usd",
                "1",
                "--",
                "--run-id",
                "y",
            ]
        )


def test_runs_skip_done_and_stop_at_the_first_capped_run(
    roots: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = roots / "out"
    _write_summary(out / "x-r01", stopped=None, cost_usd_total=0.01)
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], cwd: Path, check: bool) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        run_id = cmd[cmd.index("--run-id") + 1]
        stopped = "budget" if run_id == "x-r02" else None
        _write_summary(out / run_id, stopped=stopped, cost_usd_total=0.028)
        return subprocess.CompletedProcess(cmd, 1 if stopped else 0)

    monkeypatch.setattr(repeat.subprocess, "run", fake_run)
    rc = repeat.main(["--eval", "tools", "--n", "3", "--prefix", "x", "--budget-usd", "0.03"])
    assert rc == 1
    assert len(calls) == 1 and "x-r02" in calls[0]
    manifest = json.loads((roots / "noise" / "x" / "repeat.json").read_text())
    assert [r["status"] for r in manifest["runs"]] == ["done (skipped)", "stopped"]
    assert manifest["cost_usd_total"] == pytest.approx(0.038)
    assert "NEAR CAP" in capsys.readouterr().out


def test_dry_run_prints_commands_without_running(
    roots: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_run(*a: Any, **k: Any) -> None:
        raise AssertionError("must not run")

    monkeypatch.setattr(repeat.subprocess, "run", no_run)
    rc = repeat.main(
        [
            "--eval",
            "tools",
            "--n",
            "2",
            "--prefix",
            "x",
            "--budget-usd",
            "0.03",
            "--dry-run",
            "--",
            "--split",
            "test",
        ]
    )
    assert rc == 0
    printed = capsys.readouterr().out
    assert "x-r01" in printed and "x-r02" in printed
