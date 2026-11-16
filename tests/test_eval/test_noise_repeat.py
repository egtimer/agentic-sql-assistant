"""Tests for the repetition driver (no subprocess is started for real)."""

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from evals.noise import repeat

CREDIT_ERROR = (
    "BadRequestError: Error code: 400 - {'type': 'error', 'error': {'type': "
    "'invalid_request_error', 'message': 'Your credit balance is too low to access the "
    "Anthropic API.'}}"
)


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(repeat, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(repeat, "NOISE_ROOT", tmp_path / "noise")
    monkeypatch.setitem(
        repeat.EVALS, "tools", repeat.EvalKind("evals.tools.run", tmp_path / "out", "stopped")
    )
    return tmp_path


def _write_run(
    run_dir: Path,
    *,
    summary: dict[str, Any] | None = None,
    costs: Sequence[float] = (),
    errors: Sequence[str | None] = (),
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    if summary is not None:
        (run_dir / "summary.json").write_text(json.dumps(summary))
    with open(run_dir / "llm_calls.jsonl", "a") as f:
        for c in costs:
            f.write(json.dumps({"cost_usd": c}) + "\n")
    with open(run_dir / "predictions.jsonl", "a") as f:
        for i, e in enumerate(errors):
            f.write(json.dumps({"id": f"q{i}", "system": "s", "error": e}) + "\n")


def test_run_ids_are_numbered_from_one() -> None:
    assert repeat.run_ids("x", 3) == ["x-r01", "x-r02", "x-r03"]


def test_statuses(tmp_path: Path) -> None:
    tools = repeat.EVALS["tools"]
    t2s = repeat.EVALS["text2sql"]
    assert repeat.status_of(tools, tmp_path / "missing") == "new"
    (tmp_path / "half").mkdir()
    assert repeat.status_of(tools, tmp_path / "half") == "partial"
    _write_run(tmp_path / "ok", summary={"stopped": None}, errors=[None, "no such column"])
    assert repeat.status_of(tools, tmp_path / "ok") == "done"
    _write_run(tmp_path / "cap", summary={"aborted": "budget"})
    assert repeat.status_of(t2s, tmp_path / "cap") == "stopped"
    _write_run(tmp_path / "pe", summary={"stopped": "provider_error: anthropic: ..."})
    assert repeat.status_of(tools, tmp_path / "pe") == "provider_error"
    _write_run(tmp_path / "int", summary={"aborted": "interrupted"})
    assert repeat.status_of(t2s, tmp_path / "int") == "interrupted"


def test_a_finished_run_with_scored_api_failures_is_contaminated(tmp_path: Path) -> None:
    """What happened on 2026-11-16: the credit ran out mid-run, the old runner recorded
    each 400 as an error, scored it as a wrong answer and finished "clean"."""
    _write_run(tmp_path / "r", summary={"stopped": None}, errors=[None, CREDIT_ERROR, None])
    assert repeat.provider_error_rows(tmp_path / "r") == 1
    assert repeat.status_of(repeat.EVALS["tools"], tmp_path / "r") == "contaminated"


def test_spent_sums_every_invocation(tmp_path: Path) -> None:
    _write_run(tmp_path / "r", costs=[0.01, 0.02])
    _write_run(tmp_path / "r", costs=[0.005])
    assert repeat.spent_usd(tmp_path / "r") == pytest.approx(0.035)
    assert repeat.spent_usd(tmp_path / "none") == 0.0


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
    argv = ["--eval", "tools", "--n", "2", "--prefix", "x", "--budget-usd", "1"]
    with pytest.raises(SystemExit):
        repeat.main([*argv, "--", "--run-id", "y"])


def test_runs_skip_done_and_stop_at_the_first_capped_run(
    roots: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = roots / "out"
    _write_run(out / "x-r01", summary={"stopped": None}, costs=[0.01])
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], cwd: Path, check: bool) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        run_id = cmd[cmd.index("--run-id") + 1]
        stopped = "budget" if run_id == "x-r02" else None
        _write_run(out / run_id, summary={"stopped": stopped}, costs=[0.028])
        return subprocess.CompletedProcess(cmd, 1 if stopped else 0)

    monkeypatch.setattr(repeat.subprocess, "run", fake_run)
    rc = repeat.main(["--eval", "tools", "--n", "3", "--prefix", "x", "--budget-usd", "0.03"])
    assert rc == 1
    assert len(calls) == 1 and "x-r02" in calls[0]
    manifest = json.loads((roots / "noise" / "x" / "repeat.json").read_text())
    assert [r["status"] for r in manifest["runs"]] == ["done (skipped)", "stopped"]
    assert manifest["cost_usd_total"] == pytest.approx(0.038)
    assert "NEAR CAP" in capsys.readouterr().out


def test_a_provider_error_run_is_resumed_with_the_rest_of_its_cap(
    roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = roots / "out"
    _write_run(out / "x-r01", summary={"stopped": "provider_error: anthropic: 400"}, costs=[0.02])
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], cwd: Path, check: bool) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        _write_run(out / "x-r01", summary={"stopped": None}, costs=[0.005])
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(repeat.subprocess, "run", fake_run)
    rc = repeat.main(["--eval", "tools", "--n", "1", "--prefix", "x", "--budget-usd", "0.03"])
    assert rc == 0
    (cmd,) = calls
    assert cmd[-1] == "--resume"
    assert float(cmd[cmd.index("--budget-usd") + 1]) == pytest.approx(0.01)


def test_contaminated_and_spent_runs_are_not_run(
    roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = roots / "out"

    def no_run(*a: Any, **k: Any) -> None:
        raise AssertionError("must not run")

    monkeypatch.setattr(repeat.subprocess, "run", no_run)
    _write_run(out / "x-r01", summary={"stopped": None}, errors=[CREDIT_ERROR])
    argv = ["--eval", "tools", "--n", "1", "--budget-usd", "0.03"]
    assert repeat.main([*argv, "--prefix", "x"]) == 1
    _write_run(out / "y-r01", costs=[0.03])  # partial, but its whole cap is spent
    assert repeat.main([*argv, "--prefix", "y"]) == 1
    manifest = json.loads((roots / "noise" / "y" / "repeat.json").read_text())
    assert manifest["runs"][0]["status"] == "cap already spent"


def test_dry_run_prints_commands_without_running(
    roots: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_run(*a: Any, **k: Any) -> None:
        raise AssertionError("must not run")

    monkeypatch.setattr(repeat.subprocess, "run", no_run)
    argv = ["--eval", "tools", "--n", "2", "--prefix", "x", "--budget-usd", "0.03", "--dry-run"]
    assert repeat.main([*argv, "--", "--split", "test"]) == 0
    printed = capsys.readouterr().out
    assert "x-r01" in printed and "x-r02" in printed


def test_a_capped_run_resumes_only_when_asked_and_the_raised_cap_leaves_money(
    roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = roots / "out"
    _write_run(out / "x-r01", summary={"stopped": "budget"}, costs=[1.0])
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], cwd: Path, check: bool) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        _write_run(out / "x-r01", summary={"stopped": None}, costs=[0.2])
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(repeat.subprocess, "run", fake_run)
    argv = ["--eval", "tools", "--n", "1", "--prefix", "x"]
    assert repeat.main([*argv, "--budget-usd", "1.25"]) == 1 and calls == []
    assert repeat.main([*argv, "--budget-usd", "1.0", "--resume-stopped"]) == 1 and calls == []
    assert repeat.main([*argv, "--budget-usd", "1.25", "--resume-stopped"]) == 0
    (cmd,) = calls
    assert cmd[-1] == "--resume"
    assert float(cmd[cmd.index("--budget-usd") + 1]) == pytest.approx(0.25)
