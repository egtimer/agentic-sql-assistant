"""Tests for copying a run into a committed report."""

import json
from pathlib import Path

import pytest

from evals.save_report import save_report


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text("{}", encoding="utf-8")
    (run / "run.json").write_text("{}", encoding="utf-8")
    (run / "llm_calls.jsonl").write_text('{"cost_usd": 0.1}\n', encoding="utf-8")
    rows = [{"question_id": 1, "gold_sql": "SELECT 1", "pred_sql": "SELECT 1"}]
    (run / "predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    (run / "gold_cache.json").write_text("{}", encoding="utf-8")
    return run


def test_copies_files_and_drops_gold_sql(tmp_path: Path) -> None:
    out = tmp_path / "report"
    written = save_report(_run(tmp_path), out)
    assert set(written) == {"summary.json", "run.json", "llm_calls.jsonl", "predictions.jsonl"}
    row = json.loads((out / "predictions.jsonl").read_text(encoding="utf-8"))
    assert "gold_sql" not in row and row["pred_sql"] == "SELECT 1"
    assert not (out / "gold_cache.json").exists()


def test_refuses_an_existing_report_unless_forced(tmp_path: Path) -> None:
    run, out = _run(tmp_path), tmp_path / "report"
    save_report(run, out)
    with pytest.raises(FileExistsError):
        save_report(run, out)
    save_report(run, out, force=True)


def test_refuses_an_unfinished_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        save_report(tmp_path, tmp_path / "report")
