"""Tests for copying a judge calibration into a committed report."""

import json
from pathlib import Path

import pytest

from evals.judge.save_report import main, save_report


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _runs(tmp_path: Path) -> tuple[Path, Path]:
    items = tmp_path / "items"
    items.mkdir()
    _jsonl(
        items / "items.jsonl",
        [
            {
                "item_id": "j-01",
                "question": "q",
                "answer": "a",
                "gold_preview": {"rows": [["secret"]]},
                "ex_correct": True,
            }
        ],
    )
    _jsonl(items / "labels.jsonl", [{"item_id": "j-01", "label": "correct", "note": "ref: secret"}])
    (items / "summary.json").write_text("{}", encoding="utf-8")
    judge = tmp_path / "judge"
    judge.mkdir()
    _jsonl(
        judge / "judgements.jsonl", [{"item_id": "j-01", "label": "correct", "rationale": "secret"}]
    )
    (judge / "agreement.json").write_text("{}", encoding="utf-8")
    (judge / "run.json").write_text("{}", encoding="utf-8")
    return items, judge


def test_report_keeps_labels_and_drops_gold_quotes(tmp_path: Path) -> None:
    items, judge = _runs(tmp_path)
    out = tmp_path / "report"
    save_report(items, {"haiku": judge}, out)
    text = "".join(p.read_text(encoding="utf-8") for p in out.rglob("*.jsonl"))
    assert "secret" not in text
    item = json.loads((out / "items.jsonl").read_text(encoding="utf-8"))
    assert item["ex_correct"] is True and item["answer"] == "a"
    assert json.loads((out / "labels.jsonl").read_text(encoding="utf-8"))["label"] == "correct"
    judged = json.loads((out / "haiku" / "judgements.jsonl").read_text(encoding="utf-8"))
    assert judged == {"item_id": "j-01", "label": "correct"}
    assert (out / "haiku" / "agreement.json").exists()
    assert (out / "items_run" / "summary.json").exists()


def test_refuses_without_labels_or_agreement(tmp_path: Path) -> None:
    items, judge = _runs(tmp_path)
    (judge / "agreement.json").unlink()
    with pytest.raises(FileNotFoundError):
        save_report(items, {"haiku": judge}, tmp_path / "report")
    (items / "labels.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        save_report(items, {}, tmp_path / "report")


def test_refuses_an_existing_report_unless_forced(tmp_path: Path) -> None:
    items, judge = _runs(tmp_path)
    out = tmp_path / "report"
    save_report(items, {"haiku": judge}, out)
    with pytest.raises(FileExistsError):
        save_report(items, {"haiku": judge}, out)
    save_report(items, {"haiku": judge}, out, force=True)


def test_cli_rejects_a_judge_without_name(tmp_path: Path) -> None:
    items, judge = _runs(tmp_path)
    with pytest.raises(SystemExit):
        main(["--items", str(items), "--judge", str(judge), "--out", str(tmp_path / "r")])
