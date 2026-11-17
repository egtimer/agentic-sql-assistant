"""Each detector and its limit on synthetic records, and the monitor CLI end to end."""

import json
from pathlib import Path
from typing import Any

import pytest

from evals import monitor
from src.monitoring import detectors as d


def v2(
    qid: int,
    errors: list[str | None],
    cost: float = 0.001,
    sqls: list[str] | None = None,
    correct: bool = True,
) -> dict[str, Any]:
    sqls = sqls or [f"SELECT {i}" for i in range(len(errors))]
    return {
        "question_id": qid,
        "system": "agent_v2",
        "attempts": [{"sql": s, "error": e} for s, e in zip(sqls, errors, strict=True)],
        "cost_usd": cost,
        "correct": correct,
    }


def v1(
    qid: str,
    steps: list[tuple[str, str, bool]],
    response: str = "It is 4.",
    cost: float = 0.001,
    answer_correct: bool | None = True,
) -> dict[str, Any]:
    return {
        "id": qid,
        "system": "agent_v1",
        "steps": [{"tool": t, "input": i, "success": ok, "output": "{}"} for t, i, ok in steps],
        "response": response,
        "cost_usd": cost,
        "answer_correct": answer_correct,
    }


def test_tool_failure_rate_counts_calls_not_records() -> None:
    records = [v2(1, [None]), v2(2, ["no such column: x", None]), v2(3, [None])]
    f = d.tool_failure_rate(records, max_rate=0.2)
    assert f.value == pytest.approx(0.25) and f.alert and f.items == ["2"]
    assert not d.tool_failure_rate(records, max_rate=0.25).alert  # at the limit: no alert
    assert d.tool_failure_rate([], 0.1).value == 0.0


def test_silent_failures_need_an_answer_that_hides_the_failure() -> None:
    calc_failed = [("sql_query", "SELECT 1", True), ("calculator", "PLACEHOLDER", False)]
    said = v1("a", calc_failed, response="The calculator failed, so I did it by hand: 4.")
    hid = v1("b", calc_failed, response="It is 4.")
    fine = v1("c", [("calculator", "2+2", True)])
    f = d.silent_failures([said, hid, fine], max_share=0.0)
    assert f.alert and f.items == ["b"] and f.value == pytest.approx(1 / 3)
    no_answers = [v2(1, ["boom"])]  # a run with the answer step off
    assert not d.silent_failures(no_answers, 0.0).alert


def test_loops_by_step_count_and_by_repeated_call() -> None:
    repeated = v1("r", [("sql_query", "SELECT 1", True), ("sql_query", "SELECT  1", True)])
    long_run = v1("l", [("calculator", str(i), True) for i in range(7)])
    normal = v1("n", [("sql_query", "SELECT 1", True), ("calculator", "1+1", True)])
    assert d.is_loop(repeated, max_steps=6, max_repeats=1)
    assert d.is_loop(long_run, max_steps=6, max_repeats=1)
    assert not d.is_loop(normal, max_steps=6, max_repeats=1)
    assert not d.is_loop(repeated, max_steps=6, max_repeats=2)
    f = d.loops([repeated, normal, normal, normal], 6, 1, max_share=0.2)
    assert f.value == 0.25 and f.alert and f.items == ["r"]
    sql_repeat = v2(5, [None, None], sqls=["SELECT a", "SELECT a"])
    assert d.is_loop(sql_repeat, 6, 1)


def test_cost_mean_and_item() -> None:
    records = [v2(1, [None], cost=0.002), v2(2, [None], cost=0.06)]
    mean, item = d.cost(records, max_mean_usd=0.01, max_item_usd=0.05)
    assert mean.value == pytest.approx(0.031) and mean.alert
    assert item.alert and item.items == ["2"]
    mean, item = d.cost(records[:1], 0.01, 0.05)
    assert not mean.alert and not item.alert


def test_run_detectors_uses_the_config_and_the_shipped_config_loads() -> None:
    config = d.load_config()
    assert {"tool_failure_rate", "silent_failures", "loops", "cost", "accuracy_drop"} <= set(config)
    names = [f.detector for f in d.run_detectors([v2(1, [None])], config)]
    assert names == ["tool_failure_rate", "silent_failures", "loops", "cost_mean", "cost_item"]
    assert [f.detector for f in d.run_detectors([v2(1, [None])], {"loops": config["loops"]})] == [
        "loops"
    ]


def write_run(path: Path, records: list[dict[str, Any]]) -> Path:
    path.mkdir(parents=True)
    (path / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def test_monitor_cli_alerts_and_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good = [v2(i, [None]) for i in range(40)]
    reference = write_run(tmp_path / "ref", good)
    same = write_run(tmp_path / "same", good)
    assert monitor.main([str(same), "--reference", str(reference)]) == 0
    out = json.loads((same / "alerts.json").read_text())
    assert out["alerts"] == 0 and out["system"] == "agent_v2" and out["n_records"] == 40
    worse = [v2(i, [None], correct=i >= 30) for i in range(40)]  # 30 points lower
    bad = write_run(tmp_path / "bad", worse)
    assert monitor.main([str(bad), "--reference", str(reference)]) == 1
    found = {f["detector"]: f for f in json.loads((bad / "alerts.json").read_text())["findings"]}
    assert found["accuracy_drop"]["alert"] and found["accuracy_drop"]["value"] == pytest.approx(75)
    assert "ALERT" in capsys.readouterr().out


def test_monitor_detects_the_eval_shape(tmp_path: Path) -> None:
    run = write_run(tmp_path / "tools", [v1("a", [("calculator", "2+2", True)])])
    system, records = monitor.load_records(run, None)
    assert system == "agent_v1" and len(records) == 1
    with pytest.raises(ValueError):
        monitor.load_records(tmp_path / "nowhere", None)
    assert monitor.main([str(tmp_path / "nowhere")]) == 2


def test_accuracy_drop_needs_ci_below_zero() -> None:
    reference = [v2(i, [None]) for i in range(20)]
    one_worse = [v2(i, [None], correct=i != 0) for i in range(20)]
    f = monitor.accuracy_drop(one_worse, reference, max_drop_pts=1.0)
    assert f.value == pytest.approx(5.0) and not f.alert  # 5 points but CI reaches 0
