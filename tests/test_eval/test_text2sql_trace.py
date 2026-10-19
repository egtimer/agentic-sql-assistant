"""The text-to-SQL runner's --trace option (scripted client, events captured, no network)."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from evals.text2sql import run as runner
from evals.text2sql import smoke
from src.observability import tracing
from tests.test_observability.helpers import of_type


def read_predictions(run_dir: Path) -> list[dict[str, Any]]:
    lines = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_trace_one_trace_per_pair_with_scores(
    tmp_path: Path, events: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    smoke.main(["--out", str(run_dir), "--run-id", "r1", "--trace"])
    predictions = read_predictions(run_dir)
    assert len(predictions) == 8

    created = {b["id"]: b for b in of_type(events, "trace-create") if "name" in b}
    assert len(created) == 8
    assert {r["trace_id"] for r in predictions} == set(created)
    for r in predictions:
        body = created[r["trace_id"]]
        assert body["sessionId"] == "r1"
        assert body["name"] == f"text2sql/{r['system']}"
        assert body["tags"] == [r["system"], r["db_id"], r["difficulty"], "dev"]
        assert body["metadata"]["question_id"] == r["question_id"]
        assert body["metadata"]["run_id"] == "r1"
        assert "gold" not in json.dumps(body).lower()

    scores = of_type(events, "score-create")
    for r in predictions:
        mine = {s["name"]: s for s in scores if s["traceId"] == r["trace_id"]}
        assert mine["execution_accuracy"]["value"] == (1.0 if r["correct"] else 0.0)
        assert mine["cost_usd"]["value"] == pytest.approx(r["cost_usd"])

    generations = Counter(g["traceId"] for g in of_type(events, "generation-create"))
    for r in predictions:
        assert generations[r["trace_id"]] == r["n_llm_calls"]

    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["langfuse"]["session_id"] == "r1" and meta["langfuse"]["host"]


def test_trace_is_off_by_default_even_when_configured(
    tmp_path: Path, events: list[dict[str, Any]]
) -> None:
    smoke.main(["--out", str(tmp_path / "run")])
    assert events == []
    assert all("trace_id" not in r for r in read_predictions(tmp_path / "run"))
    # the tracer is restored afterwards
    assert tracing.enabled()


def test_trace_refused_when_langfuse_is_not_configured(tmp_path: Path) -> None:
    with tracing.use_tracer(tracing.NoopTracer()):
        with pytest.raises(SystemExit, match="LANGFUSE"):
            smoke.main(["--out", str(tmp_path / "run"), "--trace"])
        with pytest.raises(runner.TracingNotConfiguredError, match="LANGFUSE"):
            runner.run_experiment(
                config=smoke.LLMConfig(provider="ollama", model="x"),
                systems=["baseline"],
                questions=[],
                run_dir=tmp_path / "other",
                run_id="x",
                split="dev",
                trace=True,
            )
    assert not (tmp_path / "other").exists()
