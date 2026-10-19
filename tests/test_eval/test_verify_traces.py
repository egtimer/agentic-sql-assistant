"""verify_traces: checks on synthetic Langfuse API data, plus a real round trip (opt-in)."""

import json
import time
from pathlib import Path
from typing import Any

import pytest

from evals.observability import verify_traces as vt
from evals.text2sql import smoke
from src.config import settings
from src.observability import tracing


def obs(kind: str, name: str, oid: str, parent: str | None, cost: float | None = 0.0) -> dict:
    o: dict[str, Any] = {
        "id": oid,
        "type": kind,
        "name": name,
        "traceId": "T",
        "parentObservationId": parent,
    }
    if kind == "GENERATION":
        o.update(
            model="m",
            usage={"input": 10, "output": 2, "total": 12, "unit": "TOKENS"},
            calculatedTotalCost=cost,
        )
    return o


def v2_trace(with_check: bool = True) -> list[dict]:
    items = [
        obs("SPAN", "sql_agent_v2", "root", None),
        obs("SPAN", "introspect", "i", "root"),
        obs("SPAN", "generate", "g", "root"),
        obs("GENERATION", "generate", "gg", "g"),
        obs("SPAN", "execute", "e", "root"),
        obs("SPAN", "respond", "r", "root"),
    ]
    if with_check:
        items += [obs("SPAN", "check", "c", "root"), obs("GENERATION", "check", "cg", "c")]
    return items


SCORES = [
    {"name": "execution_accuracy", "value": 1.0},
    {"name": "cost_usd", "value": 0.0},
]
RECORD = {
    "system": "agent_v2",
    "question_id": 1,
    "n_llm_calls": 2,
    "cost_usd": 0.0,
    "correct": True,
    "attempts": [{}],
    "verdicts": [{}],
}
TRACE = {"id": "T", "name": "text2sql/agent_v2"}


def test_good_v2_trace_passes() -> None:
    entry = vt.check_trace(TRACE, v2_trace(), SCORES, RECORD)
    assert entry["ok"], entry["problems"]
    assert entry["generations"] == 2 and entry["spans"]["execute"] == 1


def test_missing_span_wrong_generation_count_and_missing_score_are_reported() -> None:
    items = [o for o in v2_trace() if o["id"] not in ("c", "cg")]
    entry = vt.check_trace(TRACE, items, SCORES[:1], RECORD)
    problems = " | ".join(entry["problems"])
    assert "span 'check': found 0, expected 1" in problems
    assert "generations: found 1, predictions record 2" in problems
    assert "score 'cost_usd' is missing" in problems


def test_nesting_problems_are_reported() -> None:
    items = v2_trace()
    items[2]["parentObservationId"] = None  # second root
    items[4]["parentObservationId"] = "nope"  # unknown parent
    problems = " | ".join(vt.check_trace(TRACE, items, SCORES, RECORD)["problems"])
    assert "expected one root span" in problems and "unknown parent" in problems


def test_generation_without_usage_or_cost_and_wrong_score() -> None:
    items = v2_trace()
    items[3].update(usage={"input": 0, "output": 0, "total": 0}, calculatedTotalCost=None)
    scores = [{"name": "execution_accuracy", "value": 0.0}, SCORES[1]]
    problems = " | ".join(vt.check_trace(TRACE, items, scores, RECORD)["problems"])
    assert "no token usage" in problems and "has no cost" in problems
    assert "execution_accuracy score differs" in problems


def test_baseline_has_no_spans_and_cost_must_match() -> None:
    gen = obs("GENERATION", "baseline", "g", None, cost=0.5)
    trace = {"id": "T", "name": "text2sql/baseline"}
    record = {"system": "baseline", "question_id": 1, "n_llm_calls": 1, "cost_usd": 0.5,
              "correct": True}  # fmt: skip
    assert vt.check_trace(trace, [gen], SCORES, record)["ok"]
    record["cost_usd"] = 0.4
    assert not vt.check_trace(trace, [gen], SCORES, record)["ok"]
    bad = vt.check_trace(trace, [gen, obs("SPAN", "reason", "s", None)], SCORES, None)
    assert any("no node spans" in p for p in bad["problems"])


def test_verify_reports_missing_traces(tmp_path: Path) -> None:
    (tmp_path / "predictions.jsonl").write_text(
        json.dumps({**RECORD, "trace_id": "T"}) + "\n", encoding="utf-8"
    )

    def fetch(path: str, params: dict[str, Any]) -> dict[str, Any]:
        return {"data": [], "meta": {"totalPages": 1}}

    report = vt.verify(fetch, "s", tmp_path, wait_s=0, poll_s=0)
    assert not report["passed"] and any(
        "trace T of question 1 not found" in p for p in report["problems"]
    )


@pytest.mark.requires_langfuse
def test_round_trip_against_a_local_langfuse(tmp_path: Path) -> None:
    """Smoke run with --trace, then the verifier, then a cost that is not zero."""
    run_id = f"pytest-{int(time.time())}"
    assert tracing.enabled(), "set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY (make langfuse-up)"
    smoke.main(["--out", str(tmp_path / "run"), "--run-id", run_id, "--trace"])
    fetch = vt.make_fetch(settings.langfuse_host, settings.langfuse_public_key,
                          settings.langfuse_secret_key)  # fmt: skip
    report = vt.verify(fetch, run_id, tmp_path / "run")
    assert report["passed"], vt.format_summary(report)
    assert report["n_checked"] == 8

    with tracing.trace("cost-check", session_id=run_id) as handle:
        tracing.record_generation(
            "g", model="m", input="i", output="o",
            usage={"input": 10, "output": 5, "cache_read": 0, "cache_write": 0},
            cost_usd=0.0123,
        )  # fmt: skip
    tracing.flush()
    deadline = time.monotonic() + 30
    while True:
        detail = fetch(f"/api/public/traces/{handle.id}", {})
        generations = [o for o in detail.get("observations", []) if o["type"] == "GENERATION"]
        if generations or time.monotonic() > deadline:
            break
        time.sleep(1)
    assert generations[0]["calculatedTotalCost"] == pytest.approx(0.0123)
