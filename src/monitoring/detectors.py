"""Detectors that flag runs worth a look, from the records an agent run leaves.

They read per-question records in either shape the evals write (and a service could log):

- agent v2 (text-to-SQL): ``attempts`` (each with ``sql`` and ``error``), ``cost_usd``;
- agent v1 (tool use): ``steps`` (each with ``tool``, ``input``, ``success``, ``output``),
  ``response``, ``cost_usd``.

Each detector returns a :class:`Finding` with the measured value, the limit from
``configs/alerts.yaml`` and the ids of the records behind it; ``alert`` is True when the value
passes the limit. The detectors come from the 2026-11-17 failure analysis: a tool call that
fails, a failure the answer does not mention, a loop, and cost. A drop in accuracy against a
reference needs the eval's labels and lives in ``evals/monitor.py``.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "alerts.yaml"

# What an answer says when it reports that a tool or query did not give it the data.
_MENTIONS_FAILURE = re.compile(
    r"fail|error|could ?n[o']t|unable|cannot|can't|not (?:available|found|returned)|"
    r"no (?:data|result|rows)|returned nothing|empty|manual|no pude|no se pudo|fall[oó]",
    re.IGNORECASE,
)


def mentions_failure(answer: str) -> bool:
    """True when the answer says that a tool or query did not give it the data."""
    return bool(_MENTIONS_FAILURE.search(answer or ""))


@dataclass
class Finding:
    """One detector's result on a run."""

    detector: str
    value: float
    limit: float
    alert: bool
    detail: str
    items: list[str] = field(default_factory=list)  # record ids behind the value (max 20)

    def as_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            "value": self.value,
            "limit": self.limit,
            "alert": self.alert,
            "detail": self.detail,
            "items": self.items,
        }


def load_config(path: Path | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError("alerts config must be a mapping")
    return config


def record_id(record: Mapping[str, Any]) -> str:
    return str(record.get("question_id", record.get("id", "?")))


def tool_calls(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The record's tool calls as ``{tool, input, ok, output}``, in either shape."""
    if "steps" in record:
        return [
            {
                "tool": str(s.get("tool")),
                "input": str(s.get("input", "")),
                "ok": bool(s.get("success", True)),
                "output": str(s.get("output") or ""),
            }
            for s in record.get("steps") or []
        ]
    return [
        {
            "tool": "run_query",
            "input": str(a.get("sql", "")),
            "ok": a.get("error") is None,
            "output": "",
        }
        for a in record.get("attempts") or []
    ]


def _answer(record: Mapping[str, Any]) -> str | None:
    value = record.get("response", record.get("answer"))
    return None if value is None else str(value)


def _ids(records: Sequence[Mapping[str, Any]]) -> list[str]:
    return [record_id(r) for r in records][:20]


def tool_failure_rate(records: Sequence[Mapping[str, Any]], max_rate: float) -> Finding:
    """Share of tool calls that failed."""
    calls = [(r, c) for r in records for c in tool_calls(r)]
    failed = [r for r, c in calls if not c["ok"]]
    rate = len(failed) / len(calls) if calls else 0.0
    return Finding(
        "tool_failure_rate",
        rate,
        max_rate,
        rate > max_rate,
        f"{len(failed)} of {len(calls)} tool calls failed",
        _ids(failed),
    )


def silent_failures(records: Sequence[Mapping[str, Any]], max_share: float) -> Finding:
    """Share of answered records where a tool call failed and the answer does not say so.

    Only records that carry an answer count (an eval run with the answer step off has none).
    """
    answered = [r for r in records if _answer(r) is not None]
    silent = [
        r
        for r in answered
        if any(not c["ok"] for c in tool_calls(r)) and not mentions_failure(_answer(r) or "")
    ]
    share = len(silent) / len(answered) if answered else 0.0
    return Finding(
        "silent_failures",
        share,
        max_share,
        share > max_share,
        f"{len(silent)} of {len(answered)} answered records hide a failed tool call",
        _ids(silent),
    )


def is_loop(record: Mapping[str, Any], max_steps: int, max_repeats: int) -> bool:
    """More than ``max_steps`` tool calls, or one call (tool and input) made more than
    ``max_repeats`` times."""
    calls = tool_calls(record)
    if len(calls) > max_steps:
        return True
    seen: dict[tuple[str, str], int] = {}
    for c in calls:
        key = (c["tool"], " ".join(c["input"].split()))
        seen[key] = seen.get(key, 0) + 1
    return any(n > max_repeats for n in seen.values())


def loops(
    records: Sequence[Mapping[str, Any]], max_steps: int, max_repeats: int, max_share: float
) -> Finding:
    """Share of records that look like a loop (see :func:`is_loop`)."""
    looping = [r for r in records if is_loop(r, max_steps, max_repeats)]
    share = len(looping) / len(records) if records else 0.0
    return Finding(
        "loops",
        share,
        max_share,
        share > max_share,
        f"{len(looping)} of {len(records)} records make more than {max_steps} tool calls or "
        f"the same call (tool and input) more than {max_repeats}x",
        _ids(looping),
    )


def cost(
    records: Sequence[Mapping[str, Any]], max_mean_usd: float, max_item_usd: float
) -> list[Finding]:
    """Mean cost per question, and records above the per-question limit."""
    costs = [float(r.get("cost_usd") or 0.0) for r in records]
    mean = sum(costs) / len(costs) if costs else 0.0
    over = [r for r, c in zip(records, costs, strict=True) if c > max_item_usd]
    return [
        Finding(
            "cost_mean",
            mean,
            max_mean_usd,
            mean > max_mean_usd,
            f"mean ${mean:.5f} per question over {len(costs)}",
        ),
        Finding(
            "cost_item",
            float(len(over)),
            0.0,
            bool(over),
            f"{len(over)} questions above ${max_item_usd:g}",
            _ids(over),
        ),
    ]


def run_detectors(records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> list[Finding]:
    """Every configured detector on ``records`` (one system's records)."""
    findings: list[Finding] = []
    if "tool_failure_rate" in config:
        findings.append(tool_failure_rate(records, float(config["tool_failure_rate"]["max_rate"])))
    if "silent_failures" in config:
        findings.append(silent_failures(records, float(config["silent_failures"]["max_share"])))
    if "loops" in config:
        c = config["loops"]
        findings.append(
            loops(records, int(c["max_steps"]), int(c["max_repeats"]), float(c["max_share"]))
        )
    if "cost" in config:
        c = config["cost"]
        findings += cost(records, float(c["max_mean_usd"]), float(c["max_item_usd"]))
    return findings
