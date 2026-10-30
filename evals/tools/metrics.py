"""Scores for the agent v1 tool-selection eval: tool choice per query, final answer, aggregates.

Tool choice uses the set of distinct tools the agent executed. With ``allowed = required |
optional``:

- precision: share of the used tools that are allowed (undefined when no tool was used)
- recall: share of the required tools that were used (undefined when none is required)
- exact: every required tool used and nothing outside ``allowed`` (for a ``none`` query:
  no tool at all)

The final answer passes when every check passes (see ``specs.py``); queries without checks
are not scored. Aggregates are means over the queries where a score is defined, with 95%
percentile bootstrap CIs over queries (``evals.text2sql.stats`` settings).
"""

import random
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from evals.text2sql.stats import (
    CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    _percentile_interval,
)

REL_TOLERANCE = 0.01
_NUMBER = re.compile(r"\d[\d.,]*")


def tool_scores(
    used: Iterable[str], required: Sequence[str], optional: Sequence[str] = ()
) -> dict[str, Any]:
    used_set = set(used)
    required_set, allowed = set(required), set(required) | set(optional)
    return {
        "used": sorted(used_set),
        "precision": len(used_set & allowed) / len(used_set) if used_set else None,
        "recall": len(used_set & required_set) / len(required_set) if required_set else None,
        "exact": required_set <= used_set and used_set <= allowed,
        "wrong_tools": sorted(used_set - allowed),
        "missing_tools": sorted(required_set - used_set),
    }


def parse_number(token: str) -> float | None:
    """A number written with ``.`` or ``,`` as thousands or decimal separator."""
    token = token.rstrip(".,")
    if "," in token and "." in token:
        decimal = "." if token.rfind(".") > token.rfind(",") else ","
        thousands = "," if decimal == "." else "."
        token = token.replace(thousands, "").replace(decimal, ".")
    elif "," in token:
        groups = token.split(",")
        if len(groups) > 2 or len(groups[-1]) == 3:
            token = token.replace(",", "")
        else:
            token = token.replace(",", ".")
    elif token.count(".") > 1:
        token = token.replace(".", "")
    try:
        return float(token)
    except ValueError:
        return None


def numbers_in(text: str) -> list[float]:
    return [n for n in (parse_number(t) for t in _NUMBER.findall(text)) if n is not None]


def _plain(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def _number_found(value: float, found: Sequence[float]) -> bool:
    target = abs(value)
    tolerance = max(target * REL_TOLERANCE, 0.005)
    return any(abs(n - target) <= tolerance for n in found)


def check_answer(answer: str, checks: Sequence[Mapping[str, Any]]) -> bool | None:
    """True when every check passes, None when there are no checks."""
    if not checks:
        return None
    found = numbers_in(answer)
    plain = _plain(answer)
    for check in checks:
        if check["kind"] in ("number", "sql_number"):
            if not all(_number_found(float(v), found) for v in check["values"]):
                return False
        else:
            aliases = check.get("aliases", {})
            for value in check["values"]:
                options = [value, *aliases.get(value, [])]
                if not any(_plain(o) in plain for o in options):
                    return False
    return True


def mean_ci(
    values: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = CONFIDENCE,
) -> dict[str, float] | None:
    n = len(values)
    if n == 0:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(resamples))
    low, high = _percentile_interval(means, confidence)
    return {"mean": sum(values) / n, "ci_low": low, "ci_high": high, "n": n}


def aggregate(rows: Sequence[Mapping[str, Any]], **bootstrap: Any) -> dict[str, Any]:
    """Means with CIs of the per-query scores, plus cost, latency and LLM calls."""

    def defined(key: str) -> list[float]:
        return [float(r[key]) for r in rows if r.get(key) is not None]

    return {
        "n": len(rows),
        "tool_precision": mean_ci(defined("precision"), **bootstrap),
        "tool_recall": mean_ci(defined("recall"), **bootstrap),
        "tool_exact": mean_ci(defined("exact"), **bootstrap),
        "answer_correct": mean_ci(defined("answer_correct"), **bootstrap),
        "cost_usd": mean_ci(defined("cost_usd"), **bootstrap),
        "latency_s": mean_ci(defined("latency_s"), **bootstrap),
        "llm_calls": mean_ci(defined("n_llm_calls"), **bootstrap),
    }
