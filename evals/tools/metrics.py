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
KEYWORD_WINDOW = 40  # characters between a number and its unit or keyword
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
    elif token.count(".") > 1 or (token.count(".") == 1 and len(token.split(".")[1]) == 3):
        # "1.143.120" or "214.067": dot as thousands separator (Spanish formatting);
        # gold values have at most 2 decimals
        token = token.replace(".", "")
    try:
        return float(token)
    except ValueError:
        return None


def numbers_in(text: str) -> list[float]:
    return [n for n, _, _ in number_spans(text)]


def number_spans(text: str) -> list[tuple[float, int, int]]:
    """Every number in ``text`` with its start and end offsets."""
    spans = []
    for m in _NUMBER.finditer(text):
        n = parse_number(m.group())
        if n is not None:
            spans.append((n, m.start(), m.end()))
    return spans


def _plain(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def _matches(n: float, value: float) -> bool:
    target = abs(value)
    return abs(n - target) <= max(target * REL_TOLERANCE, 0.005)


_RANGE = re.compile(r"\s*(?:-|–|to|a|and|y)\s*$")


def _near_keyword(
    plain: str, spans: Sequence[tuple[float, int, int]], i: int, keywords: Sequence[str]
) -> bool:
    """A keyword next to number ``i``: a symbol right after it, a word within the window.

    A word may not reach past another number, except across a range ("5-7 days",
    "5 to 7 days"), where the unit after the range belongs to both ends.
    """
    j = i
    while j + 1 < len(spans) and _RANGE.match(plain[spans[j][2] : spans[j + 1][1]]):
        j += 1
    end = spans[j][2]
    next_start = spans[j + 1][1] if j + 1 < len(spans) else len(plain)
    after = plain[end : min(end + KEYWORD_WINDOW, next_start)]
    start = spans[i][1]
    prev_end = spans[i - 1][2] if i > 0 else 0
    before = plain[max(start - KEYWORD_WINDOW, prev_end) : start]
    for keyword in (_plain(k) for k in keywords):
        if keyword[0].isalpha():
            if keyword in after or keyword in before:
                return True
        elif after.lstrip().startswith(keyword):
            return True
    return False


def _number_found(
    value: float, spans: Sequence[tuple[float, int, int]], plain: str, keywords: Sequence[str]
) -> bool:
    return any(
        _matches(n, value) and (not keywords or _near_keyword(plain, spans, i, keywords))
        for i, (n, _, _) in enumerate(spans)
    )


def check_answer(answer: str, checks: Sequence[Mapping[str, Any]]) -> bool | None:
    """True when every check passes, None when there are no checks."""
    if not checks:
        return None
    plain = _plain(answer)
    spans = number_spans(plain)
    for check in checks:
        if check["kind"] in ("number", "sql_number"):
            keywords = check.get("with", [])
            if not all(_number_found(float(v), spans, plain, keywords) for v in check["values"]):
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
