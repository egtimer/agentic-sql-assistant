"""Failure categories and the automatic rules that assign them, as pure functions.

Two kinds of record are classified:

- a **BIRD** prediction of agent v2 (``evals/text2sql``), joined with its final answer and the
  judge's label (``evals/failures/final_answer``) and with the shape of its result next to the
  gold one (columns and rows, computed by ``classify.py``);
- a **tool-eval** prediction of agent v1 (``evals/tools``), with its per-step trace.

A record is a failure when the user would get a wrong answer: for BIRD, EX wrong or the judge
says the answer is wrong; for the tool eval, the answer check fails or the tool selection is
not exact. Every rule that fires is kept as a tag; the **primary** category is the first tag
in ``PRIORITY`` (the most specific cause first). Rules only read the record: no LLM, no labels.
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any

from evals.tools.metrics import check_answer  # the eval's own answer check

SILENT_TOOL_FAILURE = "silent_tool_failure"
WRONG_TOOL = "wrong_tool"
BAD_ARGUMENTS = "bad_arguments"
LOOP = "loop_or_extra_steps"
SCHEMA_HALLUCINATION = "schema_hallucination"
GUARD_REFUSAL = "guard_refusal"
ANSWER_MISREAD = "answer_misread"
ANSWER_ON_LARGE_RESULT = "answer_on_large_result"
RETRIEVAL_MISS = "retrieval_miss"
TOOL_OUTPUT_CUT = "tool_output_cut"
WRONG_OUTPUT_SHAPE = "wrong_output_shape"
WRONG_QUERY_LOGIC = "wrong_query_logic"
ANSWER_OK_EX_WRONG = "answer_ok_ex_wrong"
OTHER = "other"
# Only a reviewer can give these two: no rule can tell them from the record.
NOT_A_FAILURE = "not_a_failure"
BENCHMARK_ISSUE = "benchmark_issue"

DESCRIPTIONS: dict[str, str] = {
    GUARD_REFUSAL: "the SQL guard refused a legitimate query (`blocked:`)",
    SCHEMA_HALLUCINATION: "the query names a column or table that does not exist",
    BAD_ARGUMENTS: "a tool got an input it cannot run: SQL syntax error, calculator parse "
    "error, or an unfilled `PLACEHOLDER`",
    SILENT_TOOL_FAILURE: "a tool failed or returned nothing and the answer does not say so",
    WRONG_TOOL: "a required tool was not called, or a tool that was not needed was",
    LOOP: "the agent used up its fixes or repeated the same call with the same input",
    ANSWER_ON_LARGE_RESULT: "the SQL is right (EX) but the result is larger than what the "
    "answer step sees, and the answer is judged wrong",
    ANSWER_MISREAD: "the SQL or tool result holds the right values and the answer gets them wrong",
    TOOL_OUTPUT_CUT: "a tool returned the answer, but after the first 1,500 characters of its "
    "output, which is all the agent's prompt keeps",
    RETRIEVAL_MISS: "the document search did not return the passage with the answer",
    ANSWER_OK_EX_WRONG: "EX marks the SQL wrong but the judge accepts the answer (usually extra "
    "columns or another form of the same value)",
    WRONG_OUTPUT_SHAPE: "the result has more or fewer columns than the gold one",
    WRONG_QUERY_LOGIC: "same columns as the gold result, different values or rows (filters, "
    "joins, aggregation)",
    OTHER: "none of the rules fired",
    NOT_A_FAILURE: "the reviewer found the answer acceptable",
    BENCHMARK_ISSUE: "the gold query or the question is wrong or ambiguous (e.g. the gold SQL "
    "does not follow the external knowledge)",
}

PRIORITY: tuple[str, ...] = (
    GUARD_REFUSAL,
    SCHEMA_HALLUCINATION,
    BAD_ARGUMENTS,
    SILENT_TOOL_FAILURE,
    WRONG_TOOL,
    LOOP,
    ANSWER_ON_LARGE_RESULT,
    TOOL_OUTPUT_CUT,
    ANSWER_MISREAD,
    RETRIEVAL_MISS,
    ANSWER_OK_EX_WRONG,
    WRONG_OUTPUT_SHAPE,
    WRONG_QUERY_LOGIC,
    OTHER,
)
CATEGORIES: tuple[str, ...] = (*PRIORITY, NOT_A_FAILURE, BENCHMARK_ISSUE)

DEFAULT_MAX_FIXES = 2  # agent v2's fix budget in the evaluated runs
ANSWER_PREVIEW_ROWS = 20  # rows of a result the respond step sees (src/sql_agent/executor.py)
AGENT_TOOL_OUTPUT_CHARS = 1500  # tool output kept in agent v1's prompts (src/agent/nodes.py)

_SCHEMA_ERROR = re.compile(r"no such (column|table)", re.IGNORECASE)
_SYNTAX_ERROR = re.compile(
    r"syntax error|incomplete input|unrecognized token|Unsupported expression|"
    r"could not parse|invalid syntax",
    re.IGNORECASE,
)
# What an answer says when it reports that a tool or query did not give it the data.
_MENTIONS_FAILURE = re.compile(
    r"fail|error|could ?n[o']t|unable|cannot|can't|not (?:available|found|returned)|"
    r"no (?:data|result|rows)|returned nothing|empty|manual|no pude|no se pudo|fall[oó]",
    re.IGNORECASE,
)


def mentions_failure(answer: str) -> bool:
    """True when the answer says that a tool or query did not give it the data."""
    return bool(_MENTIONS_FAILURE.search(answer or ""))


def primary(tags: Sequence[str]) -> str:
    """First tag in ``PRIORITY`` order; ``other`` for no tags."""
    for category in PRIORITY:
        if category in tags:
            return category
    return OTHER


def _error_tags(error: str | None) -> list[str]:
    if not error:
        return []
    if error.startswith("blocked:"):
        return [GUARD_REFUSAL]
    if _SCHEMA_ERROR.search(error):
        return [SCHEMA_HALLUCINATION]
    if _SYNTAX_ERROR.search(error):
        return [BAD_ARGUMENTS]
    return []


def _row_count(value: Any) -> int | None:
    """Row count as the runner stores it: an int, ``">1000"`` or ``None``/``"n/a"``."""
    if value is None:
        return None
    text = str(value)
    if text.startswith(">"):
        return int(text[1:]) + 1
    return int(text) if text.isdigit() else None


def bird_is_failure(record: Mapping[str, Any]) -> bool:
    """EX wrong, or the judge says the final answer is wrong."""
    return not record["correct"] or record.get("judge_label") == "incorrect"


def bird_tags(
    record: Mapping[str, Any],
    shape: Mapping[str, Any] | None = None,
    *,
    max_fixes: int = DEFAULT_MAX_FIXES,
) -> list[str]:
    """Rules for one agent v2 BIRD prediction joined with its answer and judge label.

    ``record``: the prediction (``correct``, ``pred_sql``, ``attempts``, ``fixes_used``) plus
    ``answer`` and ``judge_label``. ``shape``: ``{"pred_columns", "gold_columns",
    "pred_rows", "gold_rows"}`` of the re-executed final and gold SQL (None when unknown).
    """
    tags: list[str] = []
    attempts = list(record.get("attempts") or [])
    final = next((a for a in reversed(attempts) if a["sql"] == record["pred_sql"]), None)
    final_error = final["error"] if final is not None else record.get("error")
    tags += _error_tags(final_error)
    if final_error and not mentions_failure(record.get("answer", "")):
        tags.append(SILENT_TOOL_FAILURE)
    sqls = [" ".join(a["sql"].split()) for a in attempts]
    if record.get("fixes_used", 0) >= max_fixes or len(sqls) != len(set(sqls)):
        tags.append(LOOP)
    ex_ok = bool(record["correct"])
    judge = record.get("judge_label")
    if ex_ok and judge == "incorrect":
        rows = _row_count(final["row_count"]) if final is not None else None
        large = rows is not None and rows > ANSWER_PREVIEW_ROWS
        tags.append(ANSWER_ON_LARGE_RESULT if large else ANSWER_MISREAD)
    if not ex_ok:
        if judge == "correct":
            tags.append(ANSWER_OK_EX_WRONG)
        if shape is not None and shape.get("pred_columns") is not None:
            if shape["pred_columns"] != shape["gold_columns"]:
                tags.append(WRONG_OUTPUT_SHAPE)
            else:
                tags.append(WRONG_QUERY_LOGIC)
    return list(dict.fromkeys(tags))


def tools_is_failure(record: Mapping[str, Any]) -> bool:
    """The answer check fails, or the tool selection is not exact."""
    return record.get("answer_correct") is False or record.get("exact") is False


def _step_failed(step: Mapping[str, Any]) -> bool:
    if not step.get("success", True):
        return True
    output = str(step.get("output") or "")
    return '"row_count": 0' in output or '"chunks": []' in output or not output.strip()


def tools_tags(
    record: Mapping[str, Any], checks: Sequence[Mapping[str, Any]] | None = None
) -> list[str]:
    """Rules for one agent v1 tool-eval prediction with its steps.

    ``checks``: the query's answer checks with their gold values (``evals/tools/set``), used
    to tell a result that held the answer (misread) from one that did not (retrieval miss).
    """
    tags: list[str] = []
    steps = list(record.get("steps") or [])
    for step in steps:
        if str(step.get("input", "")).strip() == "PLACEHOLDER":
            tags.append(BAD_ARGUMENTS)
        tags += _error_tags(step.get("error"))
    if any(_step_failed(s) for s in steps) and not mentions_failure(record.get("response", "")):
        tags.append(SILENT_TOOL_FAILURE)
    if record.get("wrong_tools") or record.get("missing_tools"):
        tags.append(WRONG_TOOL)
    calls = [(s.get("tool"), " ".join(str(s.get("input", "")).split())) for s in steps]
    if len(calls) != len(set(calls)):
        tags.append(LOOP)
    if record.get("answer_correct") is False and checks:
        outputs = [str(s.get("output") or "") for s in steps]
        seen = "\n".join(o[:AGENT_TOOL_OUTPUT_CHARS] for o in outputs)
        if check_answer(seen, checks):
            tags.append(ANSWER_MISREAD)
        elif check_answer("\n".join(outputs), checks):
            tags.append(TOOL_OUTPUT_CUT)
        elif any(s.get("tool") == "doc_search" for s in steps):
            tags.append(RETRIEVAL_MISS)
    return list(dict.fromkeys(tags))
