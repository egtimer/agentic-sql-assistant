"""Judge prompt, parsing and the single call."""

from typing import Any

import pytest

from evals.judge.judge import (
    JUDGE_SYSTEM,
    format_preview,
    judge_answer,
    judge_user,
    parse_verdict,
)
from src.llm import CallUsage, Completion

PREVIEW = {"columns": ["name", "n"], "rows": [["a", "1"], ["b", "2"]], "row_count": 2}


class Fixed:
    def __init__(self, text: str, stop_reason: str = "end_turn") -> None:
        self.text, self.stop_reason = text, stop_reason
        self.seen: list[dict[str, Any]] = []

    def complete(self, system: Any, user: str, *, tag: str, prefix: Any = None) -> Completion:
        self.seen.append({"system": system, "user": user, "tag": tag})
        return Completion(self.text, CallUsage(tag=tag, stop_reason=self.stop_reason))


class TestParse:
    @pytest.mark.parametrize(
        ("text", "label", "rationale"),
        [
            ('{"rationale": "all values", "label": "correct"}', "correct", "all values"),
            (
                '```json\n{"label": "Incorrect", "rationale": "missing b"}\n```',
                "incorrect",
                "missing b",
            ),
            ('Sure! {"rationale": "x", "label": "correct"} Done.', "correct", "x"),
            ('{"note": 1} then {"label": " CORRECT ", "rationale": "y"}', "correct", "y"),
        ],
    )
    def test_readable(self, text: str, label: str, rationale: str) -> None:
        v = parse_verdict(text)
        assert (v.label, v.rationale) == (label, rationale)

    @pytest.mark.parametrize(
        "text",
        ["", "correct", "{broken", '{"label": "maybe"}', '{"label": 1}', '["correct"]'],
    )
    def test_unparsed(self, text: str) -> None:
        assert parse_verdict(text).label == "unparsed"

    def test_unparsed_keeps_a_short_excerpt(self) -> None:
        assert parse_verdict("x" * 500).rationale == "x" * 200


class TestJudgeAnswer:
    def test_one_call_tagged_judge_without_sql(self) -> None:
        client = Fixed('{"rationale": "ok", "label": "correct"}')
        v = judge_answer("How many?", "n is a count", "Two: a=1, b=2", PREVIEW, client)
        assert v.label == "correct" and len(client.seen) == 1
        call = client.seen[0]
        assert call["tag"] == "judge" and call["system"] == JUDGE_SYSTEM
        assert "Question: How many?" in call["user"]
        assert "External knowledge: n is a count" in call["user"]
        assert "a | 1" in call["user"] and "Answer:\nTwo: a=1, b=2" in call["user"]
        assert "SELECT" not in call["user"] and "You do not see the query" in JUDGE_SYSTEM

    @pytest.mark.parametrize("reason", ["refusal", "max_tokens"])
    def test_failed_completion_is_unparsed(self, reason: str) -> None:
        client = Fixed('{"label": "correct"}', stop_reason=reason)
        assert judge_answer("q", "", "a", PREVIEW, client).label == "unparsed"


class TestPrompt:
    def test_no_evidence_line_when_empty(self) -> None:
        assert "External knowledge" not in judge_user("q", "  ", "a", PREVIEW)

    def test_empty_answer_is_marked(self) -> None:
        assert judge_user("q", "", " ", PREVIEW).endswith("Answer:\n(empty)")

    def test_preview_shows_when_rows_are_cut(self) -> None:
        cut = {"columns": ["x"], "rows": [["1"]] * 10, "row_count": 42}
        text = format_preview(cut)
        assert "Row count: 42" in text and "First 10 rows:" in text
        assert "Rows:" in format_preview(PREVIEW) and "First" not in format_preview(PREVIEW)
