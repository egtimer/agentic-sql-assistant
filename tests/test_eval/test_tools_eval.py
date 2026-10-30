"""Tests for the agent v1 tool-selection set and its scores (no LLM)."""

import json
from pathlib import Path

import pytest

from evals.tools import build_set
from evals.tools.metrics import aggregate, check_answer, numbers_in, parse_number, tool_scores
from evals.tools.specs import SPECS


class TestToolScores:
    def test_required_and_optional_tools(self) -> None:
        s = tool_scores(["sql_query", "calculator"], ["sql_query"], ["calculator"])
        assert (s["precision"], s["recall"], s["exact"]) == (1.0, 1.0, True)

    def test_wrong_and_missing_tools(self) -> None:
        s = tool_scores(["doc_search"], ["sql_query"])
        assert (s["precision"], s["recall"], s["exact"]) == (0.0, 0.0, False)
        assert s["wrong_tools"] == ["doc_search"] and s["missing_tools"] == ["sql_query"]

    def test_repeated_calls_count_once(self) -> None:
        s = tool_scores(["sql_query", "sql_query", "doc_search"], ["sql_query", "doc_search"])
        assert (s["precision"], s["recall"], s["exact"]) == (1.0, 1.0, True)

    def test_no_tool_query(self) -> None:
        assert tool_scores([], [])["exact"] is True
        assert tool_scores([], [])["precision"] is None
        over = tool_scores(["sql_query"], [])
        assert (over["precision"], over["recall"], over["exact"]) == (0.0, None, False)


class TestNumbers:
    @pytest.mark.parametrize(
        ("token", "value"),
        [
            ("214,067.19", 214067.19),
            ("214.067,19", 214067.19),
            ("1,299", 1299.0),
            ("3,5", 3.5),
            ("27.14", 27.14),
            ("1.143.120", 1143120.0),
            ("214.067", 214067.0),
            ("4.5", 4.5),
            ("30.", 30.0),
        ],
    )
    def test_parse_number(self, token: str, value: float) -> None:
        assert parse_number(token) == pytest.approx(value)

    def test_numbers_in_text(self) -> None:
        assert numbers_in("Refunds take 5-7 business days (€29 setup).") == [5.0, 7.0, 29.0]


class TestCheckAnswer:
    def test_number_tolerance_and_absolute_value(self) -> None:
        checks = [{"kind": "sql_number", "values": [-27.14]}]
        assert check_answer("Norte revenue fell by about 27.1% from Q1 to Q4.", checks)
        assert not check_answer("Norte revenue fell by 30%.", checks)

    def test_every_value_must_appear(self) -> None:
        checks = [{"kind": "number", "values": [5, 7]}]
        assert check_answer("It takes 5 to 7 business days.", checks)
        assert not check_answer("It takes 5 business days.", checks)

    def test_text_ignores_accents_and_accepts_aliases(self) -> None:
        checks = [
            {
                "kind": "sql_text",
                "values": ["Electrónica"],
                "aliases": {"Electrónica": ["Electronics"]},
            }
        ]
        assert check_answer("It is in the Electronica category.", checks)
        assert check_answer("It is an Electronics product.", checks)
        assert not check_answer("It is an office product.", checks)

    def test_single_digit_needs_its_unit_or_keyword(self) -> None:
        pct = [{"kind": "number", "values": [7], "with": ["%", "percent"]}]
        assert check_answer("Gold customers get an extra 7% off.", pct)
        assert check_answer("They get 7 percent.", pct)
        assert not check_answer("Gold is one of 7 tiers; the discount is 10%.", pct)
        years = [{"kind": "number", "values": [2], "with": ["year", "ano"]}]
        assert check_answer("Electronics carry a 2-year warranty.", years)
        assert check_answer("La garantía es de 2 años.", years)
        assert not check_answer("See section 2 for details.", years)
        customers = [{"kind": "number", "values": [3], "with": ["customer", "client"]}]
        assert check_answer("The number of Platinum customers is 3.", customers)
        days = [{"kind": "number", "values": [5, 7], "with": ["business day", "dias laborables"]}]
        assert check_answer("The refund takes 5-7 business days.", days)
        assert not check_answer("The refund takes 5-7 days.", days)
        assert check_answer("The refund takes 5 to 7 business days.", days)
        assert not check_answer("Q3 had 7 regions and grew 10 percent.", pct)

    def test_no_checks_is_not_scored(self) -> None:
        assert check_answer("Hello!", []) is None


def test_aggregate_skips_undefined_scores() -> None:
    rows = [
        {"precision": 1.0, "recall": 1.0, "exact": True, "answer_correct": True, "cost_usd": 0.001},
        {
            "precision": None,
            "recall": None,
            "exact": True,
            "answer_correct": None,
            "cost_usd": 0.002,
        },
    ]
    out = aggregate(rows, resamples=200)
    assert out["tool_precision"]["n"] == 1
    assert out["tool_exact"]["mean"] == 1.0 and out["tool_exact"]["n"] == 2
    assert out["cost_usd"]["mean"] == pytest.approx(0.0015)
    assert out["latency_s"] is None


class TestBuildSet:
    def test_specs_are_unique_and_use_known_tools(self) -> None:
        ids = [s["id"] for s in SPECS]
        assert len(ids) == len(set(ids)) == 60
        for s in SPECS:
            assert set(s["required"]) | set(s.get("optional", [])) <= set(build_set.TOOLS)
            assert (s["category"] == "none") == (not s["checks"])

    def test_build_resolves_gold_and_splits_by_category(self, tmp_path: Path) -> None:
        build_set.main(["--out", str(tmp_path)])
        rows = [json.loads(line) for line in (tmp_path / "queries.jsonl").read_text().splitlines()]
        assert len(rows) == 60
        assert sum(r["split"] == "dev" for r in rows) == 23
        by_id = {r["id"]: r for r in rows}
        assert by_id["sql_07"]["checks"][0]["values"] == [3]
        assert by_id["sql_17"]["checks"][0]["values"][0] == "ProBook X2"
        for category in {r["category"] for r in rows}:
            splits = {r["split"] for r in rows if r["category"] == category}
            assert splits == {"dev", "test"}
        sheet = (tmp_path / "review_sheet.md").read_text()
        assert "No model has been run" in sheet and "| sql_01 |" in sheet
        assert (tmp_path / "review.csv").read_text().startswith("id,split,category,query")
