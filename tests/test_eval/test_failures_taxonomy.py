"""Failure rules, sampling, review sheet and review import, on synthetic records."""

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from evals.failures import classify, review, taxonomy

T = taxonomy


def bird(**kw: Any) -> dict[str, Any]:
    sql = kw.pop("pred_sql", "SELECT 1")
    rec: dict[str, Any] = {
        "correct": False,
        "pred_sql": sql,
        "attempts": [{"sql": sql, "error": None, "row_count": "1", "n_columns": 1}],
        "fixes_used": 0,
        "answer": "It is 1.",
        "judge_label": "incorrect",
    }
    rec.update(kw)
    return rec


SAME = {"pred_columns": 1, "gold_columns": 1, "pred_rows": 1, "gold_rows": 1}


class TestBirdRules:
    def test_failure_definition(self) -> None:
        assert T.bird_is_failure(bird(correct=False, judge_label="correct"))
        assert T.bird_is_failure(bird(correct=True, judge_label="incorrect"))
        assert not T.bird_is_failure(bird(correct=True, judge_label="correct"))

    def test_logic_vs_shape(self) -> None:
        assert T.bird_tags(bird(), SAME) == [T.WRONG_QUERY_LOGIC]
        wide = {**SAME, "pred_columns": 3}
        assert T.bird_tags(bird(), wide) == [T.WRONG_OUTPUT_SHAPE]
        assert T.bird_tags(bird(), None) == []  # shape unknown: no logic or shape tag

    def test_ex_right_answer_wrong_split_by_result_size(self) -> None:
        small = bird(correct=True)
        assert T.bird_tags(small) == [T.ANSWER_MISREAD]
        big = bird(correct=True)
        big["attempts"][0]["row_count"] = ">1000"
        assert T.bird_tags(big) == [T.ANSWER_ON_LARGE_RESULT]
        edge = bird(correct=True)
        edge["attempts"][0]["row_count"] = str(T.ANSWER_PREVIEW_ROWS)
        assert T.bird_tags(edge) == [T.ANSWER_MISREAD]

    def test_answer_ok_ex_wrong_keeps_shape_tag(self) -> None:
        tags = T.bird_tags(bird(judge_label="correct"), {**SAME, "pred_columns": 2})
        assert tags == [T.ANSWER_OK_EX_WRONG, T.WRONG_OUTPUT_SHAPE]
        assert T.primary(tags) == T.ANSWER_OK_EX_WRONG

    def test_errors_and_silence(self) -> None:
        def errored(error: str, answer: str) -> list[str]:
            rec = bird(answer=answer)
            rec["attempts"][0]["error"] = error
            return T.bird_tags(rec)

        assert errored("no such column: x", "It is 5.") == [
            T.SCHEMA_HALLUCINATION,
            T.SILENT_TOOL_FAILURE,
        ]
        assert errored("near SELECT: syntax error", "The query failed.") == [T.BAD_ARGUMENTS]
        assert errored("blocked: write statement", "I could not run it.") == [T.GUARD_REFUSAL]

    def test_loop(self) -> None:
        rec = bird(fixes_used=2)
        assert T.LOOP in T.bird_tags(rec, SAME)
        again = bird()
        again["attempts"] = [
            {"sql": "SELECT  1", "error": None, "row_count": "1"},
            {"sql": "SELECT 1", "error": None, "row_count": "1"},
        ]
        assert T.primary(T.bird_tags(again, SAME)) == T.LOOP


def step(tool: str, inp: str, ok: bool = True, out: str = "{}", err: str | None = None) -> Any:
    return {"tool": tool, "input": inp, "success": ok, "output": out, "error": err}


def tool_rec(**kw: Any) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "answer_correct": True,
        "exact": True,
        "wrong_tools": [],
        "missing_tools": [],
        "steps": [],
        "response": "Fine.",
    }
    rec.update(kw)
    return rec


NUM15 = [{"kind": "number", "values": [15]}]


class TestToolRules:
    def test_placeholder_is_bad_arguments_and_recovery_is_not_silent(self) -> None:
        steps = [
            step("sql_query", "SELECT 1", out='{"rows": [[129]], "row_count": 1}'),
            step("calculator", "PLACEHOLDER", ok=False, err="Unsupported expression element"),
        ]
        said = tool_rec(steps=steps, response="The calculator failed, so I did it by hand.")
        assert T.tools_tags(said) == [T.BAD_ARGUMENTS]
        quiet = tool_rec(steps=steps, response="It is 148.35.")
        assert T.tools_tags(quiet) == [T.BAD_ARGUMENTS, T.SILENT_TOOL_FAILURE]
        assert not T.tools_is_failure(said)

    def test_empty_result_counts_as_failed_step(self) -> None:
        rec = tool_rec(steps=[step("sql_query", "SELECT", out='{"rows": [], "row_count": 0}')])
        assert T.tools_tags(rec) == [T.SILENT_TOOL_FAILURE]

    def test_wrong_tool_and_repeated_call(self) -> None:
        rec = tool_rec(
            exact=False,
            missing_tools=["doc_search"],
            steps=[step("sql_query", "SELECT 1"), step("sql_query", "SELECT  1")],
        )
        assert T.tools_tags(rec) == [T.WRONG_TOOL, T.LOOP]

    def test_misread_cut_and_miss(self) -> None:
        def tags(output: str) -> list[str]:
            rec = tool_rec(answer_correct=False, steps=[step("doc_search", "q", out=output)])
            return T.tools_tags(rec, NUM15)

        assert tags("plazo de 15 días") == [T.ANSWER_MISREAD]
        late = "x" * T.AGENT_TOOL_OUTPUT_CHARS + " plazo de 15 días"
        assert tags(late) == [T.TOOL_OUTPUT_CUT]
        assert tags("plazo de 30 días") == [T.RETRIEVAL_MISS]

    def test_mentions_failure(self) -> None:
        assert T.mentions_failure("The calculator tool failed")
        assert T.mentions_failure("No se pudo calcular")
        assert not T.mentions_failure("It is 42.")


def failure(fid: str, category: str, item: Any = None, source: str = "bird") -> dict[str, Any]:
    return {
        "failure_id": fid,
        "source": source,
        "item": item if item is not None else fid,
        "primary": category,
        "tags": [category],
        "ex_correct": False,
        "judge_label": "incorrect",
        "shape": SAME,
        "_trace": {
            "question": "Q?",
            "evidence": "",
            "attempts": [{"sql": "SELECT 1"}],
            "verdicts": [],
            "fixes_used": 0,
            "final_sql": "SELECT 1",
            "answer": "A",
        },
        "_reference": {"columns": ["SECRET_GOLD"], "rows": [["42"]], "row_count": "1"},
    }


class TestSampleAndSheet:
    def test_stratified_then_topped_up(self) -> None:
        fs = [failure(f"a{i}", T.WRONG_QUERY_LOGIC) for i in range(20)]
        fs += [failure(f"b{i}", T.LOOP) for i in range(2)]
        fs += [failure(f"c{i}", T.ANSWER_MISREAD) for i in range(8)]
        sample = classify.sample_for_review(fs, n=14)
        got = [f["primary"] for f in sample]
        assert len(sample) == 14 and got.count(T.LOOP) == 2
        assert got.count(T.WRONG_QUERY_LOGIC) >= 5 and got.count(T.ANSWER_MISREAD) >= 5
        assert classify.sample_for_review(fs, n=14) == sample  # seeded

    def test_same_question_sampled_once_per_category(self) -> None:
        fs = [failure(f"t{i}", T.TOOL_OUTPUT_CUT, item="sqldoc_03") for i in range(4)]
        assert len(classify.sample_for_review(fs, n=10)) == 1

    def test_reference_only_in_private_sheet(self) -> None:
        sample = [failure("bird-haiku-1", T.WRONG_QUERY_LOGIC)]
        assert "SECRET_GOLD" not in classify.review_sheet(sample, private=False)
        assert "SECRET_GOLD" in classify.review_sheet(sample, private=True)

    def test_public_trace_drops_private_fields(self, tmp_path: Path) -> None:
        fs = [failure("bird-haiku-1", T.WRONG_QUERY_LOGIC)]
        classify.write_outputs(tmp_path / "out", fs, {"x": 1}, fs, tmp_path)
        files = [p.read_text() for p in (tmp_path / "out").rglob("*") if p.is_file()]
        assert not any("SECRET_GOLD" in text for text in files)
        trace = json.loads((tmp_path / "out" / "traces" / "bird-haiku-1.json").read_text())
        assert trace["trace"]["final_sql"] == "SELECT 1" and "_reference" not in trace
        assert "SECRET_GOLD" in (tmp_path / "review_private.md").read_text()


def template(tmp_path: Path, rows: list[list[str]]) -> Path:
    path = tmp_path / "t.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(classify.REVIEW_COLUMNS)
        w.writerows(rows)
    return path


class TestReview:
    TEMPLATE = [
        ["r-01", "f1", T.WRONG_QUERY_LOGIC, "", ""],
        ["r-02", "f2", T.ANSWER_MISREAD, "", ""],
        ["r-03", "f3", T.WRONG_QUERY_LOGIC, "", ""],
    ]

    def filled(self, humans: list[str]) -> list[dict[str, str]]:
        return [
            dict(zip(classify.REVIEW_COLUMNS, [*row[:3], h, ""], strict=True))
            for row, h in zip(self.TEMPLATE, humans, strict=True)
        ]

    def tmpl(self, tmp_path: Path) -> list[dict[str, str]]:
        return review.read_csv(template(tmp_path, self.TEMPLATE))

    def test_agreement(self, tmp_path: Path) -> None:
        rows = review.validate(
            self.tmpl(tmp_path),
            self.filled([T.WRONG_QUERY_LOGIC, T.ANSWER_MISREAD, T.BENCHMARK_ISSUE]),
        )
        out = review.agreement(rows)
        assert out["n_reviewed"] == 3
        assert out["rule_vs_human"]["accuracy"]["value"] == pytest.approx(2 / 3)
        assert out["per_rule_category"][T.WRONG_QUERY_LOGIC] == {"reviewed": 2, "agreed": 1}
        assert out["not_agent_failures_by_rule_category"] == {T.WRONG_QUERY_LOGIC: 1}

    def test_validation(self, tmp_path: Path) -> None:
        tmpl = self.tmpl(tmp_path)
        with pytest.raises(ValueError, match="without a human category"):
            review.validate(tmpl, self.filled([T.LOOP, "", T.LOOP]))
        assert len(review.validate(tmpl, self.filled([T.LOOP, "", T.LOOP]), allow_partial=True))
        with pytest.raises(ValueError, match="unknown category"):
            review.validate(tmpl, self.filled([T.LOOP, "nonsense", T.LOOP]))
        changed = self.filled([T.LOOP] * 3)
        changed[0]["rule_category"] = T.LOOP
        with pytest.raises(ValueError, match="rule_category changed"):
            review.validate(tmpl, changed)
