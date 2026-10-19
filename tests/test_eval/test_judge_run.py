"""Judge runner and label import, with scripted clients."""

import json
from pathlib import Path
from typing import Any

import pytest

from evals.judge import import_labels, run_judge
from evals.judge.import_labels import validate_labels
from src.llm import CallUsage, Completion, LLMConfig, Pricing, UsageTracker

CONFIG = LLMConfig(provider="ollama", model="scripted", pricing=Pricing())
COST = 0.002


def preview(rows: list[list[str]] | None = None, error: str | None = None) -> dict[str, Any]:
    p: dict[str, Any] = {"columns": ["v"], "rows": rows or [["10"]], "row_count": "1"}
    if error:
        p["error"] = error
    return p


def item(n: int, **kw: Any) -> dict[str, Any]:
    return {
        "item_id": f"j-{n:02d}",
        "question_id": 100 + n,
        "question": f"Q{n}?",
        "evidence": "",
        "answer": f"A{n}",
        "gold_preview": preview(),
        "ex_correct": True,
        **kw,
    }


class Judging:
    """Replies per answer text: A1 correct, A2 incorrect, A3 garbage, A4 raises."""

    def __init__(self, tracker: UsageTracker) -> None:
        self.tracker = tracker
        self.users: list[str] = []

    def complete(self, system: Any, user: str, *, tag: str, prefix: Any = None) -> Completion:
        self.tracker.check()
        self.users.append(user)
        assert tag == "judge"
        if "A4" in user:
            raise RuntimeError("boom")
        text = {
            "A1": '{"rationale": "same", "label": "correct"}',
            "A2": '{"rationale": "differs", "label": "incorrect"}',
        }.get(user.split("Answer:\n")[1].strip(), "no idea")
        usage = CallUsage(tag=tag, input_tokens=50, output_tokens=10, cost_usd=COST)
        usage.stop_reason = "end_turn"
        self.tracker.record(usage)
        return Completion(text=text, usage=usage)


def run(tmp_path: Path, items: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
    return run_judge.run_judge(
        config=CONFIG,
        items=items,
        items_run="set",
        run_dir=tmp_path / "run",
        run_id="t",
        client_factory=lambda c, t: Judging(t),
        launched_by="test",
        **kw,
    )


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x]


class TestRunJudge:
    def test_labels_unparsed_and_no_reference(self, tmp_path: Path) -> None:
        items = [
            item(1),
            item(2),
            item(3),
            item(5, gold_preview=preview(error="timeout")),
        ]
        summary = run(tmp_path, items)
        out = read(tmp_path / "run" / "judgements.jsonl")
        assert [(j["item_id"], j["label"]) for j in out] == [
            ("j-01", "correct"),
            ("j-02", "incorrect"),
            ("j-03", "unparsed"),
            ("j-05", "no_reference"),
        ]
        assert out[0]["rationale"] == "same" and out[0]["input_tokens"] == 50
        assert out[0]["cost_usd"] == pytest.approx(COST) and out[3]["cost_usd"] == 0
        assert summary["n_unparsed"] == 1 and summary["n_no_reference"] == 1
        assert summary["totals"]["cost_usd"] == pytest.approx(3 * COST)
        calls = read(tmp_path / "run" / "llm_calls.jsonl")
        assert len(calls) == 3 and all(c["tag"] == "judge" for c in calls)
        run_json = json.loads((tmp_path / "run" / "run.json").read_text())
        assert run_json["status"] == "completed" and run_json["launched_by"] == "test"

    def test_an_api_failure_is_an_error_label_not_a_crash(self, tmp_path: Path) -> None:
        summary = run(tmp_path, [item(1), item(4)])
        out = read(tmp_path / "run" / "judgements.jsonl")
        assert [j["label"] for j in out] == ["correct", "error"]
        assert "boom" in out[1]["rationale"] and summary["n_errors"] == 1

    def test_budget_abort_saves_what_was_judged(self, tmp_path: Path) -> None:
        summary = run(tmp_path, [item(1), item(2), item(1 + 10), item(12)], budget_usd=COST * 1.5)
        assert summary["aborted"] == "budget" and summary["n_judged"] == 2
        assert json.loads((tmp_path / "run" / "run.json").read_text())["status"] == "aborted:budget"

    def test_the_judge_never_sees_the_sql_or_ex(self, tmp_path: Path) -> None:
        seen: list[str] = []

        class Spy(Judging):
            def complete(self, *a: Any, **k: Any) -> Completion:
                seen.append(a[1])
                return super().complete(*a, **k)

        run_judge.run_judge(
            config=CONFIG,
            items=[item(1, final_sql="SELECT secret FROM t", ex_correct=False)],
            items_run="set",
            run_dir=tmp_path / "run",
            run_id="t",
            client_factory=lambda c, t: Spy(t),
        )
        assert seen and "secret" not in seen[0] and "SELECT" not in seen[0]

    def test_new_run_dir_is_required(self, tmp_path: Path) -> None:
        run(tmp_path, [item(1)])
        with pytest.raises(FileExistsError):
            run(tmp_path, [item(1)])


IDS = ["j-01", "j-02", "j-03"]


def labels(**entries: Any) -> dict[str, Any]:
    return {"labeler": "ed", "labels": entries}


class TestImportLabels:
    def good(self) -> dict[str, Any]:
        return labels(
            **{
                "j-01": {"label": "correct", "note": "ok"},
                "j-02": {"label": "incorrect"},
                "j-03": {"label": "unsure", "note": "ambiguous"},
            }
        )

    def test_valid_file(self) -> None:
        records = validate_labels(self.good(), IDS)
        assert records[0] == {"item_id": "j-01", "label": "correct", "note": "ok", "labeler": "ed"}
        assert [r["label"] for r in records] == ["correct", "incorrect", "unsure"]
        assert records[1]["note"] == ""

    @pytest.mark.parametrize(
        ("mutate", "fragment"),
        [
            (lambda d: d["labels"].update({"j-99": {"label": "correct"}}), "unknown item id"),
            (lambda d: d["labels"].update({"j-01": {"label": "yes"}}), "not one of"),
            (lambda d: d["labels"].update({"j-01": {"label": "correct", "note": 3}}), "note"),
            (lambda d: d["labels"].update({"j-01": "correct"}), "must be an object"),
            (lambda d: d["labels"].pop("j-03"), "1 items without a label: j-03"),
            (lambda d: d.update(labeler=" "), "labeler"),
        ],
    )
    def test_problems(self, mutate: Any, fragment: str) -> None:
        data = self.good()
        mutate(data)
        with pytest.raises(ValueError, match=fragment):
            validate_labels(data, IDS)

    def test_partial_is_allowed_on_request(self) -> None:
        data = self.good()
        del data["labels"]["j-03"]
        assert len(validate_labels(data, IDS, allow_partial=True)) == 2

    def test_not_an_object(self) -> None:
        with pytest.raises(ValueError, match="JSON object"):
            validate_labels([], IDS)
        with pytest.raises(ValueError, match="'labels'"):
            validate_labels({"labeler": "ed"}, IDS)

    def test_main_writes_labels_jsonl_and_refuses_overwrite(self, tmp_path: Path) -> None:
        items_dir = tmp_path / "items"
        items_dir.mkdir()
        (items_dir / "items.jsonl").write_text(
            "".join(json.dumps({"item_id": i}) + "\n" for i in IDS)
        )
        path = tmp_path / "labels.json"
        path.write_text(json.dumps(self.good()))
        args = ["--items", str(items_dir), "--labels", str(path)]
        import_labels.main(args)
        lines = read(items_dir / "labels.jsonl")
        assert [x["item_id"] for x in lines] == IDS and lines[0]["labeler"] == "ed"
        with pytest.raises(SystemExit, match="exists"):
            import_labels.main(args)
        import_labels.main([*args, "--force"])
        path.write_text("{not json")
        with pytest.raises(SystemExit, match="labels.json"):
            import_labels.main([*args, "--force"])
