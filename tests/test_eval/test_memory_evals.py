"""Tests for the memory evaluations: statistics, selection, both runners with a scripted
client (budget abort, error abort, no resume, provenance) and the CI smoke test."""

import json
from pathlib import Path
from typing import Any

import pytest

from evals.memory import run_followups, run_longterm, smoke
from evals.memory.common import gap_closed
from evals.memory.run_followups import FollowUp, load_followups, pair_questions
from evals.memory.run_followups import run_followups as fu
from evals.memory.run_longterm import (
    run_longterm as lt,
)
from evals.memory.run_longterm import (
    select_longterm_questions,
    teaching_text,
)
from evals.text2sql.data import Question, load_split
from src.llm import LLMConfig, Pricing, UsageTracker

CONFIG = LLMConfig(provider="ollama", model="scripted", pricing=Pricing())


def question(qid: int, db: str = "d", difficulty: str = "simple", evidence: str = "e") -> Question:
    return Question(
        question_id=qid,
        db_id=db,
        question=f"q{qid}",
        evidence=evidence,
        SQL="SELECT 1",
        difficulty=difficulty,  # type: ignore[arg-type]
    )


class TestGapClosed:
    def test_point_estimate_and_interval(self) -> None:
        ids = range(40)
        no = {i: i < 10 for i in ids}  # 25%
        oracle = {i: i < 30 for i in ids}  # 75%
        memory = {i: i < 20 for i in ids}  # 50%
        out = gap_closed(no, memory, oracle, resamples=500, seed=1)
        assert out is not None and out["n"] == 40
        assert out["gap_closed"] == pytest.approx(0.5)
        assert out["ci_low"] < 0.5 < out["ci_high"]
        assert out["frac_undefined"] == 0.0

    def test_zero_denominator_is_undefined_not_an_error(self) -> None:
        same = {i: i < 5 for i in range(10)}
        out = gap_closed(same, same, same, resamples=100, seed=0)
        assert out is not None
        assert out["gap_closed"] is None and out["ci_low"] is None and out["ci_high"] is None
        assert out["frac_undefined"] == 1.0

    def test_some_resamples_undefined(self) -> None:
        # one question carries the whole gain, so resamples that miss it have no denominator
        no = {i: False for i in range(5)}
        oracle = {**no, 0: True}
        memory = {**no, 0: True}
        out = gap_closed(no, memory, oracle, resamples=2000, seed=0)
        assert out is not None and out["gap_closed"] == 1.0
        assert 0.2 < out["frac_undefined"] < 0.5  # (4/5) ** 5 = 0.33
        assert out["ci_low"] == out["ci_high"] == 1.0

    def test_can_exceed_one_or_go_negative(self) -> None:
        no = {0: False, 1: False, 2: True}
        oracle = {0: True, 1: False, 2: True}
        memory = {0: True, 1: True, 2: True}
        assert gap_closed(no, memory, oracle, resamples=10)["gap_closed"] == 2.0  # type: ignore[index]
        memory = {0: False, 1: False, 2: False}
        assert gap_closed(no, memory, oracle, resamples=10)["gap_closed"] == -1.0  # type: ignore[index]

    def test_deterministic_and_uses_shared_ids_only(self) -> None:
        no = {1: False, 2: False, 3: True}
        memory = {1: True, 2: False, 9: True}
        oracle = {1: True, 2: True, 3: True}
        a = gap_closed(no, memory, oracle, resamples=50, seed=3)
        assert a == gap_closed(no, memory, oracle, resamples=50, seed=3)
        assert a is not None and a["n"] == 2
        assert gap_closed({1: True}, {2: True}, {3: True}) is None


class TestSelection:
    def test_only_questions_with_evidence_of_the_split(self) -> None:
        qs = [question(1), question(2, evidence="  "), question(3), question(4)]
        chosen = select_longterm_questions(qs, [1, 2, 3], None, 0)
        assert [q.question_id for q in chosen] == [1, 3]

    def test_limit_is_a_seeded_stratified_sample(self) -> None:
        qs = [question(i, db="a" if i < 20 else "b") for i in range(1, 41)]
        a = select_longterm_questions(qs, [q.question_id for q in qs], 10, 5)
        b = select_longterm_questions(qs, [q.question_id for q in qs], 10, 5)
        assert [q.question_id for q in a] == [q.question_id for q in b] and len(a) == 10
        assert {q.db_id for q in a} == {"a", "b"}

    def test_teaching_text(self) -> None:
        assert teaching_text("db", " x means y ") == (
            "For my questions about the db database: x means y"
        )


class TestFollowupData:
    def test_the_committed_file_is_well_formed_and_comes_from_dev(self) -> None:
        followups = load_followups()
        assert len(followups) == 30
        assert len({f.id for f in followups}) == 30
        assert len({f.source_question_id for f in followups}) == 30
        assert set(f.source_question_id for f in followups) <= set(load_split("dev"))
        assert all(f.turn1.strip() and f.turn2.strip() for f in followups)
        assert all(f.turn1 != f.turn2 for f in followups)

    @pytest.mark.requires_bird
    def test_every_pair_matches_its_source_question(self) -> None:
        from evals.text2sql.data import load_questions

        pairs = pair_questions(load_followups(), load_questions(), load_split("dev"))
        assert len(pairs) == 30
        assert all(source.db_id == f.db_id for f, source in pairs)

    def test_pairing_checks(self) -> None:
        follow = FollowUp(
            id="a", source_question_id=1, db_id="d", difficulty="simple", turn1="x", turn2="y"
        )
        qs = [question(1), question(2)]
        assert pair_questions([follow], qs, [1, 2])[0][1].question_id == 1
        with pytest.raises(ValueError, match="not in dev"):
            pair_questions([follow], qs, [2])
        with pytest.raises(ValueError, match="not found"):
            pair_questions([follow.model_copy(update={"source_question_id": 9})], qs, [1])
        with pytest.raises(ValueError, match="database"):
            pair_questions([follow.model_copy(update={"db_id": "other"})], qs, [1])
        with pytest.raises(ValueError, match="duplicate"):
            pair_questions([follow, follow], qs, [1])
        other = follow.model_copy(update={"id": "b"})
        with pytest.raises(ValueError, match="share a source"):
            pair_questions([follow, other], qs, [1])


def scripted(**overrides: Any) -> Any:
    """Client factory around the smoke test's scripted client; ``overrides`` replace methods."""

    def factory(config: LLMConfig, tracker: UsageTracker) -> Any:
        client = smoke.ScriptedClient(tracker)
        for name, fn in overrides.items():
            setattr(client, name, fn)
        return client

    return factory


@pytest.fixture
def fixture(tmp_path: Path) -> tuple[Path, list[Question]]:
    return smoke.build_fixture(tmp_path / "fixture")


def longterm(fixture: tuple[Path, list[Question]], run_dir: Path, **kw: Any) -> dict[str, Any]:
    databases_dir, questions = fixture
    return lt(
        config=CONFIG,
        questions=[q for q in questions if q.question_id <= 4],
        run_dir=run_dir,
        run_id="t",
        split="dev",
        databases_dir=databases_dir,
        skip_respond=True,
        bootstrap_resamples=100,
        **{"client_factory": scripted(), **kw},
    )


def followups(fixture: tuple[Path, list[Question]], run_dir: Path, **kw: Any) -> dict[str, Any]:
    databases_dir, questions = fixture
    by_id = {q.question_id: q for q in questions}
    return fu(
        config=CONFIG,
        pairs=[(f, by_id[f.source_question_id]) for f in smoke.FOLLOWUPS],
        run_dir=run_dir,
        run_id="t",
        databases_dir=databases_dir,
        skip_respond=True,
        bootstrap_resamples=100,
        **{"client_factory": scripted(), **kw},
    )


class TestLongtermRunner:
    def test_smoke_numbers(self, fixture: tuple[Path, list[Question]], tmp_path: Path) -> None:
        summary = longterm(fixture, tmp_path / "run")
        assert smoke.verify_longterm(summary, tmp_path / "run") == []

    def test_layout_and_no_gold_sql(
        self, fixture: tuple[Path, list[Question]], tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "run"
        longterm(fixture, run_dir)
        assert {p.name for p in run_dir.iterdir()} >= {
            "predictions.jsonl",
            "teaching.jsonl",
            "llm_calls.jsonl",
            "summary.json",
            "run.json",
        }
        predictions = [
            json.loads(x) for x in (run_dir / "predictions.jsonl").read_text().split("\n") if x
        ]
        assert len(predictions) == 12
        assert all("gold_sql" not in p for p in predictions)
        assert {p["system"] for p in predictions} == {"no_memory", "memory", "oracle"}
        memory = [p for p in predictions if p["system"] == "memory"]
        assert all(
            {"recalled_keys", "retrieval_hit", "own_memory_stored"} <= set(p) for p in memory
        )
        calls = [json.loads(x) for x in (run_dir / "llm_calls.jsonl").read_text().split("\n") if x]
        assert {c["system"] for c in calls} == {"teaching", "no_memory", "memory", "oracle"}
        assert all(c["run_id"] == "t" for c in calls)

    def test_conditions_see_the_right_prompts(
        self, fixture: tuple[Path, list[Question]], tmp_path: Path
    ) -> None:
        seen: dict[str, list[str]] = {}

        class Spy(smoke.ScriptedClient):
            def complete(self, system: Any, user: str, *, tag: str, prefix: Any = None) -> Any:
                seen.setdefault(tag, []).append(user)
                return super().complete(system, user, tag=tag, prefix=prefix)

        longterm(fixture, tmp_path / "run", client_factory=lambda c, t: Spy(t))
        generate = seen["generate"]  # 4 questions per condition, in condition order
        no_memory, memory, oracle = generate[:4], generate[4:8], generate[8:]
        assert all(
            "External Knowledge" not in p and "Definitions this user" not in p for p in no_memory
        )
        assert all("External Knowledge" not in p and "Definitions this user" in p for p in memory)
        assert all("External Knowledge" in p and "Definitions this user" not in p for p in oracle)
        assert all("Earlier in this conversation" not in p for p in generate)
        assert all(t.startswith("Text written by the user:") for t in seen["memorize"])
        assert all("For my questions about the toy database:" in t for t in seen["memorize"])

    def test_budget_abort_saves_partial_results(
        self, fixture: tuple[Path, list[Question]], tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "run"
        summary = longterm(fixture, run_dir, budget_usd=smoke.COST_PER_CALL * 6.5)
        assert summary["aborted"] == "budget"
        assert (run_dir / "summary.json").exists()
        run = json.loads((run_dir / "run.json").read_text())
        assert run["status"] == "aborted:budget" and run["llm_config"]["budget_usd"] > 0
        assert summary["systems"]["oracle"]["n"] == 0
        assert summary["systems"]["no_memory"]["aborted"] == "budget"
        assert summary["tracker"]["cost_usd_this_invocation"] < smoke.COST_PER_CALL * 8

    def test_repeated_exceptions_abort(
        self, fixture: tuple[Path, list[Question]], tmp_path: Path
    ) -> None:
        def broken(self: Any, system: Any, user: str, *, tag: str, prefix: Any = None) -> Any:
            if tag == "generate":
                raise RuntimeError("boom")
            return smoke.ScriptedClient.complete(self, system, user, tag=tag, prefix=prefix)

        class Broken(smoke.ScriptedClient):
            complete = broken  # type: ignore[assignment]

        summary = longterm(fixture, tmp_path / "run", client_factory=lambda c, t: Broken(t))
        assert summary["aborted"] == "errors"
        assert summary["systems"]["no_memory"]["failures"] == {"exception": 3}
        run = json.loads((tmp_path / "run" / "run.json").read_text())
        assert run["status"] == "aborted:errors"

    def test_existing_run_dir_is_refused(
        self, fixture: tuple[Path, list[Question]], tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "run"
        longterm(fixture, run_dir)
        before = (run_dir / "predictions.jsonl").read_text()
        with pytest.raises(FileExistsError, match="no --resume"):
            longterm(fixture, run_dir)
        assert (run_dir / "predictions.jsonl").read_text() == before

    def test_provenance_before_the_first_case(
        self, fixture: tuple[Path, list[Question]], tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "run"
        seen: dict[str, Any] = {}

        class Peek(smoke.ScriptedClient):
            def complete(self, system: Any, user: str, *, tag: str, prefix: Any = None) -> Any:
                if "run" not in seen:
                    seen["run"] = json.loads((run_dir / "run.json").read_text())
                return super().complete(system, user, tag=tag, prefix=prefix)

        longterm(
            fixture,
            run_dir,
            client_factory=lambda c, t: Peek(t),
            launched_by="unit test",
            argv=["x", "--run-id", "t"],
        )
        assert seen["run"]["status"] == "running" and seen["run"]["launched_by"] == "unit test"
        assert seen["run"]["argv"] == ["x", "--run-id", "t"]
        assert json.loads((run_dir / "run.json").read_text())["status"] == "completed"

    def test_each_database_has_its_own_store(
        self, fixture: tuple[Path, list[Question]], tmp_path: Path
    ) -> None:
        import shutil

        databases_dir, questions = fixture
        shutil.copytree(databases_dir / "toy", databases_dir / "toy2")
        (databases_dir / "toy2" / "toy.sqlite").rename(databases_dir / "toy2" / "toy2.sqlite")
        other = question(11, db="toy2", evidence="gadget refers to v = 10")
        other = other.model_copy(update={"question": "[Q11] How many gadget rows are there?"})
        memory_prompts: list[str] = []

        class Spy(smoke.ScriptedClient):
            def complete(self, system: Any, user: str, *, tag: str, prefix: Any = None) -> Any:
                if tag == "generate" and "Definitions this user" in user:
                    memory_prompts.append(user)
                return super().complete(system, user, tag=tag, prefix=prefix)

        lt(
            config=CONFIG,
            questions=[*(q for q in questions if q.question_id <= 2), other],
            run_dir=tmp_path / "run",
            run_id="t",
            split="dev",
            databases_dir=databases_dir,
            skip_respond=True,
            bootstrap_resamples=50,
            client_factory=lambda c, t: Spy(t),
        )
        gadget = [p for p in memory_prompts if "[Q11]" in p]
        toy = [p for p in memory_prompts if "[Q11]" not in p]
        assert len(gadget) == 1 and len(toy) == 2
        assert "gadget" in gadget[0] and "bigshots" not in gadget[0] and "tiny" not in gadget[0]
        assert all("gadget" not in p for p in toy)


class TestFollowupRunner:
    def test_smoke_numbers(self, fixture: tuple[Path, list[Question]], tmp_path: Path) -> None:
        summary = followups(fixture, tmp_path / "run")
        assert smoke.verify_followups(summary, tmp_path / "run") == []
        assert any("n = 3" in n for n in summary["notes"])

    def test_history_only_in_the_with_history_condition(
        self, fixture: tuple[Path, list[Question]], tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "run"
        followups(fixture, run_dir)
        rows = [json.loads(x) for x in (run_dir / "predictions.jsonl").read_text().split("\n") if x]
        by_system = {
            s: [r for r in rows if r["system"] == s] for s in smoke.EXPECTED_FOLLOWUPS["correct"]
        }
        assert all(r["n_turns"] == 2 and r["turn1_sql"] for r in by_system["with_history"])
        assert all(r["n_turns"] == 1 for r in by_system["turn2_alone"] + by_system["original"])
        for r in by_system["with_history"]:  # turn 1 counts toward cost and calls
            assert r["n_llm_calls"] == 4 and r["turn1_llm_calls"] == 2
            assert r["cost_usd"] == pytest.approx(4 * smoke.COST_PER_CALL)
            assert r["turn1_cost_usd"] == pytest.approx(2 * smoke.COST_PER_CALL)
        assert all("gold_sql" not in r for r in rows)

    def test_prompts_per_condition(
        self, fixture: tuple[Path, list[Question]], tmp_path: Path
    ) -> None:
        seen: list[str] = []

        class Spy(smoke.ScriptedClient):
            def complete(self, system: Any, user: str, *, tag: str, prefix: Any = None) -> Any:
                if tag == "generate":
                    seen.append(user)
                return super().complete(system, user, tag=tag, prefix=prefix)

        followups(fixture, tmp_path / "run", client_factory=lambda c, t: Spy(t))
        alone, history, original = seen[:3], seen[3:9], seen[9:]
        assert all("Earlier in this conversation" not in p for p in alone + original)
        turn1, turn2 = history[0::2], history[1::2]
        assert all("Earlier in this conversation" not in p for p in turn1)
        assert all("Earlier in this conversation" in p for p in turn2)
        assert all("External Knowledge" in p for p in seen[:3] + turn1 + turn2 + original)

    def test_threads_are_separate_per_pair(
        self, fixture: tuple[Path, list[Question]], tmp_path: Path
    ) -> None:
        seen: list[str] = []

        class Spy(smoke.ScriptedClient):
            def complete(self, system: Any, user: str, *, tag: str, prefix: Any = None) -> Any:
                if tag == "generate":
                    seen.append(user)
                return super().complete(system, user, tag=tag, prefix=prefix)

        followups(fixture, tmp_path / "run", client_factory=lambda c, t: Spy(t))
        turn2_prompts = seen[3:9][1::2]
        for prompt, marker in zip(turn2_prompts, ("[Q5]", "[Q6]", "[Q7]"), strict=True):
            assert marker in prompt
            assert all(other not in prompt for other in {"[Q5]", "[Q6]", "[Q7]"} - {marker})

    def test_budget_abort_and_no_resume(
        self, fixture: tuple[Path, list[Question]], tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "run"
        summary = followups(fixture, run_dir, budget_usd=smoke.COST_PER_CALL * 5)
        assert summary["aborted"] == "budget"
        assert json.loads((run_dir / "run.json").read_text())["status"] == "aborted:budget"
        with pytest.raises(FileExistsError):
            followups(fixture, run_dir)


class TestCommandLine:
    def test_run_id_is_mandatory(self) -> None:
        for module in (run_longterm, run_followups):
            with pytest.raises(SystemExit):
                module.parse_args(["--llm-config", "x.yaml"])

    def test_anthropic_needs_a_budget(self, tmp_path: Path) -> None:
        config = tmp_path / "c.yaml"
        config.write_text("provider: anthropic\nmodel: m\nmax_tokens: 10\n")
        for module in (run_longterm, run_followups):
            with pytest.raises(SystemExit, match="budget"):
                module.main(["--llm-config", str(config), "--run-id", "x"])

    def test_test_split_prints_a_warning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = tmp_path / "c.yaml"
        config.write_text("provider: ollama\nmodel: m\n")
        monkeypatch.setattr(run_longterm, "load_questions", lambda: [])
        monkeypatch.setattr(run_longterm, "load_split", lambda name: [])
        monkeypatch.setattr(run_longterm, "RESULTS_ROOT", tmp_path / "results")
        monkeypatch.setattr(run_longterm, "run_longterm", lambda **kw: {})
        monkeypatch.setattr(run_longterm, "format_report", lambda s: "report")
        run_longterm.main(["--llm-config", str(config), "--run-id", "x", "--split", "test"])
        assert "test split is for the single final run" in capsys.readouterr().err
        run_longterm.main(["--llm-config", str(config), "--run-id", "y", "--split", "dev"])
        assert "single final run" not in capsys.readouterr().err

    def test_existing_run_id_exits_cleanly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = tmp_path / "c.yaml"
        config.write_text("provider: ollama\nmodel: m\n")
        (tmp_path / "results" / "x").mkdir(parents=True)
        (tmp_path / "results" / "x" / "run.json").write_text("{}")
        monkeypatch.setattr(run_longterm, "RESULTS_ROOT", tmp_path / "results")
        monkeypatch.setattr(run_longterm, "load_questions", lambda: [])
        monkeypatch.setattr(run_longterm, "load_split", lambda name: [])
        with pytest.raises(SystemExit, match="exists"):
            run_longterm.main(["--llm-config", str(config), "--run-id", "x"])


class TestSmoke:
    def test_passes(self, tmp_path: Path) -> None:
        assert smoke.run_all(tmp_path / "out") == []

    def test_main_exits_nonzero_on_a_wrong_number(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(smoke.EXPECTED_LONGTERM["correct"], "memory", 4)  # type: ignore[arg-type]
        with pytest.raises(SystemExit) as raised:
            smoke.main(["--out", str(tmp_path / "out")])
        assert raised.value.code == 1

    def test_main_passes(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        smoke.main(["--out", str(tmp_path / "out")])
        assert "SMOKE PASSED" in capsys.readouterr().out


class TestFollowupEvidenceMode:
    def _generate_prompts(
        self, fixture: tuple[Path, list[Question]], run_dir: Path, **kw: Any
    ) -> tuple[list[str], dict[str, Any]]:
        seen: list[str] = []

        class Spy(smoke.ScriptedClient):
            def complete(self, system: Any, user: str, *, tag: str, prefix: Any = None) -> Any:
                if tag == "generate":
                    seen.append(user)
                return super().complete(system, user, tag=tag, prefix=prefix)

        summary = followups(fixture, run_dir, client_factory=lambda c, t: Spy(t), **kw)
        return seen, summary

    def test_none_removes_the_evidence_from_every_condition(
        self, fixture: tuple[Path, list[Question]], tmp_path: Path
    ) -> None:
        seen, summary = self._generate_prompts(fixture, tmp_path / "run", evidence="none")
        assert seen and not any("External Knowledge" in u for u in seen)
        assert summary["evidence"] == "none"
        assert any("No condition gets any evidence" in n for n in summary["notes"])

    def test_source_is_the_default_and_keeps_the_evidence(
        self, fixture: tuple[Path, list[Question]], tmp_path: Path
    ) -> None:
        _, questions = fixture
        seen, summary = self._generate_prompts(fixture, tmp_path / "run")
        assert summary["evidence"] == "source"
        sources = {f.source_question_id for f in smoke.FOLLOWUPS}
        if any(q.evidence.strip() for q in questions if q.question_id in sources):
            assert any("External Knowledge" in u for u in seen)

    def test_unknown_mode_is_refused(
        self, fixture: tuple[Path, list[Question]], tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError):
            followups(fixture, tmp_path / "run", evidence="some")
