"""Short-term memory: the history block, per-turn reset and checkpointer threads."""

import sqlite3
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from src.llm import CallUsage, Completion, reset_client, set_client
from src.sql_agent.executor import Attempt
from src.sql_agent.graph import build_graph, run_sql_agent
from src.sql_agent.prompts import (
    HISTORY_HEADER,
    HISTORY_TURNS,
    check_user,
    history_block,
    question_block,
)
from src.sql_agent.state import SqlAgentResult, Turn


class ScriptedLLM:
    def __init__(self, **script: list[str]) -> None:
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[tuple[str | None, str, str, str | None]] = []

    def complete(
        self, system: str | None, user: str, *, tag: str, prefix: str | None = None
    ) -> Completion:
        self.calls.append((system, user, tag, prefix))
        queue = self.script.get(tag, [])
        text = queue.pop(0) if len(queue) > 1 else (queue[0] if queue else "")
        return Completion(text=text, usage=CallUsage(tag=tag))

    def of(self, tag: str) -> list[tuple[str | None, str, str, str | None]]:
        return [c for c in self.calls if c[2] == tag]


def sql(q: str) -> str:
    return f"```sql\n{q}\n```"


ALL = "SELECT name FROM city"
BIG = "SELECT name FROM city WHERE pop > 5"
BAD = "SELECT nope FROM city"


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "d.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE city (id INTEGER PRIMARY KEY, name TEXT, pop INTEGER)")
    conn.executemany("INSERT INTO city VALUES (?, ?, ?)", [(1, "Oslo", 5), (2, "Rome", 9)])
    conn.commit()
    conn.close()
    return path


def ask(
    llm: ScriptedLLM, db: Path, question: str, evidence: str = "", **kw: object
) -> SqlAgentResult:
    token = set_client(llm)
    try:
        return run_sql_agent(question, evidence, db, skip_respond=True, **kw)  # type: ignore[arg-type]
    finally:
        reset_client(token)


class TestHistoryBlock:
    def turns(self, n: int) -> list[Turn]:
        return [Turn(question=f"q{i}", final_sql=f"SELECT {i}") for i in range(n)]

    def test_empty_history_changes_nothing(self) -> None:
        assert history_block([]) == ""
        legacy = "-- External Knowledge: e\nQuestion: q"
        assert question_block("q", "e") == legacy == question_block("q", "e", [])
        attempt = Attempt(sql="SELECT 1", columns=["a"], row_count=1, preview=[["1"]])
        assert check_user("q", "e", attempt) == check_user("q", "e", attempt, [])
        assert question_block("q", "") == "Question: q"

    def test_block_lists_the_last_three_turns_with_one_line_sql(self) -> None:
        turns = self.turns(4)
        turns[3] = Turn(question=" q3 ", final_sql="SELECT 3\n  FROM t\n WHERE x = 1")
        block = history_block(turns)
        assert block.splitlines()[0] == HISTORY_HEADER == "Earlier in this conversation:"
        assert HISTORY_TURNS == 3
        assert "q0" not in block and "q1" in block and "q2" in block
        assert "- Question: q3\n  SQL: SELECT 3 FROM t WHERE x = 1" in block

    def test_failed_turn_has_a_placeholder(self) -> None:
        assert "SQL: (no query)" in history_block([Turn(question="q", final_sql="")])

    def test_block_precedes_evidence_and_question(self) -> None:
        text = question_block("the question", "the evidence", self.turns(1))
        order = [text.index(x) for x in (HISTORY_HEADER, "External Knowledge", "Question: the q")]
        assert order == sorted(order)
        assert text.endswith("Question: the question")

    def test_check_prompt_has_the_block_before_the_candidate(self) -> None:
        attempt = Attempt(sql="SELECT 1", columns=["a"], row_count=1, preview=[["1"]])
        text = check_user("q", "", attempt, self.turns(1))
        assert text.index(HISTORY_HEADER) < text.index("Candidate query:")


class TestThreads:
    def test_second_turn_sees_the_first(self, db: Path) -> None:
        llm = ScriptedLLM(generate=[sql(ALL), sql(BIG)], check=["OK"])
        saver = InMemorySaver()
        first = ask(llm, db, "Which cities?", thread_id="t1", checkpointer=saver)
        second = ask(
            llm, db, "Only the big ones?", "big is pop > 5", thread_id="t1", checkpointer=saver
        )
        gen1, gen2 = llm.of("generate")
        assert HISTORY_HEADER not in gen1[1]
        assert f"- Question: Which cities?\n  SQL: {ALL}" in gen2[1]
        assert gen2[1].index(HISTORY_HEADER) < gen2[1].index("External Knowledge")
        assert gen2[1].endswith("Question: Only the big ones?")
        check2 = llm.of("check")[1]
        assert f"SQL: {ALL}" in check2[1] and "Candidate query" in check2[1]
        assert gen1[3] == gen2[3] == check2[3]  # the cached prefix is identical
        assert [t.question for t in first.history] == ["Which cities?"]
        assert [t.question for t in second.history] == ["Which cities?", "Only the big ones?"]

    def test_turn_stores_shape_not_rows(self, db: Path) -> None:
        llm = ScriptedLLM(generate=[sql(ALL)], check=["OK"])
        result = ask(llm, db, "Which cities?", thread_id="t", checkpointer=InMemorySaver())
        (turn,) = result.history
        assert (turn.final_sql, turn.row_count, turn.columns) == (ALL, 2, ["name"])
        assert "Oslo" not in turn.model_dump_json()

    def test_per_turn_state_is_reset(self, db: Path) -> None:
        # turn 1: error, one fix; turn 2: right at once
        llm = ScriptedLLM(generate=[sql(BAD), sql(BIG)], check=[sql(ALL), "OK"])
        saver = InMemorySaver()
        first = ask(llm, db, "Which cities?", "e1", thread_id="t", checkpointer=saver)
        assert first.fixes_used == 1 and len(first.attempts) == 2 and len(first.verdicts) == 2
        second = ask(llm, db, "Big ones?", "", thread_id="t", checkpointer=saver)
        assert second.fixes_used == 0
        assert len(second.attempts) == 1 and second.final_sql == BIG
        assert [v.kind for v in second.verdicts] == ["ok"]
        assert len(second.history) == 2  # appended once per turn, not once per fix
        assert "e1" not in llm.of("generate")[1][1]

    def test_threads_do_not_share_history(self, db: Path) -> None:
        llm = ScriptedLLM(generate=[sql(ALL)], check=["OK"])
        saver = InMemorySaver()
        ask(llm, db, "Which cities?", thread_id="a", checkpointer=saver)
        other = ask(llm, db, "Another?", thread_id="b", checkpointer=saver)
        assert HISTORY_HEADER not in llm.of("generate")[1][1]
        assert len(other.history) == 1

    def test_without_a_thread_nothing_is_remembered(self, db: Path) -> None:
        llm = ScriptedLLM(generate=[sql(ALL)], check=["OK"])
        ask(llm, db, "Which cities?")
        result = ask(llm, db, "Again?")
        assert HISTORY_HEADER not in llm.of("generate")[1][1]
        assert len(result.history) == 1

    def test_only_the_last_three_turns_reach_the_prompt(self, db: Path) -> None:
        llm = ScriptedLLM(generate=[sql(ALL)], check=["OK"])
        saver = InMemorySaver()
        for i in range(5):
            result = ask(llm, db, f"question {i}?", thread_id="t", checkpointer=saver)
        prompt = llm.of("generate")[4][1]
        assert "Question: question 0?" not in prompt
        assert all(f"- Question: question {i}?" in prompt for i in (1, 2, 3))
        assert prompt.endswith("Question: question 4?")
        assert len(result.history) == 5

    def test_respond_prompt_is_unchanged_by_history(self, db: Path) -> None:
        llm = ScriptedLLM(generate=[sql(ALL)], check=["OK"], respond=["Two."])
        saver = InMemorySaver()
        token = set_client(llm)
        try:
            for q in ("Which cities?", "And again?"):
                run_sql_agent(q, "", db, thread_id="t", checkpointer=saver)
        finally:
            reset_client(token)
        for _, user, _, _ in llm.of("respond"):
            assert HISTORY_HEADER not in user and user.startswith("Question: ")

    def test_history_survives_a_new_sqlite_saver(self, db: Path, tmp_path: Path) -> None:
        path = tmp_path / "checkpoints.sqlite"
        llm = ScriptedLLM(generate=[sql(ALL)], check=["OK"])
        with SqliteSaver.from_conn_string(str(path)) as saver:
            ask(llm, db, "Which cities?", thread_id="t", checkpointer=saver)
        with SqliteSaver.from_conn_string(str(path)) as saver:
            result = ask(llm, db, "Again?", thread_id="t", checkpointer=saver)
        assert f"- Question: Which cities?\n  SQL: {ALL}" in llm.of("generate")[1][1]
        assert len(result.history) == 2

    def test_thread_and_checkpointer_go_together(self, db: Path) -> None:
        llm = ScriptedLLM(generate=[sql(ALL)], check=["OK"])
        with pytest.raises(ValueError):
            ask(llm, db, "q", thread_id="t")
        with pytest.raises(ValueError):
            ask(llm, db, "q", checkpointer=InMemorySaver())

    def test_default_graph_stays_cached_and_checkpointed_graphs_are_separate(self) -> None:
        assert build_graph() is build_graph()
        assert build_graph(InMemorySaver()) is not build_graph()
