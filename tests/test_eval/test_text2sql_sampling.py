"""Tests for stratified sampling."""

from collections import Counter

from evals.text2sql.data import Question
from evals.text2sql.sampling import allocate, stratified_allocation, stratified_sample


def make_questions() -> list[Question]:
    sizes = {
        ("a", "simple"): 20,
        ("a", "moderate"): 30,
        ("b", "simple"): 10,
        ("b", "challenging"): 40,
    }
    out: list[Question] = []
    qid = 0
    for (db, diff), n in sizes.items():
        for _ in range(n):
            qid += 1
            out.append(
                Question(question_id=qid, db_id=db, question="q", SQL="s", difficulty=diff)  # type: ignore[arg-type]
            )
    return out


def strata(questions: list[Question], ids: list[int]) -> Counter[tuple[str, str]]:
    by_id = {q.question_id: q for q in questions}
    return Counter((by_id[i].db_id, by_id[i].difficulty) for i in ids)


def test_deterministic_and_seed_matters() -> None:
    qs = make_questions()
    assert stratified_sample(qs, 20, 0) == stratified_sample(qs, 20, 0)
    assert stratified_sample(qs, 20, 0) != stratified_sample(qs, 20, 1)


def test_exact_n_unique_sorted() -> None:
    qs = make_questions()
    for n in (1, 7, 20, 33, 99):
        ids = stratified_sample(qs, n, 0)
        assert len(ids) == n == len(set(ids))
        assert ids == sorted(ids)


def test_proportional() -> None:
    qs = make_questions()
    counts = strata(qs, stratified_sample(qs, 20, 0))
    assert counts == {
        ("a", "simple"): 4,
        ("a", "moderate"): 6,
        ("b", "simple"): 2,
        ("b", "challenging"): 8,
    }


def test_every_allocated_stratum_present() -> None:
    qs = make_questions()
    alloc = stratified_allocation(qs, 13)
    counts = strata(qs, stratified_sample(qs, 13, 5))
    assert {s: c for s, c in alloc.items() if c} == dict(counts)


def test_largest_remainder_and_tie_break() -> None:
    sizes = {("a", "x"): 1, ("b", "x"): 1, ("c", "x"): 1}
    assert allocate(sizes, 2) == {("a", "x"): 1, ("b", "x"): 1, ("c", "x"): 0}
    assert allocate({("a", "x"): 5, ("b", "x"): 3}, 4) == {("a", "x"): 3, ("b", "x"): 1}


def test_n_at_least_len_returns_all() -> None:
    qs = make_questions()
    all_ids = sorted(q.question_id for q in qs)
    assert stratified_sample(qs, len(qs), 0) == all_ids
    assert stratified_sample(qs, 10_000, 0) == all_ids
