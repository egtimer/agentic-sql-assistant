"""Stratified sampling of questions by (db_id, difficulty)."""

import random
from collections import defaultdict
from collections.abc import Sequence

from evals.text2sql.data import Question

Stratum = tuple[str, str]


def stratum_of(question: Question) -> Stratum:
    return (question.db_id, question.difficulty)


def stratum_label(stratum: Stratum) -> str:
    return f"{stratum[0]}/{stratum[1]}"


def allocate(sizes: dict[Stratum, int], n: int) -> dict[Stratum, int]:
    """Proportional allocation of ``n`` slots with largest-remainder rounding.

    Ties on the remainder are broken by stratum key (ascending), so the result is
    deterministic. Requires ``0 <= n <= sum(sizes)``; no stratum gets more than its size.
    """
    total = sum(sizes.values())
    if not 0 <= n <= total:
        raise ValueError(f"cannot allocate {n} of {total} questions")
    quotas = {s: n * size / total for s, size in sizes.items()}
    alloc = {s: int(q) for s, q in quotas.items()}
    remaining = n - sum(alloc.values())
    by_remainder = sorted(sizes, key=lambda s: (-(quotas[s] - alloc[s]), s))
    for s in by_remainder[:remaining]:
        alloc[s] += 1
    return alloc


def stratified_allocation(questions: Sequence[Question], n: int) -> dict[Stratum, int]:
    """Number of questions drawn from each (db_id, difficulty) stratum."""
    sizes: dict[Stratum, int] = defaultdict(int)
    for q in questions:
        sizes[stratum_of(q)] += 1
    return allocate(dict(sizes), min(n, len(questions)))


def stratified_sample(questions: Sequence[Question], n: int, seed: int = 0) -> list[int]:
    """Question ids of a stratified sample of size ``n`` (all ids when ``n >= len``).

    Strata are (db_id, difficulty). Slots are allocated proportionally with
    largest-remainder rounding; inside a stratum ids are drawn with
    ``random.Random(seed)`` from the sorted ids. Returns ids sorted ascending.
    """
    if n >= len(questions):
        return sorted(q.question_id for q in questions)
    groups: dict[Stratum, list[int]] = defaultdict(list)
    for q in questions:
        groups[stratum_of(q)].append(q.question_id)
    alloc = stratified_allocation(questions, n)
    rng = random.Random(seed)
    chosen: list[int] = []
    for stratum in sorted(groups):  # fixed stratum order keeps the draw reproducible
        ids = sorted(groups[stratum])
        chosen.extend(rng.sample(ids, alloc[stratum]))
    return sorted(chosen)
