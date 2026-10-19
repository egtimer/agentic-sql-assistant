# Guardrail evaluations

Design and results: [ADR-005](../../docs/ADR-005-sql-guardrails.md). Every case runs on a disposable copy of a database inside a temporary directory that is also the working directory for that case.

- `attacks.jsonl`: `{id, category, sql, expect, db}`; `expect` is `blocked` (security cases) or `bounded` (resource cases); `db` is `student_club` (a copy of the BIRD database) or `fixture` (a synthetic database built by `cases.build_fixture`, with the policy in `cases.FIXTURE_POLICY`). A `<<REPEAT n unit>>` macro in `sql` is expanded at load time.
- `benign.jsonl`: `{id, sql, db}`, legitimate queries that look dangerous. The 500 BIRD gold queries are added at run time when `data/external/` is present.
- `injection.jsonl`: `{id, type, db, question, evidence, setup?}`; `type` is `direct`, `indirect` (the `setup` UPDATE writes an instruction into a copy of the database before the run) or `control`; `db` may also be `sales` (the sample database, seeded into the temporary copy).
- `guards.py`: `legacy_v1`, `readonly_only`, `sql_guard`, `no_lexical` behind one interface.
- `run_sql.py`: `uv run python -m evals.guardrails.run_sql --out evals/reports/<date>-guardrails-sql/` (no LLM).
- `classify.py` and `run_injection.py`: `uv run python -m evals.guardrails.run_injection --llm-config configs/llm/claude-haiku-5-5.yaml --budget-usd 0.25 --out evals/reports/<date>-guardrails-injection/` (real LLM calls; `--budget-usd` is a hard cap).

Unguarded variants can allocate several GB in the resource cases.
