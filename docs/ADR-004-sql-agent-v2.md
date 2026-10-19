# ADR-004: SQL Agent v2, a Code-Driven Generate-Execute-Check Loop

**Status:** Accepted
**Date:** 2026-10-19

## Context

The text-to-SQL evaluation (ADR-003) measured agent v1 against the single-shot baseline on the 150 dev questions of BIRD Mini-Dev, with `claude-haiku-5-5` (report: `evals/reports/2026-10-19-agent-v1-haiku-dev-150/`):

- Baseline EX 59.3% (89/150), agent v1 EX 40.7% (61/150). The paired difference is negative with a 95% CI that excludes zero.
- Of v1's 89 errors, 67 return more columns than the gold query. v1's prompt teaches "fetch data, compute with the calculator, write an answer", so it returns the ingredients of the answer instead of the answer.
- The baseline's 61 errors: 25 right shape but wrong values, 16 more columns than gold, 8 different row count (LIMIT, DISTINCT, ties), 7 fewer columns, 5 empty results (filters on values that do not exist).

These counts come from the owner's analysis of that dev run. They say what a better agent has to do: produce one final SQL query that directly answers the question, and correct itself when the result looks wrong (extra columns, empty result, errors).

## Decision

A new package, `src/sql_agent/` (agent v2), separate from `src/agent/`. Agent v1 and the sales assistant keep their behaviour; nothing in the API changes.

### Graph

A LangGraph `StateGraph` with one node per step. The flow is decided by code (conditional edges on the state); the model only writes SQL or an OK verdict.

1. `introspect` (no LLM, cached per database file and options): the schema context is the `CREATE TABLE` DDL from `sqlite_master`, the primary and foreign keys that the DDL does not already state (`PRAGMA`), and, with `sample_values`, an "Example values" section: up to 3 distinct non-null values (length <= 40) for every column with TEXT affinity, written as SQL literals. Each sampling query has a 2 s timeout and a column is skipped on error. The output is deterministic for a given file. Databases are opened read-only.
2. `generate` (LLM): writes one SQLite query whose result is the answer. The rules in the system prompt: only the requested columns in the requested order, ratios and differences computed in SQL as one value, `ORDER BY ... LIMIT` for superlatives, `DISTINCT` for lists of entities, the external knowledge applied as given, only existing tables, columns and values consistent with the examples.
3. `execute` (no LLM): runs the candidate read-only with a 30 s timeout and records an `Attempt`: columns, row count (up to 1001 rows are fetched, ">1000" is reported beyond that), the first 20 rows with each value cut at 60 characters, or the error ("timeout" on interrupt).
4. `check` (LLM): a reviewer checklist (columns, aggregation, row count, empty result or error, evidence formula, joins and filters). It answers exactly `OK` when it finds no concrete problem, otherwise the corrected full query in a ```sql block. Its input includes signals computed in code ("The result is empty.", the number of columns and rows). A ```sql block is a fix, text starting with `OK` is ok, anything else is treated as ok and counted as `unparsed`. A "fix" identical to the current query counts as ok.
5. `respond`: sets the final SQL; unless `skip_respond`, one LLM call writes a short natural-language answer. EX runs use `skip_respond`.

### Routing and bounds

After `generate` the flow always goes to `execute`. After `execute`, a failed attempt (error or timeout) goes to `check` while `fixes_used < max_fixes`; a successful attempt goes to `check` only with `use_check` and while `fixes_used < max_fixes`; otherwise to `respond`. After `check`, `ok` goes to `respond` and a fix increments `fixes_used` and goes back to `execute`. With `use_check=False` (the ablation "v2 without check") `check` is still used to recover from failed executions, never to review successful ones. `max_fixes` defaults to 2, so an EX run makes at most `1 + max_fixes` LLM calls per question.

### Final SQL

A pure function, tested: the last attempt that executed successfully (if the last attempt failed, the latest successful one); if none succeeded, the last attempt's SQL; with no attempt, the empty string.

### Prompt caching

Every LLM call goes through `get_client().complete(system, user, tag=<node>, prefix=<stable part>)`. For `generate` and `check` the prefix is the schema block of the database (byte-identical across the questions of a database, and across both nodes); the per-question text (evidence, question, candidate and result) is in the user part. A test asserts this.

### Why code-driven and not a free tool loop

- Provider-agnostic: only plain text completions are needed, no native tool-calling, so it works on Ollama and on the API.
- Testable: routing, parsing and the final-SQL rule are pure functions, and the whole graph runs with a scripted fake model.
- Bounded cost: the number of LLM calls per question has a hard upper bound, which matters with a budget cap.
- Diagnosable: attempts and verdicts are recorded, so we can see what the loop changed.

## Evaluation

Systems, all run with `evals/text2sql/run.py` (`--system` takes a comma-separated list):

| System | What it is |
|---|---|
| `baseline` | single shot, plain DDL |
| `baseline_ctx` | single shot, same prompt, with the v2 context (keys and example values) |
| `agent_v2_nocheck` | v2 with error recovery only |
| `agent_v2` | full v2 |

`agent_v1` stays available for reference (`agent` is an alias of it; `both` means `baseline,agent_v1`). The ablation separates the effect of the context (`baseline_ctx` vs `baseline`), of error recovery (`agent_v2_nocheck` vs `baseline_ctx`) and of the check step (`agent_v2` vs `agent_v2_nocheck`).

The runner stores a paired bootstrap of EX(j) - EX(i) for every pair of systems, in the order given (`comparison.agent_v2_minus_baseline`, ...). For v2 systems it also records, per question, the attempts (SQL, error, row count, number of columns), the verdicts, `fixes_used` and `first_sql_correct` (the first generated SQL scored like the final one). The summary adds `flips` (among the questions where the final SQL differs from the first one: wrong to right, right to wrong, wrong to wrong, right to right), the fraction of questions with at least one fix, the mean number of LLM calls, the number of unparsed checks and the EX of the first SQL.

**Success criterion:** on dev, the lower bound of the 95% paired CI of `agent_v2` minus `baseline` is above zero.

### Discipline

- At most 3 prompt iterations on dev. Every run is saved in `evals/reports/`, including the ones that did not help. Then the prompts are frozen.
- The test split is run once, by the owner, with the frozen prompts.

## Out of scope

Moving the API to agent v2 is a later step. The sales profile and agent v1 are untouched.

## Risks

- BIRD gold queries sometimes include extra columns that the question does not ask for. A v2 that follows the question strictly can be marked wrong there; this caps the achievable EX.
- The check step can turn right answers into wrong ones. That is why the flips are measured, and why `agent_v2_nocheck` exists as a control.
- Example values cost prompt tokens (the schema context is larger on some databases) and are a sample, not an exhaustive list of the values.
- Sampling example values can take a few seconds on large databases (the 2 s timeout is per column); the result is cached per process.

## Results on dev (2026-10-19)

Claude Haiku 5.5, effort `low`, 150 dev questions, `--skip-respond`. Two runs with the same frozen prompts: iteration 1 (`evals/reports/2026-10-19-agent-v2-it1-haiku-dev-150/`) and a replication (`evals/reports/2026-10-19-agent-v2-replication-haiku-dev-150/`). Prompts were frozen after iteration 1 (1 of the 3 allowed iterations).

| System | EX iteration 1 [95% CI] | EX replication [95% CI] | LLM calls/q | Cost/q (USD) |
|---|---|---|---|---|
| `baseline` | 60.0% [52.0, 68.0] | 61.3% [53.3, 68.7] | 1 | 0.00021 |
| `baseline_ctx` | 60.7% [52.7, 68.0] | 59.3% [51.3, 67.3] | 1 | 0.00024-0.00025 |
| `agent_v2_nocheck` | 67.3% [60.0, 74.7] | 68.7% [61.3, 76.0] | 1.00 | 0.00024-0.00025 |
| `agent_v2` | 68.7% [60.7, 76.0] | 68.7% [61.3, 76.0] | 2.05-2.07 | 0.00041-0.00042 |

Paired differences (95% paired bootstrap, iteration 1 / replication):

- `agent_v2` - `baseline`: +8.7 [+3.3, +14.0] / +7.3 [+3.3, +12.0]. The success criterion is met in both runs.
- `baseline_ctx` - `baseline`: +0.7 [-4.0, +5.3] / -2.0 [-6.7, +2.0]. Example values alone do not help.
- `agent_v2_nocheck` - `baseline_ctx`: +6.7 [+2.0, +12.0] / +9.3 [+4.0, +14.7]. No execution errors occurred, so `agent_v2_nocheck` made one call per question: this gain comes from the v2 generate prompt (one answer-shaped query), not from the loop.
- `agent_v2` - `agent_v2_nocheck`: +1.3 [-2.0, +5.3] / +0.0 [-4.0, +4.0]. The check step is not significant on this model while doubling the calls.

Check step flips (questions whose final SQL differs from the first): 11 and 8; wrong to right 2 and 2, right to wrong 0 and 0, wrong to wrong 8 and 5, right to right 1 and 1. A fix was applied in 7.3% and 5.3% of the questions.

Reading: on Claude Haiku 5.5 the gain over single shot comes from asking for an answer-shaped query; the check step is precise (no right-to-wrong flips) but rarely finds a fix. The remaining errors (47 per run) are mostly semantic (wrong table or interpretation) or BIRD gold quirks, not shape errors. Whether the check step pays off on a weaker model is open. Agent v1 for reference: 40.7% on the same split (`evals/reports/2026-10-19-agent-v1-haiku-dev-150/`).
