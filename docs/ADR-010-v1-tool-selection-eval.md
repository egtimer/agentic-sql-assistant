# ADR-010: Measuring agent v1's tool selection, and fixing it on dev only

**Status:** Accepted (prompt fixed on the dev split on 2026-10-30, before the test split was run)
**Date:** 2026-10-30

## Context

The API serves agent v1 (ADR-001, ADR-002): an LLM plans a JSON list of calls to `sql_query`, `doc_search` and `calculator`, the graph runs them and writes an answer. Its tool selection had only been checked with a mocked LLM on 15 queries (`evals/run_eval.py`), which tests the wiring, not the choices a real model makes. The BIRD evaluation (ADR-003) runs v1 with the SQL-only profile, without `doc_search`, and scores the SQL, not the tool choice.

## Decision

### Query set (`evals/tools/`)

- **60 hand-written queries** over the seeded demo data (`src/db/seed.py`, seed 42) and the three policy files in `data/documents/`: `sql` 18, `doc` 12, `calc` 6, `sql_math` 12, `sql_doc` 6, and `none` 6 (greetings and out-of-scope questions that need no tool). Split once into 23 dev and 37 test queries, stratified by category (seed 0).
- **Expected tools** per query: `required` (needed), `optional` (allowed: arithmetic that SQL can also do), and everything else counts as wrong.
- **Gold answers**: database values from a gold SQL query run on the seeded database at build time (`build_set.py`), facts quoted from the policy files, or plain arithmetic. Single-digit values must appear with their unit or keyword ("7%", "3 customers", "2 years").
- **Human review before any run.** The owner reviewed every query, its expected tools and its gold answer on a sheet without model output (`evals/tools/set/review.csv`, commit 4a1f25c); the changes (two rewordings, gold queries joined by name instead of ids, units for single-digit answers) left every gold value unchanged.

### Scores (`evals/tools/metrics.py`)

Over the distinct tools the agent executed: precision (used tools that are allowed), recall (required tools used) and exact match (all required, nothing outside the allowed set; for a `none` query, no tool at all). The answer passes when every number and name of the gold appears in it (numbers by absolute value with 1% relative tolerance, text without accents, accepted translations listed). `none` queries score tool selection only. Means with 95% percentile bootstrap CIs over queries; paired bootstrap of each baseline against `agent_v1`.

### Systems (`evals/tools/run.py`)

- `agent_v1`: the agent as the API serves it.
- `simple_prompt`: the same graph with a minimal tool-selection prompt (tool names, one line each, the table columns; no usage hints, no examples).
- `no_tools`: one LLM call that answers without data or documents.

### Protocol

Prompts change only on dev; the test split runs once, with Claude Haiku 5.5 and Claude Sonnet 5.5, after the last change; nothing in the prompt changes after test.

## The prompt change made on dev

**Before** (dev run `evals/reports/2026-10-30-v1-tools-haiku-dev`, Haiku 5.5, commit 870554a): the tool-selection prompt asked for a JSON array of tool calls and had no way to say "no tool". On the two `none` dev queries ("Hello! What can you help me with?", "What is the capital of France?") v1 called `doc_search`; an empty array, had the model returned one, was treated as an error and answered with a fixed error message.

**Change** (commit 4b1d04a): the default profile's prompt now says that a question needing no sales data, company document or calculation (a greeting, thanks, general knowledge, anything outside the company's sales and policies) gets an empty array, with one example (`"Hola, ¿qué tal?"` → `[]`). An explicit `[]` is answered with one direct LLM call (`DIRECT_RESPONSE_PROMPT`) instead of an error. The SQL-only profile used by the BIRD runs is unchanged.

**After** (dev run `evals/reports/2026-10-30-v1-tools-haiku-dev-it2`, commit 4b1d04a). Both runs were scored with the same scorer (rescored with `--rescore` after the number-parser fix in 2605471, which only changed one `simple_prompt` answer in the second run):

| `agent_v1`, dev (n = 23) | Before | After |
|---|---|---|
| Tool selection, exact | 91.3% [78.3, 100.0] | 100.0% [100.0, 100.0] |
| Tool precision | 91.3% [78.3, 100.0] | 100.0% [100.0, 100.0] |
| `none` queries without a tool | 0 of 2 | 2 of 2 |
| Answer correct (n = 21 scored) | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] |
| Cost / query | $0.00032 | $0.00030 |
| Latency / query | 3.4 s [2.9, 4.1] | 3.4 s [2.8, 4.2] |

The only queries whose tools or answer changed are the two `none` queries; v1 now answers them directly ("Hello! I'm the business intelligence assistant…", "The capital of France is Paris."). With 2 such queries on dev this is a check that the fix works, not a measurement of how well; the test split has 4 more.

Baselines on the same dev runs, before → after: `simple_prompt` exact 95.7% → 95.7%, answers 85.7% → 85.7%; `no_tools` answers 14.3% → 9.5%. `simple_prompt` loses answers on policy questions because it searches the Spanish documents in English and retrieves chunks without the fact; the tool choice is right. Dev cost: $0.0164 and $0.0160.

## Results on test

To be added after the single test run.

## Consequences

- The API's answers to greetings and out-of-scope questions change from a document search (or an error message) to one direct reply.
- The tool-selection scores describe this demo dataset and these 60 queries; they say nothing about other schemas or documents.
- The answer check looks for the gold values in the text; an answer that states the right number for the wrong reason passes. Single-digit values need their unit, which makes false passes less likely but not impossible.
