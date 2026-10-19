# ADR-003: Text-to-SQL Evaluation on BIRD Mini-Dev

**Status:** Accepted
**Date:** 2026-10-18

## Context

The existing eval runner (`evals/run_eval.py`) mocks the LLM and the tools, so it measures nothing about agent quality. We need a real, reproducible measurement of how well the agent turns questions into correct SQL, compared with a simple baseline, on public data, with the real token usage and cost of every run.

## Decision

### Data

- **BIRD Mini-Dev, SQLite version**: 500 question/SQL pairs over 11 databases (148 simple, 250 moderate, 102 challenging). License CC BY-SA 4.0.
  - Questions and gold SQL: Hugging Face `birdsql/bird_mini_dev`, file `data/mini_dev_sqlite-00000-of-00001.json` (the 2025-07-04 update). sha256 `88ceb0710163cae46a256ecea8f0a8c98286599530b60587fda5c3cfe57d45d2`. The copy inside the zip below is older (two gold SQL fixes missing, repeated question ids) and is not used.
  - Databases: `https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip` (linked from the mini_dev repository), sha256 `cc48ba16838204e4e214512030cb572eeb5f7bcdd999bae4b9b6ff12ec13b92f`. Only `minidev/MINIDEV/dev_databases/*/*.sqlite` is extracted.
  - Downloaded to `data/external/` (gitignored), never committed. `make bird-data` downloads, verifies both hashes and extracts.
- **Split**: fixed 150 dev / 350 test, stratified by difficulty, seed 0. Only the question ids are committed (`evals/text2sql/splits/{dev,test}.json`). Prompts and code are tuned on dev only; test is reported, never tuned on.
- **Chinook** stays for the interactive demo (later step), not for scoring.

### Metric

**Execution accuracy (EX)**, as in the BIRD paper: run predicted and gold SQL on the database and compare the result rows as sets (`set(pred) == set(gold)`), ignoring column names and row order. Databases are opened read-only (`mode=ro` URI). Each query has a 30 s timeout (SQLite progress handler); a timeout, an error or no SQL counts as wrong. Gold results are cached per question. EX is reported overall and per difficulty, with the number of questions.

### Systems compared

1. **Baseline, single shot**: one LLM call. Prompt = schema (the `CREATE TABLE` statements from `sqlite_master`) + external knowledge (the BIRD `evidence` field) + question, asking for one SQLite query only. SQL is taken from a fenced code block or the raw text.
2. **Agent**: the existing LangGraph graph (reason, execute, validate, respond), run unchanged except that the schema and the database are parameters instead of hardcoded. The question passed to the agent is the question followed by the evidence. The prompt for BIRD runs lists only `sql_query` and `calculator` (`doc_search` has nothing to search on BIRD) and has no sales-specific examples. The predicted SQL is the input of the last successful `sql_query` call, else of the last `sql_query` call, else empty.

Default behaviour of the API (sales database, three tools, current prompt text) must stay exactly the same.

### LLM providers

A small client layer in `src/llm/` with one method, `complete(system, user, *, tag) -> Completion(text, usage)`:

- **Ollama** through `langchain_ollama.ChatOllama`, temperature 0, `num_ctx` set explicitly (8192) because Ollama's default context is 2048 tokens and silently truncates longer prompts such as BIRD schemas. Cost $0.
- **Anthropic** through the official `anthropic` SDK (`messages.create`). Model `claude-haiku-5-5`. Thinking is on by default on this model and is billed as output tokens; a non-default `temperature` returns a 400, so temperature is not sent. `output_config.effort` is a config value. `stop_reason` is checked (`refusal`, `max_tokens` are recorded as failures for that question, not crashes).

Provider, model and parameters come from YAML files in `configs/llm/`. The API keeps using Ollama from `settings`.

### Token usage and cost

- Every LLM call records: run id, system, question id, tag (node name or `baseline`), input tokens, output tokens, cache read and cache write tokens, latency, stop reason. Written as JSONL next to the run results.
- Cost = tokens x prices from the config. Claude Haiku 5.5 prices (platform.claude.com pricing page, checked 2026-10-18), USD per million tokens, prompts up to 100K tokens: input 0.10, output 0.50, cache read 0.01. Prompts above 100K tokens are priced at 0.50 / 2.50. Prompt caching is not enabled.
- **Hard budget**: every run takes `--budget-usd`. The tracker refuses to start a call once spending has reached the cap and aborts the run with partial results saved. The overshoot is at most one call.

### Final answer step in EX runs

The respond step produces the natural-language answer, which EX does not score. EX runs may skip it with `--skip-respond` (`SKIP_RESPOND=1` in `make eval-sql`): `respond_node` then returns an empty response without an LLM call. Costs reported for such runs exclude that step, and `summary.json` records `skip_respond: true/false`. Runs with and without the flag are not cost-comparable.

### Outputs

`evals/results/text2sql/<run_id>/`: `predictions.jsonl` (per question and system: pred SQL, correct, error, tokens, cost, latency, LLM calls), `llm_calls.jsonl`, `summary.json` (EX overall and per difficulty, totals of tokens and cost, mean latency, config, git commit, split, timestamps). Summaries quoted in the README are copied to `evals/reports/` and committed.

## Alternatives considered

- **Spider**: bigger and older; BIRD has dirtier data, external knowledge and harder questions, closer to real BI use. Mini-Dev is small enough to run on a budget.
- **LLM-as-judge or string match on SQL**: execution accuracy is the standard for this task and needs no judge; many different SQL strings are correct.
- **langchain-anthropic instead of the `anthropic` SDK**: one more abstraction over usage fields and thinking blocks; the official SDK gives exact usage, cache fields and stop reasons.

## Consequences

- The first numbers measure the current agent, which plans once and has no error-recovery loop. They are the "before" for the agent rewrite (A3).
- Running Ollama on a CPU-only machine is slow; the runner supports `--limit` and resuming from existing predictions.
