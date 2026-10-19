# agentic-sql-assistant

LLM agents that answer questions over SQL databases, and the evaluation that measures whether they work. Two agents live here: the original tool-calling agent (v1, served by the API) and a text-to-SQL agent (v2) that writes one answer-shaped query, runs it and reviews the result in a bounded loop. Both are scored on the public BIRD Mini-Dev benchmark by execution accuracy, with confidence intervals and per-call cost.

**Headline (BIRD Mini-Dev test split, 350 questions):** agent v2 beats a single-shot baseline by +3.7 points with both Claude Haiku 5.5 (64.3% vs 60.6%) and Claude Sonnet 5.5 (69.7% vs 66.0%), and Haiku with agent v2 matches Sonnet single shot at a fifth of the cost per question. Details and caveats in [Evaluation](#evaluation-text-to-sql-on-bird-mini-dev).

## Status

- Agent v2 (`src/sql_agent/`) is evaluated through the eval runner; the API still serves agent v1. Moving the API to v2 is a planned step.
- Langfuse per-node spans are not working yet.
- The `sql_query` guardrail is a keyword blocklist, not a full read-only guarantee (being replaced).

## Architecture

```
                    ┌──────────────┐
                    │   FastAPI    │
                    │ POST /query  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   LangGraph  │
                    │    Agent     │
                    │              │
                    │ reason ─────►│
                    │ execute ────►│
                    │ validate ───►│
                    │ respond ────►│
                    └──┬───┬───┬──┘
                       │   │   │
              ┌────────┘   │   └────────┐
              ▼            ▼            ▼
         ┌────────┐  ┌─────────┐  ┌──────────┐
         │ SQLite │  │ Qdrant  │  │Calculator│
         │ Query  │  │  RAG    │  │ (AST)    │
         └────────┘  └─────────┘  └──────────┘
                           │
                    ┌──────▼───────┐
                    │   Langfuse   │
                    │   Tracing    │
                    └──────────────┘
```

## At a glance

| Item | Value |
|---|---|
| Tools | 3 (sql_query, doc_search, calculator) |
| Agent nodes | 4 (reason, execute, validate, respond) |
| Retrieval | Dense candidates re-scored with BM25, fused with RRF (k=60) |
| Sample data | 10 products, 20 customers, 150 sales |
| Documents | 3 policy files for RAG |
| Observability | Langfuse integration in progress (per-node spans not yet working) |

## Quick Start

```bash
make setup          # Install dependencies with uv (Python 3.12)
make up             # Start Ollama, Qdrant and Langfuse (docker compose up -d)
make seed           # Populate the SQLite sample database
make ingest         # Embed and index the policy documents into Qdrant
make serve          # Start the FastAPI server
make test           # Run the test suite
```

## Example

```bash
curl -X POST http://localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How much did we sell in Q3 2024?"}'
```

Multi-tool example:

```bash
curl -X POST http://localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was the sales growth Q3 vs Q2?"}'
```

## Tools

**sql_query** — Executes a single SELECT against SQLite. Write keywords are blocked with a regex blocklist; this is not a full guardrail yet (see Status).

**doc_search** — Dense search in Qdrant returns candidates, BM25 re-scores those candidates, and the two rankings are fused with Reciprocal Rank Fusion. Searches company policy documents.

**calculator** — Safe math evaluation using AST parsing. No `eval()`. Whitelisted operators and functions only, with exponent limits.

## Evaluation: text-to-SQL on BIRD Mini-Dev

**Setup.** [BIRD Mini-Dev](https://huggingface.co/datasets/birdsql/bird_mini_dev) (SQLite version, 500 questions over 11 databases, CC BY-SA 4.0; Li et al., "Can LLM Already Serve as a Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs", NeurIPS 2024), split once into 150 dev and 350 test questions, stratified by difficulty (seed 0). Metric: execution accuracy (EX), the predicted and gold queries return the same set of rows. 95% CIs by bootstrap over questions (10,000 resamples); differences between systems use a paired bootstrap on the same questions. Prompts were developed on dev only and frozen before the single test run. Every number below comes from a committed run in [`evals/reports/`](evals/reports/) (summary, per-question predictions without gold SQL, and every LLM call with tokens and cost). Design: [ADR-003](docs/ADR-003-text-to-sql-evaluation.md), [ADR-004](docs/ADR-004-sql-agent-v2.md).

Systems:

| System | What it does |
|---|---|
| `baseline` | one LLM call: schema DDL + external knowledge + question -> SQL |
| `baseline_ctx` | same, with the v2 schema context (keys and example values per text column) |
| `agent_v1` | the original agent: plans a JSON list of tool calls (SQL, calculator), then writes an answer |
| `agent_v2_nocheck` | v2 generate prompt (one query whose result is the answer); review only after execution errors |
| `agent_v2` | v2 full: generate, execute, check the result against the question, up to 2 fixes |

All runs: effort `low`, prompt caching on the schema prefix, final natural-language answer skipped (EX scores the SQL).

### Test split (350 questions, run once)

| Model | System | EX [95% CI] | Cost / question | LLM calls / question | Mean latency |
|---|---|---|---|---|---|
| Claude Haiku 5.5 | `baseline` | 60.6% [55.4, 65.7] | $0.00019 | 1.00 | 1.9 s |
| Claude Haiku 5.5 | `baseline_ctx` | 61.7% [56.6, 66.9] | $0.00022 | 1.00 | 2.1 s |
| Claude Haiku 5.5 | `agent_v2_nocheck` | 64.6% [59.4, 69.4] | $0.00021 | 1.00 | 1.9 s |
| Claude Haiku 5.5 | `agent_v2` | 64.3% [59.1, 69.4] | $0.00040 | 2.07 | 3.9 s |
| Claude Sonnet 5.5 | `baseline` | 66.0% [61.1, 71.1] | $0.00201 | 1.00 | 2.3 s |
| Claude Sonnet 5.5 | `agent_v2` | 69.7% [64.9, 74.6] | $0.00515 | 2.07 | 4.9 s |

Paired differences on the same 350 questions (95% CI):

| Comparison | Difference |
|---|---|
| Haiku: `agent_v2` - `baseline` | +3.7 pts [+0.6, +7.1] |
| Haiku: `agent_v2_nocheck` - `baseline` | +4.0 pts [+1.1, +7.1] |
| Haiku: `agent_v2` - `agent_v2_nocheck` (the check step) | -0.3 pts [-2.9, +2.3] |
| Haiku: `baseline_ctx` - `baseline` (example values) | +1.1 pts [-1.4, +3.7] |
| Sonnet: `agent_v2` - `baseline` | +3.7 pts [+0.9, +6.6] |
| Sonnet `baseline` - Haiku `agent_v2` | +1.7 pts [-2.3, +5.7] |
| Sonnet `agent_v2` - Haiku `agent_v2` | +5.4 pts [+2.3, +8.9] |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/cost-vs-ex-dark.svg">
  <img alt="Cost per question (log scale) against execution accuracy with 95% CIs on the test split: Haiku single shot 60.6% at $0.00019, Haiku agent v2 64.3% at $0.00040, Sonnet single shot 66.0% at $0.00201, Sonnet agent v2 69.7% at $0.00515" src="docs/img/cost-vs-ex-light.svg">
</picture>

**Which model.** On a tight budget, use Haiku 5.5 with agent v2: it is statistically level with Sonnet 5.5 single shot (-1.7 pts, CI -5.7 to +2.3) at a fifth of the cost per question. When accuracy matters more than cost, Sonnet 5.5 with agent v2 is the best measured setup, +5.4 pts [+2.3, +8.9] over Haiku with agent v2 for about 13 times the cost per question ($0.0052 vs $0.0004).

### How we got here: v1 -> single shot -> v2 (dev split, Haiku 5.5)

| Run | System | EX [95% CI] | vs `baseline` in the same run |
|---|---|---|---|
| [A2](evals/reports/2026-10-19-agent-v1-haiku-dev-150/) | `agent_v1` | 40.7% [32.7, 48.7] | -18.7 pts [-26.7, -10.7] |
| A2 | `baseline` | 59.3% [51.3, 66.7] | |
| [v2 iteration 1](evals/reports/2026-10-19-agent-v2-it1-haiku-dev-150/) | `agent_v2` | 68.7% [60.7, 76.0] | +8.7 pts [+3.3, +14.0] |
| [v2 replication](evals/reports/2026-10-19-agent-v2-replication-haiku-dev-150/) | `agent_v2` | 68.7% [61.3, 76.0] | +7.3 pts [+3.3, +12.0] |

1. **The naive agent loses 18.7 points to a single LLM call.** Agent v1 plans "fetch data with SQL, compute with a calculator, write an answer", so its SQL returns the pieces (two counts instead of a ratio) or extra context columns. Of its 89 errors on dev, 67 return more columns than the gold query. EX scores the query, and the query is not the answer.
2. **Asking for an answer-shaped query is where the gain comes from.** `agent_v2_nocheck` makes one call per question on every split (no execution errors occurred), so it is a single-shot system with the v2 prompt, and it carries the whole improvement over `baseline`. On test with Haiku, wrong results with more columns than the gold query drop from 27 (`baseline`) to 7 (`agent_v2`). Example values in the schema add nothing measurable.
3. **The check step does not pay off on Haiku.** It doubles calls and latency for -0.3 pts on test (+1.3 and +0.0 on the two dev runs). It is precise but rarely acts: on test it changed 24 of 350 queries, 6 wrong to right, 3 right to wrong, 12 still wrong, 3 still right. On Sonnet its effect is measured within the run only (first query 68.3% -> final 69.7%; 8 wrong to right, 3 right to wrong); there is no Sonnet `agent_v2_nocheck` control. On Sonnet most check answers explain the review before saying OK; the parser counted them as OK, and the full check text was not stored, so a fix written outside a ```sql block cannot be ruled out ([report](evals/reports/2026-10-19-sonnet-test-350/README.md)).
4. **Dev overstates the gain.** v2 led by +7 to +9 pts on dev and by +3.7 on test, after one prompt iteration on dev. Most remaining errors return the right shape with wrong values (72 of agent v2's 125 errors with Haiku, 57 of 106 with Sonnet): a wrong table, filter or interpretation, and some quirks of the gold queries (extra columns, booleans per row). Error shapes are computed with `python -m evals.text2sql.error_shapes <report>`.

### Local model: Ollama on CPU

`qwen2.5:3b` through Ollama on a 4-core CPU without GPU, 20 stratified dev questions ([report](evals/reports/2026-10-19-ollama-qwen2.5-3b-dev-20/)):

| System | EX on the 20 questions [95% CI] | Mean / median latency per question | Haiku 5.5 on the same 20 questions |
|---|---|---|---|
| `baseline` | 30% [10, 50] | 43.0 s / 18.2 s | 55% (`baseline`), 65% (`agent_v2`) |
| `agent_v1` | 20% [5, 40] | 66.2 s / 39.8 s | 30% (`agent_v1`) |

On these 20 questions Haiku single shot leads the local baseline by +25 pts [+5, +45] at about 2 s and $0.0002 per question. The local model only solves simple questions (0 of the moderate and challenging ones), and prompt processing dominates its latency (about 14 tokens/s on a cold prefix). n=20 gives wide intervals.

### Cost of the evaluation

All API runs, including a [10-question pilot](evals/reports/2026-10-18-pilot-haiku-10/) and a [20-question Sonnet pilot](evals/reports/2026-10-19-sonnet-dev-20-pilot/) used to estimate costs, cost $3.57 in total (Haiku test run $0.36, Sonnet test run $2.51), as recorded in the reports' `llm_calls.jsonl`.

### Reproduce

```bash
make bird-data   # download BIRD Mini-Dev, verify sha256, extract, write the split
uv run python -m evals.text2sql.run --llm-config configs/llm/claude-haiku-5-5.yaml \
  --system baseline,baseline_ctx,agent_v2_nocheck,agent_v2 --split dev --skip-respond \
  --workers 4 --budget-usd 0.50 --run-id my-dev-run
uv run python -m evals.text2sql.compare_runs 2026-10-19-haiku-test-350:agent_v2 2026-10-19-sonnet-test-350:baseline
uv run python -m evals.text2sql.plot_cost_ex --out docs/img/cost-vs-ex
```

API runs need `ANTHROPIC_API_KEY` and spend money; `--budget-usd` is a hard cap. The exact command of every reported run is in its report's README.

### Golden dataset (legacy)

`python -m evals.run_eval` scores agent v1 on 15 hand-written queries with a mocked LLM and mocked tools; it checks wiring, not quality.

| Category | Count | Description |
|---|---|---|
| single-tool-sql | 5 | Queries that require only `sql_query` |
| single-tool-rag | 3 | Queries that require only `doc_search` |
| single-tool-calculator | 2 | Queries that require only `calculator` |
| multi-tool | 5 | Queries requiring 2+ tools in sequence |

## Architecture Decision Records

- [ADR-001: Agent Architecture](docs/ADR-001-agent-architecture.md) — LangGraph StateGraph with 4 nodes
- [ADR-002: Tool Selection](docs/ADR-002-tool-selection.md) — LLM-based tool selection with JSON output
- [ADR-003: Text-to-SQL Evaluation](docs/ADR-003-text-to-sql-evaluation.md) — BIRD Mini-Dev, execution accuracy, budgets and caching
- [ADR-004: SQL Agent v2](docs/ADR-004-sql-agent-v2.md) — code-driven generate, execute, check loop for text-to-SQL

## Stack

Python 3.12 · LangGraph · Claude (Anthropic SDK) · Ollama · Qdrant · FastAPI · Langfuse · Pydantic · SQLite · Docker Compose

## Infrastructure

```bash
docker compose up -d   # app + Ollama + Qdrant + Langfuse + PostgreSQL (CPU only)
```

Langfuse server secrets in `docker-compose.yml` have local-development defaults and can be overridden through the variables documented in `.env.example`.

## Development

```bash
make lint              # ruff check, ruff format --check, mypy --strict
make test              # pytest
make cov               # pytest with coverage report
make docker-build      # build the application image
docker compose up -d   # run the full stack
```

GitHub Actions (`.github/workflows/ci.yml`) runs lint, tests and a Docker build on pushes and pull requests to `main`.

## Related

- [fastapi-rag-lab](https://github.com/egtimer/fastapi-rag-lab) — Production RAG system with hybrid retrieval, benchmarking (Recall@5 0.98), and RAGAS evaluation pipeline
