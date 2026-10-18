# agentic-sql-assistant

AI Agent with tool calling and multi-step reasoning. The agent plans tool calls with an LLM, executes them in sequence, and composes an answer.

## Status

- The evaluation runner uses a mocked LLM and mocked tools, so it does not measure agent quality yet.
- Langfuse per-node spans are not working yet.
- The `sql_query` guardrail is a keyword blocklist, not a full read-only guarantee.

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

## Evaluation

The project includes a golden dataset of 15 queries across 4 categories. The current runner (`python -m evals.run_eval`) uses a mocked LLM and mocked tools, so it does not measure agent quality. It is being replaced by an execution-based eval against a public text-to-SQL benchmark, with a baseline. No results are reported yet.

### Golden Dataset Breakdown

| Category | Count | Description |
|---|---|---|
| single-tool-sql | 5 | Queries that require only `sql_query` |
| single-tool-rag | 3 | Queries that require only `doc_search` |
| single-tool-calculator | 2 | Queries that require only `calculator` |
| multi-tool | 5 | Queries requiring 2+ tools in sequence |

## Architecture Decision Records

- [ADR-001: Agent Architecture](docs/ADR-001-agent-architecture.md) — LangGraph StateGraph with 4 nodes
- [ADR-002: Tool Selection](docs/ADR-002-tool-selection.md) — LLM-based tool selection with JSON output

## Stack

Python 3.12 · LangGraph · Ollama · Qdrant · FastAPI · Langfuse · Pydantic · SQLite · Docker Compose

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
