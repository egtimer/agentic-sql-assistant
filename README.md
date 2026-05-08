# agentic-sql-assistant

AI Agent with tool calling, multi-step reasoning, and full observability. The agent autonomously decides which tools to use, executes them in sequence, validates results, and produces traced, auditable responses.

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

## Key Metrics

| Metric | Value |
|---|---|
| Total tests | 103 |
| Tools | 3 (sql_query, doc_search, calculator) |
| Agent nodes | 4 (reason, execute, validate, respond) |
| Retrieval | Hybrid dense + BM25 + RRF (k=60) |
| Sample data | 10 products, 20 customers, 150 sales |
| Documents | 3 policy files for RAG |
| Observability | Langfuse tracing per node |

## Quick Start

```bash
make setup          # Install dependencies
make seed           # Populate SQLite + ingest docs
make serve          # Start FastAPI server
make test           # Run all 103 tests
```

## Example

```bash
curl -X POST http://localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How much did we sell in Q3 2024?"}'
```

```json
{
  "answer": "Total Q3 2024 sales were €234,500.",
  "tools_used": ["sql_query"],
  "trace_id": "abc123"
}
```

Multi-tool example:

```bash
curl -X POST http://localhost:8000/agent/query \
  -d '{"query": "What was the sales growth Q3 vs Q2?"}'
```

The agent executes `sql_query` twice (Q3 and Q2 totals), then `calculator` for the percentage, returning a combined answer with full tool traceability.

## Tools

**sql_query** — Read-only SQLite queries with injection protection and row limits. Rejects INSERT/UPDATE/DELETE/DROP.

**doc_search** — Hybrid retrieval (dense embeddings + BM25 keyword search) combined with Reciprocal Rank Fusion. Searches company policy documents in Qdrant.

**calculator** — Safe math evaluation using AST parsing. No `eval()`. Whitelisted operators and functions only, with exponent limits.

## Evaluation

The project includes an evaluation pipeline that measures agent quality against a golden dataset of 15 queries across 4 categories.

```bash
python -m eval.run_eval
```

### Metrics

| Metric | Description |
|---|---|
| Tool selection accuracy | Did the agent pick the correct tools for the query? |
| Answer contains accuracy | Does the response contain expected keywords/values? |

### Golden Dataset Breakdown

| Category | Count | Description |
|---|---|
| single-tool-sql | 5 | Queries that require only `sql_query` |
| single-tool-rag | 3 | Queries that require only `doc_search` |
| single-tool-calculator | 2 | Queries that require only `calculator` |
| multi-tool | 5 | Queries requiring 2+ tools in sequence |

Results are saved to `eval/results/latest.json` with per-query breakdowns and aggregate scores by category.

## Architecture Decision Records

- [ADR-001: Agent Architecture](docs/ADR-001-agent-architecture.md) — LangGraph StateGraph with 4 nodes
- [ADR-002: Tool Selection](docs/ADR-002-tool-selection.md) — LLM-based tool selection with JSON output

## Stack

Python 3.11 · LangGraph · Ollama · Qdrant · FastAPI · Langfuse · Pydantic · SQLite · Docker Compose

## Infrastructure

```bash
docker compose up -d   # Ollama + Qdrant + Langfuse + PostgreSQL
```

## Related

- [fastapi-rag-lab](https://github.com/egtimer/fastapi-rag-lab) — Production RAG system with hybrid retrieval, benchmarking (Recall@5 0.98), and RAGAS evaluation pipeline
