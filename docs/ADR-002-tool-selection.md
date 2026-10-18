# ADR-002: Tool Selection Strategy

**Status:** Accepted  
**Date:** 2024-11-15

## Context

The agent must decide which tools to invoke for a given user query. Some queries require a single tool (e.g., a SQL query), while others require multiple tools in sequence (e.g., SQL followed by calculator). The selection mechanism must handle both cases reliably.

## Decision

Use **LLM-based tool selection** in the `reason_node`. The LLM receives tool descriptions and the user query, then outputs a JSON array of `ToolCall` objects specifying which tools to use, their inputs, and the reasoning.

```json
[
  {"tool": "sql_query", "input": "SELECT ...", "reason": "Get sales data"},
  {"tool": "calculator", "input": "PLACEHOLDER", "reason": "Compute growth %"}
]
```

The `PLACEHOLDER` pattern allows the LLM to plan a calculator step that depends on SQL results before those results are available.

## Alternatives Considered

### Classifier Model
A fine-tuned classification model that maps queries to tool names.

- **Pros:** Fast inference, deterministic output.
- **Cons:** Requires labeled training data. Cannot handle multi-tool queries without combinatorial explosion of classes. Cannot generate tool inputs (only selects the tool). Adding a new tool requires retraining.

### Keyword Routing
Rule-based routing using regex or keyword matching (e.g., "policy" → doc_search, "calculate" → calculator).

- **Pros:** Zero latency, fully deterministic, no LLM cost.
- **Cons:** Brittle with natural language variation. Cannot handle multi-tool queries. Cannot generate appropriate SQL or search inputs. Requires constant maintenance as query patterns evolve.

### Embedding Similarity
Compute embeddings for the query and predefined tool descriptions, then select the tool with highest cosine similarity.

- **Pros:** Handles paraphrasing better than keywords.
- **Cons:** Still cannot handle multi-tool selection without additional logic. Cannot generate tool inputs. Threshold tuning is fragile. Adds an embedding inference step without solving the full problem.

## Why LLM-Based Selection

1. **Multi-tool planning** — The LLM can output an ordered list of tools in a single pass, handling dependencies like "first query the database, then calculate growth." No other approach handles this naturally.

2. **Input generation** — The LLM doesn't just select tools; it generates the specific inputs (SQL queries, search terms, math expressions). This eliminates the need for separate input-generation logic.

3. **No training data required** — Works out of the box with prompt engineering. Tool descriptions in the prompt serve as the "training set." Adding a new tool means adding its description to the prompt.

4. **Flexible reasoning** — The `reason` field in each ToolCall provides an audit trail. The LLM can explain why it chose each tool, which aids debugging and observability.

5. **Graceful degradation** — If the LLM produces invalid JSON or unknown tools, the `reason_node` catches these errors and routes to the error response path rather than crashing.

## Trade-offs

- **Latency** — An LLM call adds ~200-500ms to the reasoning step. Acceptable for this use case since the bottleneck is typically tool execution, not planning.
- **Non-determinism** — Two identical queries may produce slightly different tool plans. Mitigated by using `temperature=0`.
- **Prompt sensitivity** — Tool descriptions and the system prompt must be carefully crafted. Changes to the prompt can affect tool selection accuracy.

## Consequences

- Tool descriptions in `src/agent/prompts.py` must be kept in sync with actual tool capabilities.
- The JSON extraction logic (`_extract_json`) must handle LLM output variations (markdown blocks, surrounding text).
- Evaluation of tool selection accuracy requires a golden dataset (see `evals/golden_dataset.json`).
