# ADR-001: Agent Architecture

**Status:** Accepted  
**Date:** 2024-11-15

## Context

The system needs an agent that can reason about user queries, select appropriate tools, execute them (potentially in sequence), validate results, and produce a final response. The architecture must support multi-step reasoning where later tools depend on earlier results.

## Decision

Use **LangGraph StateGraph** with 4 nodes: `reason`, `execute`, `validate`, `respond`.

```
reason → execute → validate → respond → END
           ↑          │
           └──────────┘ (loop if needs_more_tools)
```

State flows through all nodes via a typed `AgentState` (Pydantic model) that accumulates tool calls, results, and the final response.

## Alternatives Considered

### ReAct Loop
A single loop where the LLM alternates between reasoning and acting until it decides to stop.

- **Pros:** Simple to implement, well-documented pattern.
- **Cons:** No explicit validation step. Hard to test individual stages. The LLM controls loop termination, making behavior less predictable. No clean separation between planning and execution.

### LangChain AgentExecutor
The built-in LangChain agent executor with automatic tool dispatch.

- **Pros:** Minimal code, handles tool calling automatically.
- **Cons:** Opaque execution flow. Limited control over routing logic. Difficult to add custom validation. Hard to instrument individual steps for observability. Couples tightly to LangChain's internal abstractions.

### Custom Python Loop
A hand-written orchestration loop with if/else branching.

- **Pros:** Full control, no framework dependency.
- **Cons:** No built-in state management. Manual graph traversal logic becomes error-prone as complexity grows. No visualization or debugging tools. Harder to add new nodes or edges without refactoring.

## Why LangGraph

1. **Explicit state management** — `AgentState` is a typed Pydantic model. Every node receives and returns a well-defined state, making data flow visible and debuggable.

2. **Conditional edges** — Routing decisions (`_should_continue_execution`, `_should_loop_or_respond`, `_should_reason_or_error`) are pure functions that can be unit tested independently of the nodes they connect.

3. **Testable nodes** — Each node (`reason_node`, `execute_node`, `validate_node`, `respond_node`) is a standalone function. They can be tested in isolation with constructed state, without running the full graph.

4. **Observability** — The `@trace_node` decorator instruments each node with Langfuse spans. The graph structure itself is introspectable, enabling visual debugging of execution paths.

5. **Extensibility** — Adding a new node (e.g., a caching layer or a safety checker) requires adding one node and one edge, without restructuring existing code.

## Consequences

- Team must understand LangGraph's StateGraph API and conditional edge patterns.
- State schema changes require updating `AgentState` and potentially all nodes that read the changed fields.
- The graph is compiled once and cached (`get_graph()`), so structural changes require a restart.
