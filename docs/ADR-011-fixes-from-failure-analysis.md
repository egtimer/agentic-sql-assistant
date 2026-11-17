# ADR-011: Fixes from the failure analysis: calculator input, tool output length, answer context

**Status:** Accepted (measured before and after on 2026-11-17)
**Date:** 2026-11-17

## Context

The failure analysis of 2026-11-17 ([report](../evals/reports/2026-11-17-failure-analysis/README.md)) classified the failures of agent v1 (tool eval, 555 runs) and agent v2 (BIRD test, Haiku) and checked 40 of them by hand. Three causes were bugs in the agents rather than limits of the models:

1. **Agent v1's calculator input.** The planner writes `PLACEHOLDER` for a calculation that needs SQL results. `execute_node` replaced it with a fixed growth formula, `(b − a) / a × 100`, over the first two numbers of the last SQL result, for any question; when the result had another shape the placeholder stayed and the calculator failed (19 of 555 runs). `validate_node` could fill it with the model, but the graph runs every planned tool before validation, so for a calculator planned after its SQL that path never ran. In the 15 noise repetitions, 15 of 92 calculator results on `sql_math` queries matched the gold value; the answers were still right because the respond step ignored the calculator or redid the arithmetic.
2. **Agent v1's tool output length.** Each tool output was cut to 1,500 characters in the prompts, silently. A document search returns about 2,500. All 4 tool-eval failures were runs where the passage with the answer landed after character 1,500.
3. **Agent v2's answer context.** The respond step received the question and up to 20 result rows, not the query, and `>1000` as the row count of large results. The reviewer found answers that doubted a filter the query had already applied, and answers that guessed the size of a large result.

## Decision

1. **The model writes the calculator expression** just before the calculator runs (`_fill_placeholder`, prompt `CALC_FILL_PROMPT`, tag `fill`): the question, the planned reason and the earlier results in, one arithmetic expression or `NONE` out. Anything that is not a bare expression of digits, operators, brackets and decimal points is reported as a failed step and not run. The fixed formula is removed. The extra call happens only when a placeholder is planned.
2. **Tool outputs keep 6,000 characters** in the prompts (`MAX_TOOL_OUTPUT_CHARS`), and a cut says how many characters were left out. 6,000 holds a full search result with room to spare and stays small next to the context window.
3. **The answer step sees the final SQL and the exact row count.** `respond_user` adds the query that ran; when the fetch stopped at 1,000 rows, `count_rows` runs `SELECT COUNT(*)` over the query under the SQL guard and the prompt gets that number. `RESPOND_SYSTEM` says that the query's filters are already applied, and that a partial list must be called partial with the total given. Generate and check prompts do not change; the snapshot test was regenerated on purpose and differs only in the respond prompts.
4. **The old answer prompt stays reproducible** (`evals/failures/respond_versions.py`, `--respond v1`), so final-answer passes can compare the two on the same SQL.
5. **Embedded Qdrant for machines without Docker.** `src/rag/qdrant.py` opens every client; `QDRANT_URL=file:<dir>` uses Qdrant's local mode with one shared client per directory. The default stays the Qdrant server.

## Measurement

Before and after on the same items and environment ([report](../evals/reports/2026-11-17-fixes/README.md), $1.73):

- **Calculator:** 8 of 48 results equal the gold value before, 49 of 49 after, over 5 Haiku and 3 Sonnet runs of the tool eval each; 0 failed calls after (8 before). Tool selection and answers are 100% in all 16 runs, before and after.
- **Tool output length:** the late passage occurred once in the 16 runs, after the fix, and that answer was right; earlier, all 4 runs with a late passage were wrong. One run is weak evidence; a unit test checks the passage now reaches the prompt.
- **Answer context:** over 348 BIRD questions and two passes per prompt, the judge accepts 64.9% of answers with the old prompt and 65.9% with the new one (+1.0 points [−1.3, +3.3]): no measurable change. The 5 earlier misreads are right in both new passes (2 of 5 in the independent old pass). On results over 20 rows the new answers give the total and say the list is partial, which the hand review accepted and the judge does not.

## Consequences

- Published tool-eval and BIRD scores do not change: the fixes leave the eval scores where they were, and earlier reports describe the code of their own commit.
- Agent v1 makes one more LLM call on queries with a planned calculation (about 4% more cost on the Haiku tool eval).
- The judge's rule for long lists (every reference value) is stricter than the hand review. Until that is settled, answer-quality numbers on large results understate answers that give the total and say they are partial.
- A large result may be better returned as a table than as a sentence; that is a product decision, not taken here.
