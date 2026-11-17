# Failure analysis (2026-11-17)

Where the agents go wrong, by cause, with every failure's trace in this folder. Two sources:

- **BIRD, agent v2, Haiku 5.5**: repetition `r01` of the run-to-run noise runs
  ([`2026-11-16-noise-bird-haiku-test-350/run-r01`](../2026-11-16-noise-bird-haiku-test-350/)),
  350 test questions, joined with a final-answer pass (below);
- **tool eval, agent v1**: all 15 noise repetitions (10 Haiku, 5 Sonnet) on the 37 test queries,
  555 runs with their per-step traces.

Nothing here changes a published number: the SQL, EX and tool scores are the runs' own.

## Final-answer pass ($0.053)

EX scores the SQL; a user reads the answer. The noise runs generated agent v2's answer and then
dropped it (fixed in 7c93f44), so `evals/failures/final_answer.py` rebuilt it without re-running
the agent: it re-executes the run's final SQL read-only, makes the respond call with the agent's
own prompt, and grades the answer with the Haiku judge of [ADR-009](../../../docs/ADR-009-judge-and-ci-evals.md)
(kappa 0.95 against the owner's labels on 50 dev answers). 700 calls, cap $0.15, spent $0.053.
Output: [`final-answer/`](final-answer/) (`answers_public.jsonl` has the answers and labels; the
judge's rationales and the reference rows quote BIRD gold content and stay out of the repo).

| | Judge: answer correct | Judge: answer wrong |
|---|---|---|
| EX correct | 199 | 22 |
| EX wrong | 28 | 99 |

Two questions have no reference (the gold query times out) and are left out. On the other 348,
EX and the judge disagree on 50 (14%), in both directions: 22 right queries whose answer is
judged wrong, and 28 queries EX rejects whose answer the judge accepts. Which of them a person
would accept is what the hand review below measures.

## Categories and rules

`evals/failures/taxonomy.py` assigns categories with rules that read only the record (no LLM,
no labels). A record is a failure when the user would get a wrong answer: BIRD, EX wrong or the
judge says the answer is wrong; tool eval, the answer check fails or the tool selection is not
exact. Every rule that fires is kept as a tag; the primary category is the first tag in this
order.

| Category | Rule |
|---|---|
| `guard_refusal` | final SQL error starts with `blocked:` |
| `schema_hallucination` | `no such column` / `no such table` |
| `bad_arguments` | SQL syntax error, calculator parse error, or a calculator input left as `PLACEHOLDER` |
| `silent_tool_failure` | a step failed or returned no rows/chunks and the answer does not say so |
| `wrong_tool` | `wrong_tools` or `missing_tools` not empty |
| `loop_or_extra_steps` | the two fixes used up, or the same call repeated with the same input |
| `answer_on_large_result` | EX correct, judge wrong, the result has more than the 20 rows the respond step sees |
| `tool_output_cut` | the expected values are in a tool's output only after the 1,500 characters agent v1 keeps in its prompt |
| `answer_misread` | EX correct (or the expected values are in what the agent saw), answer wrong |
| `retrieval_miss` | a doc search ran and its output does not hold the expected values |
| `answer_ok_ex_wrong` | EX wrong, judge correct |
| `wrong_output_shape` | EX wrong, more or fewer columns than the gold result |
| `wrong_query_logic` | EX wrong, same columns, different values or rows |
| `other` | no rule fired |

The reviewer has two more labels no rule can give: `not_a_failure` and `benchmark_issue` (the
gold query or the question is wrong or ambiguous).

## Results (rules; hand review pending)

| Category | BIRD agent v2 (348) | Tool eval agent v1 (555 runs) |
|---|---|---|
| `wrong_query_logic` | 82 | |
| `answer_ok_ex_wrong` | 28 | |
| `answer_on_large_result` | 17 | |
| `wrong_output_shape` | 15 | |
| `answer_misread` | 5 | |
| `loop_or_extra_steps` | 2 | |
| `tool_output_cut` | | 4 |
| **Failures** | **149** | **4** |

What the rules found:

- **No tool failures in agent v2's SQL.** None of the 350 questions has a SQL error, a schema
  hallucination or a guard refusal in any attempt; 24 questions used the review-and-fix loop and
  4 used both fixes. The failures are wrong queries that run (82 + 15) and the answer step (22).
- **The answer step loses what it cannot see.** 17 of the 22 right queries with a wrong answer
  return more rows than the 20 the respond step is shown, and the answer reports a partial list
  or a count it cannot know ("more than 1,000").
- **All 4 tool-eval failures are one cause, and it is not retrieval.** `sqldoc_03` (return period
  of a tablet) failed in 1 Haiku and 3 Sonnet runs. The document search did return the exception
  for electronics (15 days), at about character 1,800 of its output, but agent v1 cuts each tool
  output to 1,500 characters before the model sees it (`src/agent/nodes.py`), so the agent
  answered with the general 30 days. Sonnet said the excerpt was cut off.
- **A recovered bug: the calculator gets a literal `PLACEHOLDER`.** In 19 of 555 runs (two
  queries, `sqlm_02` and `sqlm_05`), the plan's calculator input stayed `PLACEHOLDER` and the
  tool failed. Every time, the answer said so and did the arithmetic itself, and it was right,
  so these are not failures; they are counted in [`detectors.json`](detectors.json)
  (`tags_on_items_answered_correctly`). Neither this nor the 1,500-character cut is fixed here:
  fixing them changes agent v1, so it needs its own before/after run.

## Hand review

40 failures were sampled for the author to label (up to 5 per category, then topped up from
the largest, the same question at most once per category, seed 0):
[`review/review_sheet.md`](review/review_sheet.md) (no reference rows) and
`review/labels_template.csv`. `evals/failures/review.py` validates the filled labels and writes
`review/labels.csv` and `review/agreement.json`: accuracy and Cohen's kappa of rule vs human with
95% bootstrap CIs, the confusion matrix, and how many rule-flagged failures the reviewer did not
count as agent failures.

## Files

- [`classification.jsonl`](classification.jsonl): one row per failure (source, item, primary
  category, tags, EX, judge label, result shape vs gold).
- [`traces/`](traces/): one JSON per failure: question, attempts, reviewer verdicts, final SQL
  or tool steps, answer, labels. No gold SQL, no gold rows.
- [`detectors.json`](detectors.json): what every rule finds on all items, failures or not.
- [`final-answer/`](final-answer/): the final-answer pass.

## Reproduce

```
make bird-data
uv run python -m evals.failures.final_answer \
    --report 2026-11-16-noise-bird-haiku-test-350/run-r01 \
    --llm-config configs/llm/claude-haiku-5-5.yaml --run-id final-answer-haiku-r01 \
    --budget-usd 0.15 --workers 4 --launched-by owner
uv run python -m evals.failures.classify \
    --bird-report 2026-11-16-noise-bird-haiku-test-350/run-r01 \
    --answers evals/results/failures/final-answer-haiku-r01 \
    --tools-report 2026-11-16-noise-tools-haiku-test \
    --tools-report 2026-11-16-noise-tools-sonnet-test \
    --out evals/reports/2026-11-17-failure-analysis
```

The respond and judge calls are not deterministic, so a new final-answer pass can move a few
labels; the classification is deterministic given the pass.
