# Fixes from the failure analysis, measured before and after (2026-11-17)

The [failure analysis](../2026-11-17-failure-analysis/README.md) pointed to three bugs. Each fix
is in commit `f010868`; design in [ADR-011](../../../docs/ADR-011-fixes-from-failure-analysis.md).
Every number below compares the code before and after on the same items, in the same
environment. Total cost $1.73.

| Fix | Before | After |
|---|---|---|
| 1. Calculator input written by the model (agent v1) | 8 of 48 calculator results equal the question's gold value; 8 calls failed on an unfilled `PLACEHOLDER` | 49 of 49 equal the gold value; 0 failed |
| 2. Tool output kept to 6,000 characters, not 1,500 (agent v1) | 4 of 4 runs whose search put the answer past character 1,500 answered wrong (earlier runs) | 1 of 1 such run answered right |
| 3. Answer step sees the SQL and the exact row count (agent v2) | judge accepts 64.9% of answers; 2 of 5 earlier misreads fixed by chance | 65.9% (+1.0 points [−1.3, +3.3]); 5 of 5 in both passes |

## 1. The calculator's input (agent v1)

**Bug.** The planner writes `PLACEHOLDER` for a calculation that depends on SQL results.
`execute_node` filled it with a fixed growth formula, `(b − a) / a × 100`, built from the first
two numbers of the SQL result, whatever the question asked; when the result did not have that
shape it left `PLACEHOLDER` and the calculator failed. The validation step that was meant to fill
it with the model never ran for a calculator planned last.

**Fix.** Just before the calculator runs, the model writes the expression from the question, the
planned reason and the earlier results (prompt `CALC_FILL_PROMPT`, tag `fill`, one extra call
only when a placeholder is planned). An answer that is not a bare arithmetic expression, or
`NONE`, is reported as a failed step instead of being run.

**Measure.** The tool eval's 37 test queries, `agent_v1` only, run 5 times with Haiku 5.5 and 3
times with Sonnet 5.5 at commit `c620be1` (before) and at `f010868` (after), in the same
environment (embedded Qdrant, same demo database hash and embedding model).

| Calculator calls on the 12 `sql_math` queries | Before (8 runs) | After (8 runs) |
|---|---|---|
| calls | 48 | 49 |
| result equals the question's gold value (1%) | 8 | **49** |
| growth formula on a question that is not about growth | 32 | 0 |
| unfilled `PLACEHOLDER`, failed call | 8 | 0 |

The answers were already right in every run before and after (answer check 100%): the
respond step ignored the wrong calculator output or did the arithmetic itself, as in the earlier
runs (15 of 92 calculator results right in the 15 noise repetitions). So this fix does not move a
score; it removes tool outputs that were wrong without failing, which nothing in the eval
caught. Reports: [`2026-11-17-fix-tools-haiku-before`](../2026-11-17-fix-tools-haiku-before/),
[`-sonnet-before`](../2026-11-17-fix-tools-sonnet-before/),
[`-haiku-after`](../2026-11-17-fix-tools-haiku-after/),
[`-sonnet-after`](../2026-11-17-fix-tools-sonnet-after/). The extra call adds about 4% to the
Haiku runs' cost.

## 2. The tool output cut (agent v1)

**Bug.** Each tool output was cut to 1,500 characters in the prompts, without saying so. A
document search returns about 2,500 characters, and where the passage with the answer lands
depends on the search query the agent writes.

**Fix.** 6,000 characters, and a cut says how many characters were left out.

**Measure.** Only `sqldoc_03` (the return period of a tablet; the electronics exception is 15
days) is affected, and only when the search puts that passage late. In the earlier 15 noise
repetitions it landed past character 1,500 in 4 runs, and those are exactly the 4 wrong answers. In the 16 runs
here it landed past 1,500 once, after the fix (character 1,812), and the answer was right; before
the fix it never landed that late, so this comparison is one run. A unit test checks that text at
character 1,800 now reaches the prompt.

## 3. The answer step (agent v2)

**Bug.** The answer step saw the question and up to 20 result rows, not the query, and `>1000`
for large results. It doubted filters the query had applied ("I can't tell whether the colours
match") and guessed the size of large results.

**Fix.** The prompt now has the final SQL, the exact row count when the fetch stopped at 1,000
(`SELECT COUNT(*)` over the query, under the guard), and an instruction to give the total and say
when a list is partial. Generate and check prompts are unchanged (snapshot test).

**Measure.** Agent v2's answers for the 350 BIRD test questions of Haiku run `r01` were rebuilt
from the run's own SQL, twice with the old prompt and twice with the new one, and graded by the
calibrated judge (`evals/failures/compare_answers.py`; two-stage bootstrap over questions and
passes; 2 questions without a reference left out). Output:
[`answer-step.json`](answer-step.json), passes in [`answer-step/`](answer-step/).

| Questions | n | Old prompt | New prompt | Difference |
|---|---|---|---|---|
| all | 348 | 64.9% | 65.9% | +1.0 [−1.3, +3.3] |
| EX correct | 221 | 89.6% | 90.0% | +0.5 [−2.3, +3.4] |
| EX wrong | 127 | 22.0% | 24.0% | +2.0 [−1.6, +5.5] |
| result over 20 rows | 33 | 4.5% | 9.1% | +4.5 [−4.5, +15.2] |
| earlier `answer_misread` failures | 5 | 2 of 5 in the independent pass | 5 of 5 in both passes | |

- **No measurable change overall.** The gain is inside the run-to-run noise.
- **The misreads are fixed:** the 5 questions where the answer contradicted a correct result are
  right in both passes with the new prompt. Five questions is a small set, and they were chosen
  because the first old-prompt pass got them wrong, so only the second old pass (2 of 5) is a fair
  comparison.
- **Large results: the judge and the reviewer disagree.** With the new prompt the answer to a
  157-row question gives the exact total, lists the first rows and says the list is partial. The
  hand review accepted that kind of answer (`r-06`); the judge does not, because its rule asks for
  every reference value. Whether the new answers are better there needs a human check, or a
  product decision to return large results as a table rather than a sentence.

## Environment

Docker could not pull the Qdrant image here, so these runs use Qdrant's embedded local mode
(`QDRANT_URL=file:<dir>`, commit `c620be1`); the before runs used the same mode, so the
comparison is like for like, but it is not the Qdrant 1.12.5 server of the earlier reports. The
before runs ran from a git worktree at `c620be1` with links to the shared virtualenv and BIRD
data, which marks them `dirty` in `run.json`.
