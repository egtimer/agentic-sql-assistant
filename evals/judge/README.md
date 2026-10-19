# Answer judge and its calibration

Execution accuracy (EX) scores the SQL; a user reads the answer. This folder grades the
agent's natural-language answer with an LLM judge and measures how well the judge agrees with
a human who labelled the same answers. Method and limits: `docs/ADR-009-judge-and-ci-evals.md`.

All paid commands need `--budget-usd` and `--launched-by`, write to `evals/results/judge/<run-id>/`
(gitignored, a run id must be new) and write `run.json` first, so a partial run can be traced.

## 1. Build the item set (paid, about 3 LLM calls per item, at most 4)

```
uv run python -m evals.judge.build_set --llm-config configs/llm/claude-haiku-5-5.yaml \
    --run-id judge-set-dev-50 --budget-usd 0.50 --launched-by owner
```

50 dev questions, stratified by database and difficulty (`--n`, `--seed 0`), answered by agent v2
with the respond step on. Output:

- `items.jsonl`: full records. Holds the gold result preview and the EX of the final SQL (BIRD
  content, CC BY-SA): never commit it.
- `items_public.jsonl`: `item_id` (`j-01`...), `question_id`, `db_id`, `difficulty`, `question`,
  `evidence`, `answer`, `final_sql`, `result_preview` (columns and up to 10 rows). No gold SQL, no
  gold rows, no EX. This is the file that may be committed.

## 2. Judge the answers (paid, 1 LLM call per item)

```
uv run python -m evals.judge.run_judge --items evals/results/judge/judge-set-dev-50 \
    --llm-config configs/llm/claude-haiku-5-5.yaml --run-id judge-haiku-dev-50 \
    --budget-usd 0.25 --launched-by owner

uv run python -m evals.judge.run_judge --items evals/results/judge/judge-set-dev-50 \
    --llm-config configs/llm/claude-sonnet-5-5.yaml --run-id judge-sonnet-dev-50 \
    --budget-usd 1.00 --launched-by owner
```

`judgements.jsonl`: `item_id`, `label` (`correct`, `incorrect`, `unparsed`, `error` for an API
failure, `no_reference` when the gold query failed), `rationale`, tokens, `cost_usd`. The judge
sees the question, the external knowledge, the gold result preview and the answer; never the SQL.

## 3. Import the human labels (free)

The labels are collected on a private page that shows each item's question, external knowledge,
the same 10-row reference preview the judge sees and the answer (no SQL, no EX, no judge verdict),
and are written out as `labels.json`:

```
{"labeler": "name", "labels": {"j-01": {"label": "correct", "note": "..."}, "j-02": {"label": "unsure"}}}
```

`label` is `correct`, `incorrect` or `unsure`; `note` is optional.

```
uv run python -m evals.judge.import_labels --items evals/results/judge/judge-set-dev-50 \
    --labels path/to/labels.json
```

It checks that every id is a known item, that labels are allowed and that every item has one
(`--allow-partial` relaxes the last check), and writes `labels.jsonl` into the items run.

## 4. Agreement (free)

```
uv run python -m evals.judge.agreement --items evals/results/judge/judge-set-dev-50 \
    --judgements evals/results/judge/judge-haiku-dev-50
```

Prints a table and writes `agreement.json` into the judge run: accuracy, Cohen's kappa and the
confusion matrix for judge vs human, human vs EX and judge vs EX, each with a 95% percentile
bootstrap CI (items resampled, B = 10000, seed 0), plus the counts of `unsure` and `unparsed`
items (both are left out of the comparisons they affect).

## Regression gate for the weekly API eval

```
uv run python -m evals.text2sql.check_regression \
    --reference evals/reports/2026-10-19-ci-reference-dev-30 \
    --current evals/results/text2sql/<run id> --system agent_v2 \
    --max-drop-pts 10 --max-cost-ratio 2.0
```

Exit code 1 when EX dropped by more than `--max-drop-pts` and the paired CI of the change lies
below 0, or when the cost per question is above `--max-cost-ratio` times the reference.
