# Answer judge calibration (dev, 50 answers)

Calibration of the LLM judge of `evals/judge/` against the owner's hand labels, and of both
against execution accuracy (EX). Method and limits: `docs/ADR-009-judge-and-ci-evals.md`.

- **Answers** (`items_run/`, run id `judge-set-dev-50`, commit `eac2a34`, clean tree): 50 dev
  questions stratified by database and difficulty (seed 0), answered by agent v2 with Claude
  Haiku 5.5 and the respond step on. 36 of 50 final queries are EX-correct. Cost $0.0274.
- **Human labels** (`labels.jsonl`): Eduardo Garcia Timermans, 2026-10-19, on a private page
  that showed each question, its external knowledge, the gold result preview (columns, up to 10
  rows, row count) and the answer; no SQL, no EX and no judge verdict. 34 correct, 16
  incorrect, 0 unsure. The labeler's notes are not committed because some quote gold rows.
- **Judges** (`haiku/`, `sonnet/`): one call per answer with the frozen judge prompt, Claude
  Haiku 5.5 (run `judge-haiku-dev-50`, commit `eac2a34`, $0.0052) and Claude Sonnet 5.5 (run
  `judge-sonnet-dev-50`, commit `81439f2`, $0.0862; the judge code is the same at both commits,
  only documentation changed in between). 0 unparsed verdicts. The rationales are not committed
  because they quote gold rows.

## Results

95% percentile bootstrap CIs over the 50 items (B = 10000, seed 0).

| Comparison | n | Accuracy [95% CI] | Cohen's kappa [95% CI] |
|---|---|---|---|
| Judge (Haiku 5.5) vs human | 50 | 0.98 [0.94, 1.00] | 0.95 [0.84, 1.00] |
| Judge (Sonnet 5.5) vs human | 50 | 0.98 [0.94, 1.00] | 0.95 [0.85, 1.00] |
| Human vs EX | 50 | 0.84 [0.74, 0.94] | 0.62 [0.35, 0.83] |
| Judge (Haiku 5.5) vs EX | 50 | 0.86 [0.76, 0.94] | 0.66 [0.40, 0.87] |
| Judge (Sonnet 5.5) vs EX | 50 | 0.86 [0.76, 0.94] | 0.67 [0.42, 0.87] |

Share of answers marked correct: human 68% [54, 80], EX 72% [60, 84], Haiku judge 70% [56, 82],
Sonnet judge 66% [52, 78]. EX - human: +4 points [-6, +16] (paired bootstrap).

Where the raters differ (items in `items.jsonl`):

- **Each judge disagrees with the human once**, on long results the 10-row preview cannot show
  in full. Haiku accepted an answer that gives the right count of 75 molecules but lists only
  20 of them (j-33; human: incorrect). Sonnet rejected an answer that hedges but lists the 8
  distinct values of a 32-row result made of repeats (j-10; human: correct).
- **EX-correct, answer wrong (5 items).** The query was right and the respond step lost it:
  four answers name only some rows of a long list (j-14, j-21, j-33, j-37), one turns a value
  that is already a percentage (0.20) into 20.2% (j-17). Sonnet catches all 5, Haiku 4 (it
  accepts j-33).
- **EX-wrong, answer right (3 items).** EX is stricter than a reader: full names in one column
  instead of two (j-31), a sum that differs from the gold one only by floating-point
  rounding (j-32), and distinct values against a result with repeats plus an extra column (j-10).

Reading: on this set either judge reproduces the owner's reading of the answers almost exactly
(one disagreement in 50, on a borderline case), so the cheaper Haiku judge ($0.0001 per answer)
is enough. Answer correctness and EX measure different things: they disagree on 8 of 50 items,
in both directions, and the respond step's main failure is shortening long lists. One labeler,
50 items, dev only: the intervals are wide and nothing here is a test-set number.

## Reproduce

```
uv run python -m evals.judge.build_set --llm-config configs/llm/claude-haiku-5-5.yaml \
  --run-id judge-set-dev-50 --budget-usd 0.50 --launched-by <who>
uv run python -m evals.judge.run_judge --items evals/results/judge/judge-set-dev-50 \
  --llm-config configs/llm/claude-haiku-5-5.yaml --run-id judge-haiku-dev-50 \
  --budget-usd 0.25 --launched-by <who>
uv run python -m evals.judge.run_judge --items evals/results/judge/judge-set-dev-50 \
  --llm-config configs/llm/claude-sonnet-5-5.yaml --run-id judge-sonnet-dev-50 \
  --budget-usd 1.00 --launched-by <who>
uv run python -m evals.judge.import_labels --items evals/results/judge/judge-set-dev-50 \
  --labels labels.json
uv run python -m evals.judge.agreement --items evals/results/judge/judge-set-dev-50 \
  --judgements evals/results/judge/judge-haiku-dev-50     # and judge-sonnet-dev-50
uv run python -m evals.judge.save_report --items evals/results/judge/judge-set-dev-50 \
  --judge haiku=evals/results/judge/judge-haiku-dev-50 \
  --judge sonnet=evals/results/judge/judge-sonnet-dev-50 \
  --out evals/reports/2026-10-19-judge-calibration-dev-50
```

The agreement can be recomputed from this directory alone:
`uv run python -m evals.judge.agreement --items evals/reports/2026-10-19-judge-calibration-dev-50
--judgements evals/reports/2026-10-19-judge-calibration-dev-50/haiku --out <file>` gives the
same numbers as `haiku/agreement.json` (checked).
