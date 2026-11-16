# Run-to-run noise (2026-11-16)

The headline evals were each run once. These four reports run them again several times with
nothing changed (same code, prompts, models, effort `low`, split and questions) to measure how
much a single run moves, and to give intervals that include that movement.

| Report | Eval | Model | Items | Runs | Cost |
|---|---|---|---|---|---|
| [`2026-11-16-noise-tools-haiku-test`](2026-11-16-noise-tools-haiku-test/) | agent v1 tool selection, test | Haiku 5.5 | 37 (33 with a scored answer) | 10 | $0.155 |
| [`2026-11-16-noise-tools-sonnet-test`](2026-11-16-noise-tools-sonnet-test/) | agent v1 tool selection, test | Sonnet 5.5 | 37 (33) | 5 | $2.028 |
| [`2026-11-16-noise-bird-haiku-test-350`](2026-11-16-noise-bird-haiku-test-350/) | BIRD Mini-Dev test, `baseline` and `agent_v2` | Haiku 5.5 | 350 | 5 | $1.058 |
| [`2026-11-16-noise-bird-sonnet-test-120`](2026-11-16-noise-bird-sonnet-test-120/) | BIRD Mini-Dev test, stratified sample (`--limit 120 --seed 0`) | Sonnet 5.5 | 120 | 3 | $3.061 |

Total $6.30, every repetition under its own `--budget-usd` cap. All of it is on the test
splits, with frozen configurations, and no choice was made from these numbers.

In the BIRD repetitions' `run.json`, the interpreter path in `parent_cmdline` was shortened to
`.venv/bin/python` (it pointed into a temporary directory of the machine that ran them).

## Method

`python -m evals.noise.repeat` runs the eval's own runner N times, one subprocess and one run id
per repetition (`run-r01` ...), and writes `repeat.json`. `python -m evals.noise.analyze` reads
the repetitions as an items x runs matrix of verdicts and writes `analysis.json`:

- **dispersion**: accuracy of each run, their mean, sample SD, min and max;
- **flips**: items whose verdict is not the same in every run;
- **interval with run noise**: two-stage percentile bootstrap (items resampled, then each
  item's runs resampled; 10,000 resamples, seed 0), i.e. errors clustered by question; shown
  next to the item-only interval of a single run;
- **variance components**: between-item and within-item (run) variance of a verdict, and their
  ICC;
- **paired difference** of two systems on the same items and run indices, same bootstrap;
- **power**: items needed to detect a 3-point paired difference (two-sided alpha 0.05, power
  0.8) with 1 run per item and with N, from the observed variance components, each checked by
  a simulation that keeps the observed item and run structure.

`--reference` is the published single run of the same configuration, scored on the same items;
it is shown next to the repetitions and is not part of them. Tool-eval references are read as
rescored (the scorer the repetitions use).

## What happened on the way: the API credit ran out

The first attempt at these runs stopped being a measurement halfway through: the account's
credit ran out and every call from then on returned `400 invalid_request_error: "Your credit
balance is too low to access the Anthropic API"`. The runners of the time caught any exception
from the agent, recorded it in `error` and scored the empty answer as wrong, so the runs went on
and finished looking clean. The symptoms, as they were seen: a Haiku tool run with `no_tools` at
0% for $0.00003 and 0.6 s per query, and a Sonnet tool run whose `no_tools` answers dropped from
33% to 18% because 19 of 37 came back empty. The BIRD runner was supposed to abort after three
exceptions in a row, but that check only ran without `--workers`, so a 4-worker run finished
with 300 of 350 `agent_v2` pairs failed.

Damaged at that point: tool runs Haiku r04 (30 rows), r05 (74), r06 (9 of a partial run); Sonnet
r01 (19), r02 (74), r03 (19 of a partial run); BIRD Haiku r01 (300 `agent_v2` rows), r02 (93 of
a partial run). Spent: $0.48. The original files are kept outside the repository.

What changed before anything was run again (commits 04ee5dc, c9f5341):

- the LLM clients raise `ProviderError` when the provider cannot answer (after the SDK's
  retries); both runners stop on it, like on the budget cap, without writing the pair, so
  `--resume` runs it again; the BIRD runner's error abort now also works with workers;
- the repetition driver resumes a repetition cut by a provider error with what is left of its
  cap, and refuses a finished run that contains API failures scored as answers;
- the analysis leaves API failures out (an answer that never came back is missing, not wrong)
  and counts them (`provider_failures_left_out`: 0 in every run here).

In the damaged repetitions only the API-failure rows were removed and those queries re-run with
`--resume`; everything else in them is from the first attempt. Every published run was checked
for the same failure: none has an API error among its predictions.

## Results

Accuracy in %; "published" is the single run in the top-level README, on the same items.

**Agent v1 tool selection (exact), test, 37 queries.** Unchanged in every run: 100.0 for
`agent_v1` with both models (0 of 37 items change), 10.8 for `no_tools`.

**Agent v1 answers correct, test, 33 queries with a scored answer:**

| Model | System | Published | Mean of N | SD | Range | CI, one run | CI with run noise | Items changing |
|---|---|---|---|---|---|---|---|---|
| Haiku 5.5 (N = 10) | `agent_v1` | 97.0 | 99.7 | 1.0 | 97.0–100.0 | [100.0, 100.0] | [98.5, 100.0] | 1/33 |
| Haiku 5.5 (N = 10) | `no_tools` | 12.1 | 12.7 | 1.3 | 12.1–15.2 | [3.0, 24.2] | [3.0, 24.6] | 2/33 |
| Sonnet 5.5 (N = 5) | `agent_v1` | 97.0 | 98.2 | 1.7 | 97.0–100.0 | [90.9, 100.0] | [93.3, 100.0] | 1/33 |
| Sonnet 5.5 (N = 5) | `no_tools` | 33.3 | 27.9 | 3.3 | 24.2–33.3 | [18.2, 48.5] | [15.2, 42.4] | 7/33 |

Paired, `agent_v1` - `no_tools` (answers): Haiku +87.0 pts [+75.2, +96.7], Sonnet +70.3
[+55.8, +84.2].

**BIRD Mini-Dev test, execution accuracy:**

| Model | System | Items | Published | Mean of N | SD | Range | CI, one run | CI with run noise | Items changing | ICC |
|---|---|---|---|---|---|---|---|---|---|---|
| Haiku 5.5 (N = 5) | `agent_v2` | 350 | 64.3 | 63.7 | 0.6 | 63.1–64.3 | [58.0, 68.3] | [58.9, 68.6] | 32/350 | 0.91 |
| Haiku 5.5 (N = 5) | `baseline` | 350 | 60.6 | 60.9 | 1.1 | 59.4–62.3 | [56.3, 66.3] | [55.8, 65.9] | 29/350 | 0.92 |
| Sonnet 5.5 (N = 3) | `agent_v2` | 120 | 65.0 | 64.7 | 0.5 | 64.2–65.0 | [55.8, 73.3] | [56.1, 72.8] | 9/120 | 0.89 |
| Sonnet 5.5 (N = 3) | `baseline` | 120 | 60.8 | 61.9 | 1.3 | 60.8–63.3 | [52.5, 70.0] | [53.3, 70.3] | 3/120 | 0.96 |

| Paired, `agent_v2` - `baseline` | Published (one run) | Mean of N [CI with run noise] | Per run | Items for 3 pts, 1 run / N runs |
|---|---|---|---|---|
| Haiku, 350 questions | +3.7 [+0.6, +7.1] | +2.8 [+0.2, +5.4] | +2.0, +1.4, +2.9, +2.9, +4.9 | 747 / 458 |
| Sonnet, 120 of the 350 | +4.2 on these 120 (+3.7 [+0.9, +6.6] on 350) | +2.8 [-2.8, +8.3] | +3.3, +4.2, +0.8 | 969 / 759 |

Reading:

- Every published single-run accuracy lies inside the range of its repetitions. Several sit at
  the edge of it: Sonnet `no_tools` 33.3 is the highest of its five runs, Haiku `agent_v2` 64.3
  the highest of its five.
- On BIRD the run-to-run part is small next to the question-to-question part (ICC about 0.9): a
  question is mostly right or mostly wrong, and 8-9% of the Haiku questions (3-8% of the Sonnet
  sample) change verdict between runs. So the interval of one system's accuracy barely widens
  with run noise, and averaging runs shrinks it little (standard error 2.57 points with one
  Haiku run, 2.47 with five). A paired difference gains more from repeats: the two systems
  mostly agree on a question, so pairing cancels much of the question-to-question part, and
  about half of what is left is run noise (Haiku: 0.041 of 0.085). 747 items are needed to
  detect 3 points with one run, 458 with five.
- The agent v2 gain is smaller than the single run suggested: +2.8 points averaged over five
  Haiku runs, against +3.7 in the published run, which is inside the range of the five. Its
  interval still excludes zero, narrowly (1.5% of resamples at or below zero). On the Sonnet
  sample of 120 the same +2.8 cannot be told apart from zero.
- 350 questions run once are not enough to detect a 3-point difference between two systems
  with 80% power: that takes about 750 with one run, or about 460 with five runs each.
- The tool-selection result does not move at all; the `no_tools` answers move by a few points.
  The tool set is small (37), so its power numbers say little: they are driven by the size of
  the `agent_v1` - `no_tools` gap.
