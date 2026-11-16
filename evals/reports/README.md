# Reports

Each subdirectory is one committed eval run: a `summary.json` and/or `run.json` with the provenance (`git`, LLM config), the predictions, and every LLM call with tokens and cost. Most runs also have a README with the exact command; the four `2026-10-30-v1-tools-*` runs keep it in `run.json`. The top-level README quotes its numbers from these files.

## `git.dirty` in the provenance

Several runs record `"dirty": true` next to their commit: `2026-10-19-haiku-test-350`, `2026-10-19-sonnet-test-350`, `2026-10-19-sonnet-dev-20-pilot`, `2026-10-19-a5-trace-check`, one of the two records in `2026-10-19-ci-reference-dev-30`, and some of the records in the four `2026-10-30-v1-tools-*` runs. The flag comes from `git status --porcelain` (`evals/provenance.py`): it is set by any modified tracked file or any untracked file outside `.gitignore` (run outputs go to the ignored `evals/results/`, so they do not set it). What was uncommitted at the time was not recorded, so the commit in those reports identifies the code only up to that. These runs were not repeated on a clean tree. Independently of the flag, each report records the LLM config (model id, effort, pricing, prompt caching) and the split (the text-to-SQL runs also their sampling and question ids), and the BIRD data is pinned to a Hugging Face revision and checked by sha256 (`evals/text2sql/prepare.py`).
