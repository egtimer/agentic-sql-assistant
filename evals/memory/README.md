# Memory evaluation data

`followups.jsonl`: 30 two-turn conversations for the short-term (conversation) memory eval.
Each pair rewrites one BIRD Mini-Dev **dev** question (`source_question_id`) into a first turn
that sets the context (entity, filter or set) and a second, elliptical turn ("How many of them
...", "What is its ...") that can only be answered with that context. The second turn is scored
with execution accuracy against the original question's gold SQL; the first turn is not scored.
The original question's external knowledge (`evidence`) is given in every condition, so the
conversation history is the only variable.

Provenance: written by hand on 2026-10-19 from the question text only (gold SQL not used for wording), restricted to dev questions with a clear
entity or filter to move into the first turn. 30 is small: report results with CIs and treat
them as indicative. BIRD Mini-Dev is CC BY-SA 4.0; only question ids are referenced here and the
rewritten wording is a derivative under the same license.

## Runners

- `run_followups.py` reads `followups.jsonl` (conditions `turn2_alone`, `with_history`, `original`).
- `run_longterm.py` builds its data from the evidence of BIRD questions (no file here): it teaches the evidence to the agent and compares `no_memory`, `memory` and `oracle`.
- `smoke.py` runs both with a scripted client, offline.

Both runners need a new `--run-id`, write to `evals/results/memory/<run-id>/` (gitignored) and refuse an existing directory. Method and limits: `docs/ADR-007-memory.md`.
