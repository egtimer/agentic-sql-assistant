# Guardrails: SQL adversarial evaluation

Command (from `agentic-sql-assistant/`, needs the BIRD data from `make bird-data` for the gold queries and the BIRD-based cases; no LLM, no API key, about 3 minutes):

```bash
uv run python -m evals.guardrails.run_sql --out evals/reports/2026-10-19-guardrails-sql/
```

Environment: Python 3.12.11, SQLite 3.40.1, 152 attacks (`evals/guardrails/attacks.jsonl`), 40 benign queries (`benign.jsonl`) and the 500 BIRD Mini-Dev gold queries, each run against four guards. Files: `results.jsonl` (one line per guard and case: outcome, layer, side effects, elapsed time, peak memory growth for resource cases) and `summary.json`.

Guards: `legacy_v1` is the v1 `sql_query` tool before this step (regex blocklist, read-write connection, 50-row fetch); `readonly_only` is a `mode=ro` connection (what agent v2 and the scorer used before); `sql_guard` is `src/sql_guard/`; `no_lexical` is `sql_guard` without layer 1 (ablation: authorizer, connection limits and resources only).

Every case ran on a disposable copy of its database inside a temporary directory that was the working directory for that case. The gold queries ran on one set of copies per guard run, with the copies hashed before and after. Cases use relative file names only.

## Result

```
Attacks: blocked/cases, then ran = attacks whose statement executed or changed something, fx = cases with a side effect (resource: bounded/cases)
category         legacy_v1               readonly_only           sql_guard               no_lexical              
write            22/29 ran=7 fx=5        0/29 ran=5              29/29                   28/29 ran=1             
cte_write        7/7                     0/7                     7/7                     7/7                     
multi_statement  15/15                   0/15                    15/15                   15/15                   
file_access      0/13 ran=10 fx=7        0/13 ran=10 fx=7        13/13                   13/13                   
metadata         0/17 ran=17             0/17 ran=17             17/17                   17/17                   
transaction      0/9 ran=4               0/9 ran=4               9/9                     9/9                     
obfuscation      18/22 ran=3 fx=2        0/22 ran=3 fx=2         22/22                   14/22                   
policy           0/22 ran=22             0/22 ran=22             22/22                   22/22                   
resource         14/18 bounded           14/18 bounded           18/18 bounded           18/18 bounded           

benign false blocks / cases       legacy_v1               readonly_only           sql_guard               no_lexical              
own benign cases                  15/40                   0/40                    0/40                    0/40                    
BIRD gold queries                 0/500                   0/500                   0/500                   0/500                   
```

- `blocked/cases`: attacks refused by the guard itself. `readonly_only` has no blocking layer; its engine rejects writes with an error, which is not counted as a block. `ran`: attacks whose statement executed without an error or changed something (for `policy` and `metadata` a leak, for `file_access` a file written). `fx`: cases with a side effect, meaning the database copy's SHA-256 changed or a new file appeared in the directory.
- `legacy_v1` side effects (14 cases): `PRAGMA user_version`/`writable_schema`/`journal_mode`/`auto_vacuum`, `VACUUM`, `ANALYZE` and `REINDEX` change the database file; `ATTACH` and `VACUUM INTO` create files (7 file-access cases, 2 obfuscated ones). `readonly_only` still allows the 9 file cases (9 cases with side effects).
- Resource cases: `legacy_v1` and `readonly_only` have no timeout of their own; the harness stopped them at 3 s. Cases that exceed 3 s plus 2 s or grow memory by more than 256 MB are not bounded. Largest resident-memory growth in a resource case: legacy_v1 3228 MB (the doubling-string CTE, 8.9 s), readonly_only 3208 MB, sql_guard 0 MB. Running this report on a small machine therefore needs a few GB of free memory for those two guards.
- Benign: `legacy_v1` falsely blocks 15 of 40 hand-written benign queries (the `REPLACE()` function, keywords inside string literals and quoted identifiers, `;` inside a string, comments with keywords). `sql_guard` blocks none of the 40 or of the 500 gold queries, and returns the same rows as `readonly_only` on all of them (`rows_differ_from_readonly_only` = 0 and 0).
- `no_lexical` shows what layer 1 adds: 8 obfuscation cases (full-width, Cyrillic, zero-width and no-break-space lookalikes, NUL, quoted keywords, a parenthesised statement) fail with a syntax error from the engine instead of being blocked, and `DROP TRIGGER IF EXISTS pwned` (the trigger does not exist, so the statement compiles to nothing the authorizer sees) runs as a no-op. None of them changes anything.
- A side effect check on the gold queries covers each database copy (hash) and the directory; the `-wal`/`-shm` files that SQLite creates for the WAL-mode `card_games` database on a read-only connection are not counted.
- Timings and memory are measured on one run and vary; outcomes are deterministic.
