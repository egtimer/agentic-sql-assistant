# MCP server demo on Chinook

A scripted MCP client session against the real server over stdio (`python -m src.mcp_server.demo`, LLM config `configs/llm/claude-haiku-5-5.yaml`). Every tool call and its structured result as returned to the client. Total: 10 LLM calls, $0.0017.

## 1. Which databases can I use?

`list_databases` {}

```json
{
  "databases": [
    {
      "name": "Chinook",
      "tables": 11
    }
  ]
}
```

## 2. Run my own read-only query

`run_sql` {"sql": "SELECT COUNT(*) AS tracks FROM Track", "database": "Chinook"}

```json
{
  "database": "Chinook",
  "ok": true,
  "blocked": false,
  "error": null,
  "columns": [
    "tracks"
  ],
  "rows": [
    [
      3503
    ]
  ],
  "row_count": 1,
  "truncated": false
}
```

## 3. Ask a question

`ask` {"question": "Which 3 genres have the most tracks?", "database": "Chinook", "thread_id": "demo-thread"}

```json
{
  "database": "Chinook",
  "answer": "The three genres with the most tracks are Rock, Latin, and Metal.",
  "sql": "SELECT g.Name\nFROM Genre g\nJOIN Track t ON t.GenreId = g.GenreId\nGROUP BY g.GenreId, g.Name\nORDER BY COUNT(t.TrackId) DESC\nLIMIT 3",
  "columns": [
    "Name"
  ],
  "rows": [
    [
      "Rock"
    ],
    [
      "Latin"
    ],
    [
      "Metal"
    ]
  ],
  "row_count": "3",
  "blocked": false,
  "error": null,
  "llm_calls": 3,
  "cost_usd": 0.001174,
  "trace_id": "14f37447-7208-46e5-bb6d-ea2a56369d29"
}
```

## 4. Follow-up in the same thread

`ask` {"question": "And how many tracks does the second one have?", "database": "Chinook", "thread_id": "demo-thread"}

```json
{
  "database": "Chinook",
  "answer": "The second one has 579 tracks.",
  "sql": "SELECT COUNT(t.TrackId) AS TrackCount\nFROM Genre g\nJOIN Track t ON t.GenreId = g.GenreId\nGROUP BY g.GenreId, g.Name\nORDER BY TrackCount DESC\nLIMIT 1 OFFSET 1",
  "columns": [
    "TrackCount"
  ],
  "rows": [
    [
      "579"
    ]
  ],
  "row_count": "1",
  "blocked": false,
  "error": null,
  "llm_calls": 3,
  "cost_usd": 0.000294,
  "trace_id": "c4df69a8-3b36-4f7d-b381-8c3007aea5ed"
}
```

## 5. Teach a definition

`teach` {"text": "For me, a long track is a track that lasts more than 6 minutes (Milliseconds > 360000).", "database": "Chinook", "user_id": "demo-user"}

```json
{
  "database": "Chinook",
  "user_id": "demo-user",
  "stored": [
    {
      "term": "long track",
      "definition": "a track that lasts more than 6 minutes, i.e. Milliseconds > 360000"
    }
  ],
  "llm_calls": 1,
  "cost_usd": 5.6e-05
}
```

## 6. Ask with the taught definition, in a new session

`ask` {"question": "How many long tracks are there?", "database": "Chinook", "user_id": "demo-user"}

```json
{
  "database": "Chinook",
  "answer": "There are 623 long tracks.",
  "sql": "SELECT COUNT(*) AS long_track_count\nFROM Track\nWHERE Milliseconds > 360000",
  "columns": [
    "long_track_count"
  ],
  "rows": [
    [
      "623"
    ]
  ],
  "row_count": "1",
  "blocked": false,
  "error": null,
  "llm_calls": 3,
  "cost_usd": 0.000151,
  "trace_id": "1adf364d-8153-44f4-9217-ca367b460b8b"
}
```

## 7. A statement the guard refuses

`run_sql` {"sql": "DELETE FROM Track", "database": "Chinook"}

```json
{
  "database": "Chinook",
  "ok": false,
  "blocked": true,
  "error": "blocked: Write operations are not allowed. Only SELECT queries are permitted",
  "columns": [],
  "rows": [],
  "row_count": 0,
  "truncated": false
}
```

## 8. Forget the definition

`forget` {"database": "Chinook", "user_id": "demo-user"}

```json
{
  "database": "Chinook",
  "user_id": "demo-user",
  "deleted": [
    "long track"
  ]
}
```

