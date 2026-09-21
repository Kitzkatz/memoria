---
title: API
description: HTTP API for Memoria — endpoints, request shapes, and common flows.
---

# API

Memoria exposes an HTTP API through FastAPI. The schema is generated at startup from the running app.

**Live schema:**

```text
http://localhost:8000/docs     Swagger UI
http://localhost:8000/redoc    ReDoc
```

Start the server with either:

```bash
python main.py --host 0.0.0.0 --port 8000
```

or:

```bash
python cli.py serve --port 8000
```

Both launch the same application. The schema at `/docs` is always current — it's derived from the route signatures, so it cannot go stale the way a hand-written reference can. This page documents the **concepts**, gives working examples, and lists the routers that make up the API. For exact field types and validation rules, read `/docs`.

---

## Concepts

### A memory is raw text

The unit of record is the raw text you send. Metadata, entities, relationships, and embeddings are derived and attached to that raw memory. Retrieval returns the memory itself, not a decomposed fragment of it.

### `store` returns an ID

```json
{"status": "stored", "id": 42}
```

The ID is what later endpoints and the analyzer refer to. IDs are assigned by the database.

### `query` returns ranked candidates

```json
{
  "query": "What does Kevin like?",
  "count": 3,
  "results": [
    {
      "rank": 1,
      "id": 42,
      "text": "Kevin likes ramen.",
      "metadata": {},
      "score": 0.94,
      "final_score": 0.94,
      "diagnostics": {}
    }
  ],
  "diagnostics": {}
}
```

`score` is the retrieval-layer score. `final_score` is the score after ranking, if ranking is enabled. When `RANKING_ENABLED` is false they are the same.

### Diagnostics are optional but useful

`diagnostics` on the response carries retrieval-layer metadata: which workers were submitted, which completed, scheduler wait times, and (when ranking is on) per-candidate signal breakdowns. Set `RANKER_DIAGNOSTICS=true` to populate the ranking diagnostics.

### No authentication

Memoria binds to `0.0.0.0` by default and has no auth layer. Run it on localhost or a trusted network. If you expose it, put it behind a reverse proxy that handles auth.

---

## Common Flows

### Store and query

```bash
curl -X POST http://localhost:8000/memory/store \
  -H "Content-Type: application/json" \
  -d '{"text": "Kevin Johnson likes ramen."}'

curl -X POST http://localhost:8000/memory/query \
  -H "Content-Type: application/json" \
  -d '{"text": "What does Kevin like?"}'
```

### Batch store

```bash
curl -X POST http://localhost:8000/memory/batch_store \
  -H "Content-Type: application/json" \
  -d '{"texts": ["first", "second", "third"]}'
```

### Chat

```bash
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{"text": "What does Kevin like?"}'
```

Chat retrieves relevant memories and passes them as context to the configured LLM. `POST /chat/raw` skips retrieval and sends the prompt directly.

### Maintenance check

```bash
curl http://localhost:8000/maintenance/verify
```

Returns whether the SQLite row count and FAISS vector count agree. If `synced` is false, run `POST /maintenance/rebuild_index`.

---

## Routers

The API is composed of five routers plus the root app.

### `/memory`

Store and retrieve memories.

| Method | Path | Body | Returns |
|--------|------|------|---------|
| `POST` | `/memory/store` | `{"text": "..."}` | `{"status": "stored", "id": N}` |
| `POST` | `/memory/query` | `{"text": "..."}` | `{"query", "count", "results", "diagnostics"}` |
| `POST` | `/memory/batch_store` | `{"texts": ["...", "..."]}` | `{"stored": N, "ids": [...]}` |
| `POST` | `/memory/store_many` | `{"texts": [...]}` | Alias for `/memory/batch_store` |
| `POST` | `/memory/reflect` | — | `{"reflection": ...}` |
| `POST` | `/memory/test_store` | `{"text": "..."}` | `{"id": N, "status": "stored"}` |
| `GET`  | `/memory/stats` | — | `{"memory_count": N, "goals": N}` |

> **Note on request shape:** `/memory/query` takes a body with a `text` field, not a `query` field. This is intentional at the code level but easy to get wrong from the docs — use `/docs` to confirm the schema before writing client code.

### `/chat`

Chat and raw LLM passthrough.

| Method | Path | Body | Returns |
|--------|------|------|---------|
| `POST` | `/chat/` or `/chat/chat` | `{"text": "...", "top_n": N}` | `{"input", "response", "status"}` |
| `POST` | `/chat/raw` | `{"text": "..."}` | `{"input", "response", "status"}` |
| `GET`  | `/chat/history` | — | `{"history": [...]}` if enabled |

`POST /chat/` and `POST /chat/chat` are the same handler. `top_n` is optional.

`/chat/history` returns an empty list with a `"Chat history not enabled"` message unless `MemoryController` exposes a `chat_history` attribute. Whether that's wired up depends on the current controller — check `/docs` and the controller source before relying on it.

### `/debug`

Diagnostics. Read-only.

| Method | Path | Returns |
|--------|------|---------|
| `GET` | `/debug/stats` | DB rows, vector rows, embedding cache size |
| `GET` | `/debug/latest?limit=N` | Latest N memories |
| `GET` | `/debug/health` | Full diagnostics report |
| `GET` | `/debug/probe` | Counts, sync status, 5-sample, health |
| `GET` | `/debug/cache` | Embedding cache size, path, max size |
| `GET` | `/debug/graph` | Graph statistics |
| `GET` | `/debug/router` | Memory types and default |

`/debug/health` and `/debug/probe` return a `"degraded"` object with an `error` string instead of raising, so a broken subsystem doesn't take down the health endpoint.

### `/maintenance`

Write operations on the index and cache. Treat these as administrative.

| Method | Path | Body / Query | Purpose |
|--------|------|--------------|---------|
| `POST` | `/maintenance/rebuild_index` | — | Rebuild FAISS from the database |
| `GET`  | `/maintenance/verify` | — | Check DB / vector sync |
| `POST` | `/maintenance/reset_index` | — | Empty the vector index |
| `POST` | `/maintenance/save` | — | Force-save the vector index to disk |
| `POST` | `/maintenance/cleanup` | `?dry_run=true` | Run the pruner |
| `POST` | `/maintenance/consolidate` | `?threshold=0.85&dry_run=true` | Run consolidation |
| `POST` | `/maintenance/rebuild_cache` | — | Rebuild the embedding cache from DB + vectors |

`cleanup` and `consolidate` default to `dry_run=true`. Pass `dry_run=false` explicitly to actually modify state.

`rebuild_index` resets the vector store first, then re-embeds every row in the database. Ghost vectors that no longer correspond to a DB row are removed in the process. This is the correct endpoint to run after `/maintenance/verify` reports a mismatch.

### `/benchmark`

Small HTTP-accessible benchmark endpoints. These are **not** the LongMemEval pipeline — for that, see [Benchmarks](benchmarks.md) and [Adapters](adapters.md).

| Method | Path | Body | Returns |
|--------|------|------|---------|
| `POST` | `/benchmark/run` | `[{"query": "...", "expected": "..."}]` | Accuracy stats and sample failures |
| `POST` | `/benchmark/speed` | `{"queries": [...]}` | `{queries, seconds, qps}` |
| `POST` | `/benchmark/store_speed` | `{"texts": [...]}` | `{stored, seconds, stores_per_second, ...}` |
| `POST` | `/benchmark/full` | Optional combination of the above | All requested sections |
| `POST` | `/benchmark/quick` | — | Speed test over 3 sample queries |

`/benchmark/run` reports accuracy as substring match: a query counts as correct if the `expected` string appears in the concatenated top-K retrieved text. That is a coarse metric and is not the same as the session-level or turn-level retrieval metrics in [Benchmarks](benchmarks.md).

### Root

| Method | Path | Returns |
|--------|------|---------|
| `GET` | `/` | `templates/index.html` if present, or a fallback page pointing at `/docs` |
| `GET` | `/health` | `{"status": "ok", "version": ..., "service": "Memory Daemon"}` |

The root `/health` is different from `/debug/health`. Root is a liveness check that always answers quickly. `/debug/health` runs the full diagnostics and may return a degraded status.

---

## Error Handling

The API raises `HTTPException` with status `500` for any unhandled error inside an endpoint. The response body is FastAPI's standard error envelope:

```json
{"detail": "..."}
```

Validation errors — wrong field name, wrong type, missing required field — return `422` with FastAPI's structured validation detail:

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "text"],
      "msg": "Field required"
    }
  ]
}
```

Empty or whitespace-only strings passed to `/chat/` or `/chat/raw` return a normal `200` with `{"status": "error", "response": "Empty prompt provided."}` rather than a `4xx`. Handle the `status` field, not just the HTTP code.

---

## CORS

CORS is off by default. `app.py` checks `settings.ENABLE_CORS`, but that attribute is not defined in `cache/config.py`, so it currently always evaluates to `False` and the middleware is never mounted. If you need CORS for a browser client, add `ENABLE_CORS: bool = True` to `Settings` and restart — the middleware block is already written.

---

## Client Libraries

There is no published client library. `/docs` provides a built-in interactive console for every endpoint, and FastAPI can export the OpenAPI spec for import into Postman, Insomnia, or HTTPie:

```bash
python -c "from app import app; import json; print(json.dumps(app.openapi(), indent=2))" > openapi.json
```

The spec reflects the current code — regenerate it whenever routes change.
