---
title: Retrieval
description: Memoria's parallel retrieval pipeline: workers, scheduler, fusion, temporal, and routing.
---

# Retrieval

Memoria's retrieval layer is a **parallel, scheduler-coordinated pipeline** with declarative completion policies. The goal is to keep retrieval responsible for **finding candidates** and ranking responsible for **deciding usefulness**.

---

## Retrieval Workers

| Worker | Purpose |
|--------|---------|
| **FAISS** | Dense semantic retrieval over embeddings |
| **BM25** | Lexical retrieval with BM25 scoring |
| **Graph** | Entity/relationship traversal over the numpy graph |
| **Phrase** | Phrase matching via the inverted index |
| **Attribute** | Structured attribute lookup |
| **Fusion** | Combined retrieval (RRF over FAISS + BM25) |
| **Temporal** | Standalone temporal/session retrieval |

Workers are independently replaceable. New workers can be registered via the plugin system.

---

## Blackboard / Scheduler

Workers submit tasks to a scheduler that executes them in parallel. The scheduler uses **completion policies** rather than synchronous joins.

The default policy is **source coverage**: the scheduler requires that a set of sources completes before the query handler proceeds. If the policy cannot be satisfied within the configured deadline, the scheduler terminates and the query handler uses whatever results are available.

Key pieces:

- `SourceCoveragePolicy` — the default policy
- `AllCompletePolicy` — require every submitted source
- `QuorumPolicy` — require N completions
- `SufficientPolicy` — caller-supplied evaluator

---

## Candidate Records

Each worker returns `(memory_id, score)` tuples. The query handler constructs `CandidateRecord` objects that attach:

- the underlying `MemoryRecord`
- the retrieval source
- the source-specific score
- graph-hit flags
- embeddings (for MMR / cross-encoder)
- diagnostics

Candidate records are the contract between retrieval and ranking.

---

## Fusion

The fusion worker performs **Reciprocal Rank Fusion (RRF)** over FAISS and BM25.

```text
RRF(d) = Σ 1 / (k + rank_i(d))
```

- `k` is configurable via `RRF_K` (default: 10)
- Overlap between FAISS and BM25 improves a candidate's final score
- Source-specific score scales do not need to be normalized

Fusion is treated as an explicit retrieval primitive. The scheduler may instead submit FAISS and BM25 independently and fuse downstream.

---

## Temporal Retrieval

The temporal worker is a **standalone retrieval path** for temporal queries.

The parser extracts:

- **Session references** (`"session 4"`, `"previous session"`, `"3 sessions ago"`)
- **Explicit temporal phrases** (`"last week"`, `"yesterday"`)
- **Relational temporal language** (`"before"`, `"after"`, `"since"`, `"until"`)
- **Recency markers** (`"most recent"`, `"latest"`, `"newest"`)
- **Event-time queries** (`"when did X happen?"`)

Two paths are handled:

1. **Explicit constraints** — the query mentions a session, date, or relative period. The temporal worker resolves the constraint and retrieves matching memories.
2. **Event-time queries** — the query has temporal *intent* (`"when did..."`) but no explicit constraint. The temporal worker performs semantic retrieval to locate the event, then applies entity and recency boosts to rank candidates.

Sparse session resolution is supported via `TEMPORAL_USE_SPARSE_RESOLUTION`.

---

## Routing

Memoria routes queries by **memory type**. The router returns:

- a worker set
- a graph depth
- a signal set
- a pool
- fallback pools

A temporal query is routed to the **temporal worker only** unless the temporal worker returns nothing, in which case the query falls back to fusion.

---

## Configuration Levers

| Goal | Lever |
|------|-------|
| More candidates | `TOP_K`, `TOP_K_PER_SHARD` |
| More lexical recall | `USE_BM25`, `USE_INVERTED_INDEX` |
| More graph recall | `GRAPH_TOP_K`, `GRAPH_DEPTH` |
| Temporal coverage | `TEMPORAL_EXACT_MATCH_BOOST`, `TEMPORAL_RECENCY_SCALE` |
| Fusion behavior | `RRF_K` |
| Sharding | `USE_SHARDING`, `NUM_SHARDS` |

See [Configuration](configuration.md) for the full list.

---

## Where Ranking Fits

Ranking is **optional**. When ranking is disabled, Memoria sorts candidates by the score already attached by retrieval. When ranking is enabled, the full pipeline runs:

```text
MemoryRanker → ScoreNormalizer → AttributeBooster → BM25 → ScoreFinalizer → MMR
```

The architectural boundary is intentional: retrieval finds candidates, ranking decides usefulness.
