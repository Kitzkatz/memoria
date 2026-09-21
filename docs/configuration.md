---
title: Configuration
description: Every Memoria setting, grouped by subsystem, overridable via MEMORY_-prefixed environment variables.
---

# Configuration

Memoria exposes its configuration through Pydantic settings in `cache/config.py`. Every setting can be overridden with an environment variable prefixed by `MEMORY_`.

Example:

```bash
export MEMORY_TOP_K=1000
export MEMORY_CONTEXT_TOKEN_BUDGET=20000
export MEMORY_RANKING_ENABLED=false
```

---

## Retrieval

| Setting | Purpose |
|---------|---------|
| `TOP_K` | Maximum candidates returned per query |
| `TOP_N` | Secondary candidate cap |
| `TOP_K_PER_SHARD` | Per-shard candidate limit (used with `USE_SHARDING`) |
| `GRAPH_TOP_K` | Max candidates from the graph worker |
| `GRAPH_SEARCH_LIMIT` | Max nodes visited during graph traversal |
| `GRAPH_DEPTH` | Default traversal depth for graph retrieval |
| `USE_BM25` | Enable BM25 lexical retrieval |
| `USE_PHRASE_SEARCH` | Enable phrase worker |
| `USE_INVERTED_INDEX` | Enable inverted index (required for phrase + BM25) |
| `RETRIEVAL_MIN_CANDIDATES` | Minimum candidates before scheduler can finish |
| `MIN_RETRIEVAL_SOURCES` | Minimum distinct sources required |
| `RETRIEVAL_DEADLINE` | Hard deadline for retrieval |

---

## Routing

| Setting | Purpose |
|---------|---------|
| `USE_ROUTING` | Enable the memory-type router |
| `ROUTING_MATRIX_OVERRIDE` | Allow per-query routing overrides |
| `ROUTING_FALLBACK_ENABLED` | Fall back to general routing when a type is unknown |

---

## Ranking

| Setting | Purpose |
|---------|---------|
| `RANKING_ENABLED` | Master toggle for the full ranking pipeline |
| `RANKING_SEMANTIC` | Weight for semantic similarity |
| `RANKING_TOKEN` | Weight for token overlap |
| `RANKING_TFIDF` | Weight for TF-IDF |
| `RANKING_BM25` | Weight for BM25 |
| `RANKING_ENTITY` | Weight for entity overlap |
| `RANKING_SUBJECT` | Weight for subject match |
| `RANKING_ATTRIBUTE` | Weight for attribute match |
| `RANKING_IMPORTANCE` | Weight for memory importance |
| `RANKING_RECENCY` | Weight for recency |
| `RANKING_FEEDBACK` | Weight for feedback loop signals |

Ranking weights are normalized; the sum should be approximately 1.0.

---

## Memory Lifecycle

| Setting | Purpose |
|---------|---------|
| `MEMORY_DECAY_DAYS` | Days used for recency decay |
| `MEMORY_DECAY_RATE` | Decay rate applied to importance |
| `CONSOLIDATE_THRESHOLD` | Similarity threshold for memory consolidation |
| `CONSOLIDATE_BATCH_SIZE` | Batch size for consolidation |
| `PRUNE_THRESHOLD` | Importance threshold below which memories are pruned |
| `PRUNE_MAX_AGE_DAYS` | Max age before pruning eligibility |
| `AUTO_STORE_MEMORIES` | Automatically store memories from query results |
| `AUTO_STORE_THRESHOLD` | Confidence threshold for auto-store |

---

## Architecture

| Setting | Purpose |
|---------|---------|
| `USE_BLACKBOARD` | Enable the blackboard/scheduler path |
| `USE_SHARDING` | Enable type-based sharding |
| `NUM_SHARDS` | Number of shards when sharding is enabled |
| `MMR_ENABLED` | Enable Maximal Marginal Relevance re-ranking |
| `USE_ADAPTIVE_WEIGHTS` | Enable adaptive weight tuning |
| `RANKER_DIAGNOSTICS` | Attach per-candidate ranking diagnostics |
| `ENABLE_SIGNAL_REGISTRY` | Enable the signal registry for per-type weights |

---

## Temporal

| Setting | Purpose |
|---------|---------|
| `USE_TEMPORAL_WORKER` | Enable the standalone temporal worker |
| `TEMPORAL_INDEX_PATH` | Path to the temporal index cache |
| `TEMPORAL_EXACT_MATCH_BOOST` | Score boost for exact session match |
| `TEMPORAL_ADJACENT_BOOST` | Score boost for adjacent session |
| `TEMPORAL_RECENCY_SCALE` | Decay scale for recency queries |
| `TEMPORAL_CONVERSATIONAL_BOOST` | Boost for conversational recency markers |
| `TEMPORAL_USE_SPARSE_RESOLUTION` | Use sparse session resolution |
| `TEMPORAL_FALLBACK_TO_ARITHMETIC` | Fall back to arithmetic resolution |

---

## Finalizer

| Setting | Purpose |
|---------|---------|
| `FINALIZER_RELEVANCE` | Relevance contribution |
| `FINALIZER_IMPORTANCE` | Importance contribution |
| `FINALIZER_RECENCY` | Recency contribution |
| `FINALIZER_DIVERSITY` | Diversity contribution |
| `FINALIZER_ATTRIBUTE` | Attribute contribution |
| `FINALIZER_BM25` | BM25 contribution |
| `FINALIZER_TEMPORAL` | Temporal contribution |
| `FINALIZER_USE_SIGMOID` | Use sigmoid squashing for relevance |
| `FINALIZER_SIGMOID_SCALE` | Sigmoid scale factor |

---

## Performance Tuning Quick Reference

| Goal | Action |
|------|--------|
| **Speed** | Use embedding cache / `SKIP_EMBEDDING` |
| **Recall** | Increase `TOP_K_PER_SHARD` and `TOP_K` |
| **Quality** | Tune ranking and finalizer weights |
| **Context** | Adjust `CONTEXT_TOKEN_BUDGET` |

Memoria is designed so retrieval, ranking, routing, scheduling, and storage behavior can be tuned independently.
