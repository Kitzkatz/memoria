---
title: Configuration
description: Every Memoria setting, grouped by subsystem.
---

# Configuration

Memoria's configuration is defined in `cache/config.py` as a Pydantic `Settings` class. Every setting can be overridden with an environment variable prefixed by `MEMORY_`.

```bash
export MEMORY_TOP_K=1000
export MEMORY_CONTEXT_TOKEN_BUDGET=20000
export MEMORY_RANKING_ENABLED=false
```

Overrides are applied at import time via `load_from_env()`. Numeric and boolean values are coerced automatically based on the type of the default.

---

## Paths

| Setting | Default | Purpose |
|---------|---------|---------|
| `DB_PATH` | `memory.db` | SQLite memory database |
| `VECTOR_INDEX_PATH` | `memory.index` | FAISS index file |
| `CACHE_PATH` | `cache/embedding_cache.pkl` | Embedding cache file |

---

## Models

| Setting | Default | Purpose |
|---------|---------|---------|
| `EMBEDDING_MODEL` | `memory/models/all-MiniLM-L6-v2` | Local embedding model path |
| `VECTOR_DIM` | `384` | Embedding dimensionality |
| `CHAT_TEMPLATE_DIR` | `chat_templates` | Prompt template directory |
| `CHAT_TEMPLATE_FILE` | `llama3.txt` | Active chat template file |
| `CHAT_TEMPLATE` | `llama3` | Template selector |
| `CHAT_MODEL` | `mistral` | Label for the LLM backend |
| `LLM_URL` | `http://localhost:8080` | LLM HTTP endpoint base |
| `LLM_ENDPOINT` | `/v1/completions` | Completions path |
| `LLM_MAX_TOKENS` | `256` | Max generated tokens |
| `LLM_TEMPERATURE` | `0.7` | Sampling temperature |
| `LLM_TIMEOUT` | `600` | LLM request timeout (seconds) |
| `LLM_STOP_TOKENS` | `["<|eot_id|>"]` | Stop sequences |

---

## Retrieval

| Setting | Default | Purpose |
|---------|---------|---------|
| `TOP_K` | `5000` | Maximum candidates returned per query |
| `TOP_N` | `550` | Secondary candidate cap |
| `GRAPH_TOP_K` | `50` | Max candidates from the graph worker |
| `GRAPH_SEARCH_LIMIT` | `200` | Max nodes visited during graph traversal |
| `GRAPH_DEPTH` | `3` | Default traversal depth for graph retrieval |
| `GRAPH_SEARCH_DEPTH` | `1` | Search depth for graph expansion |
| `USE_QUERY_EXPANSION` | `False` | Enable query expansion |
| `SYNONYM_PATH` | `retrieval/synonyms.json` | Synonym table used by expansion |

---

## Indexing

| Setting | Default | Purpose |
|---------|---------|---------|
| `USE_INVERTED_INDEX` | `True` | Enable inverted index (required for phrase + BM25) |
| `USE_PHRASE_SEARCH` | `True` | Enable phrase worker |
| `USE_BM25` | `True` | Enable BM25 lexical retrieval |
| `USE_FUSION` | `True` | Enable fusion worker |
| `FUSION_SEMANTIC_WEIGHT` | `0.5` | Semantic weight inside fusion |
| `RRF_K` | `10` | Reciprocal Rank Fusion smoothing constant |
| `MMR_ENABLED` | `False` | Maximal Marginal Relevance re-ranking |
| `USE_BLACKBOARD` | `True` | Enable the blackboard/scheduler path |
| `USE_CASE_FOLDING` | `True` | Case-fold tokens before indexing |

---

## Retrieval Workers

| Setting | Default | Purpose |
|---------|---------|---------|
| `WORKERS_TO_USE` | `["fusion"]` | Worker list submitted by the scheduler |

Valid worker names include `faiss`, `bm25`, `graph`, `phrase`, `attribute`, and `fusion`.

---

## Cross-Encoder

| Setting | Default | Purpose |
|---------|---------|---------|
| `USE_CROSS_ENCODER` | `False` | Enable cross-encoder re-scoring |
| `CROSS_ENCODER_MODEL_PATH` | `memory/models/cross-encoder` | Cross-encoder model path |
| `CROSS_ENCODER_TOP_K` | `10` | Candidates passed through cross-encoder |

---

## Routing

| Setting | Default | Purpose |
|---------|---------|---------|
| `USE_ROUTING` | `True` | Enable the memory-type router |
| `ROUTING_MATRIX_OVERRIDE` | `True` | Allow per-query routing overrides |
| `ROUTING_FALLBACK_ENABLED` | `True` | Fall back to general routing when a type is unknown |
| `RETRIEVAL_MIN_CANDIDATES` | `550` | Minimum candidates before scheduler can finish |
| `MIN_RETRIEVAL_SOURCES` | `2` | Minimum distinct sources required |
| `RETRIEVAL_DEADLINE` | `0.050` | Hard deadline for retrieval (seconds) |

---

## Ranking

| Setting | Default | Purpose |
|---------|---------|---------|
| `RANKING_ENABLED` | `False` | Master toggle for the full ranking pipeline |
| `CONTEXT_MAX_MEMORIES` | `50` | Max memories passed into context |
| `CONTEXT_MIN_SCORE` | `0.15` | Minimum score for context inclusion |
| `CONTEXT_TOKEN_BUDGET` | `10000` | Token budget for constructed context |
| `SCORE_NORMALIZER_METHOD` | `zscore` | Normalization method (`zscore` or `minmax`) |

### Signal weights

| Setting | Default |
|---------|---------|
| `RANKING_SEMANTIC` | `0.40` |
| `RANKING_TOKEN` | `0.15` |
| `RANKING_TFIDF` | `0.15` |
| `RANKING_BM25` | `0.15` |
| `RANKING_SUBJECT` | `0.10` |
| `RANKING_IMPORTANCE` | `0.01` |
| `RANKING_RECENCY` | `0.01` |
| `RANKING_FEEDBACK` | `0.01` |
| `RANKING_ENTITY` | `0.01` |
| `RANKING_ATTRIBUTE` | `0.01` |

Ranking weights are validated at startup and must sum to approximately 1.0.

| Setting | Default | Purpose |
|---------|---------|---------|
| `ENABLE_SIGNAL_REGISTRY` | `True` | Enable per-type signal registry |
| `SIGNAL_REGISTRY_PATH` | `ranking/signal_registry.json` | Registry file path |

---

## Finalizer

| Setting | Default | Purpose |
|---------|---------|---------|
| `FINALIZER_RELEVANCE` | `1.0` | Relevance contribution |
| `FINALIZER_IMPORTANCE` | `0.0` | Importance contribution |
| `FINALIZER_RECENCY` | `0.0` | Recency contribution |
| `FINALIZER_DIVERSITY` | `0.0` | Diversity contribution |
| `FINALIZER_ATTRIBUTE` | `0.0` | Attribute contribution |
| `FINALIZER_BM25` | `0.0` | BM25 contribution |
| `FINALIZER_USE_SIGMOID` | `False` | Sigmoid squashing for relevance |
| `FINALIZER_SIGMOID_SCALE` | `0.5` | Sigmoid scale factor |

---

## Memory Lifecycle

| Setting | Default | Purpose |
|---------|---------|---------|
| `MEMORY_DECAY_DAYS` | `30` | Days used for recency decay |
| `MEMORY_DECAY_RATE` | `0.001` | Decay rate applied to importance |
| `IMPORTANCE_DELTA` | `0.01` | Importance adjustment per feedback event |
| `PRUNE_THRESHOLD` | `0.1` | Importance threshold below which memories are pruned |
| `PRUNE_MAX_AGE_DAYS` | `365` | Max age before pruning eligibility |
| `PRUNE_BATCH_SIZE` | `100` | Batch size for pruning |
| `PRUNE_INTERVAL_SECONDS` | `3600` | Interval between pruning passes |
| `PRUNE_AUTO_START` | `False` | Start pruner automatically |

---

## Consolidation

| Setting | Default | Purpose |
|---------|---------|---------|
| `CONSOLIDATE_THRESHOLD` | `0.85` | Similarity threshold for consolidation |
| `CONSOLIDATE_BATCH_SIZE` | `500` | Batch size for consolidation |
| `CONSOLIDATE_AUTO` | `False` | Run consolidation automatically |
| `CONSOLIDATE_INTERVAL` | `3600` | Interval between consolidation passes (seconds) |

---

## Sharding

| Setting | Default | Purpose |
|---------|---------|---------|
| `USE_SHARDING` | `False` | Enable type-based sharding |
| `NUM_SHARDS` | `5` | Number of shards when sharding is enabled |
| `TOP_K_PER_SHARD` | `200` | Per-shard candidate limit |

---

## Boosting

| Setting | Default | Purpose |
|---------|---------|---------|
| `ENTITY_BOOST` | `0.50` | Score boost for entity matches |

---

## Feedback Loop

| Setting | Default | Purpose |
|---------|---------|---------|
| `FEEDBACK_WEIGHT` | `0.08` | Weight applied to feedback signals |
| `FEEDBACK_PERSIST_PATH` | `feedback_data.json` | Feedback persistence file |
| `QUERY_HISTORY_PERSIST_PATH` | `query_history.json` | Query history persistence file |
| `QUERY_HISTORY_MAX` | `1000` | Max query history entries retained |

---

## Adaptive Weighter

| Setting | Default | Purpose |
|---------|---------|---------|
| `USE_ADAPTIVE_WEIGHTS` | `True` | Enable adaptive weight tuning |
| `ADAPTIVE_WEIGHT_STEP` | `0.02` | Step size per tuning iteration |
| `ADAPTIVE_WEIGHT_MAX` | `0.40` | Max adaptive weight |
| `ADAPTIVE_WEIGHT_MIN` | `0.01` | Min adaptive weight |

---

## Auto-Store

| Setting | Default | Purpose |
|---------|---------|---------|
| `AUTO_STORE_MEMORIES` | `False` | Auto-store memories from query results |
| `AUTO_STORE_THRESHOLD` | `0.7` | Confidence threshold for auto-store |
| `AUTO_STORE_MAX_PER_SESSION` | `10` | Max auto-stored memories per session |
| `AUTO_STORE_TYPES` | `["general", "chat"]` | Memory types eligible for auto-store |

---

## Embedding Cache

| Setting | Default | Purpose |
|---------|---------|---------|
| `SKIP_EMBEDDING` | `False` | Skip embedding computation |
| `EMBEDDING_CACHE_MAX_SIZE` | `100000` | Max cached embeddings |

---

## Debug

| Setting | Default | Purpose |
|---------|---------|---------|
| `DEBUG` | `False` | Verbose debug output |
| `RANKER_DIAGNOSTICS` | `False` | Attach per-candidate ranking diagnostics |

---

## CLI

| Setting | Default | Purpose |
|---------|---------|---------|
| `CLI_DEFAULT_LIMIT` | `3` | Default recall limit |
| `CLI_OUTPUT_FORMAT` | `table` | Output format |
| `CLI_HISTORY_FILE` | `.memory_history` | CLI history file |
| `CLI_SHOW_SCORES` | `True` | Show scores in recall output |
| `CLI_TABLE_WIDTH` | `80` | Table width |

---

## Plugin System

| Setting | Default | Purpose |
|---------|---------|---------|
| `PLUGIN_ENABLED` | `True` | Enable plugin system |
| `PLUGIN_DIR` | `plugins` | Plugin directory |
| `PLUGIN_AUTO_LOAD` | `True` | Auto-load plugins on startup |

---

## Safety

The `Safety` helper reads the plain `ENV` environment variable — no `MEMORY_` prefix — to gate destructive operations.

| Value | Effect |
|-------|--------|
| `test` | Destructive operations permitted |
| `production` | Destructive operations refused |
| unset | Destructive operations refused |

```bash
export ENV=test
```

---

## Performance Tuning Quick Reference

| Goal | Action |
|------|--------|
| **Speed** | Enable embedding cache / set `SKIP_EMBEDDING=true` |
| **Recall** | Increase `TOP_K`, `TOP_N`, `TOP_K_PER_SHARD` |
| **Quality** | Tune ranking and finalizer weights |
| **Context** | Adjust `CONTEXT_TOKEN_BUDGET`, `CONTEXT_MAX_MEMORIES` |

Memoria is designed so retrieval, ranking, routing, scheduling, and storage behavior can be tuned independently.
