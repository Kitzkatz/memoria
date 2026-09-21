# Retrieval

Memoria V4 uses a policy-driven, parallel retrieval architecture built around typed routing, multiple retrieval workers, a shared blackboard/scheduler, and a downstream candidate-ranking pipeline.

The retrieval layer is intentionally separated from ranking. Retrieval is responsible for finding and constructing a candidate set. Ranking can then apply additional signals, reranking, diversity, and finalization.

This separation also allows Memoria to run retrieval-focused experiments with the full ranking pipeline disabled.

---

## Retrieval Pipeline

The current V4 blackboard path is approximately:

```text
Query
  ↓
QueryProcessor
  ↓
Query Expansion (optional)
  ↓
Query Embedding
  ↓
Router
  ↓
Relevance Pool
  ↓
Shard Selection
  ↓
Parallel Retrieval Workers
  ↓
Scheduler Completion Policy
  ↓
Completed Worker Results
  ↓
Candidate Construction
  ↓
Candidate Deduplication
  ↓
Candidate Cap
  ↓
Optional Cross-Encoder
  ↓
Type Filtering
  ↓
Ranking Pipeline
  ↓
Response
```

The retrieval boundary extends from query routing through candidate construction and type filtering. Ranking occurs afterward.

Memoria also retains a V3 fallback path. When the blackboard system is disabled or unavailable, the system can fall back to the earlier FAISS-based retrieval path.

---

## Query Processing

A query first passes through `QueryProcessor`.

The resulting query representation contains the normalized query information used by downstream retrieval components, including:

* normalized text
* tokens
* metadata
* memory-type hints
* entities
* phrases
* subject/attribute information when available

V4 can optionally perform query expansion before embedding and retrieval.

When enabled, the query expander can add tokens to the query representation while retaining the original and expanded token sets in query metadata.

This allows lexical retrieval workers such as BM25 to operate on the expanded query while preserving diagnostic information about the transformation.

---

## Query Embedding

After query processing, Memoria generates a query embedding from the normalized query text.

The embedding is produced once at the query boundary and is then supplied to semantic retrieval components such as FAISS and the fusion worker.

The embedding path is independent from the batch ingestion optimization used by `store_many()`. Batch ingestion can skip document embedding when operating against an already-loaded vector index, while query-time embedding remains part of the normal retrieval path.

---

## Routing

V4 uses a router to determine the retrieval configuration for the query.

Routing begins with the query's `memory_type_hint`.

The resulting route can provide:

* workers to use
* graph depth
* retrieval signals
* retrieval pool
* fallback pools

The configured worker list can also be supplied directly through `WORKERS_TO_USE`.

Conceptually:

```text
Query Type
    ↓
Router
    ├── Workers
    ├── Graph Depth
    ├── Signals
    ├── Pool
    └── Fallback Pools
```

Routing is therefore not simply a choice between "semantic" and "keyword" retrieval. It can determine which combination of retrieval mechanisms participates in a query.

The router is also exposed to the plugin system through pre-routing and post-routing hooks.

---

## Relevance Pool

Before submitting retrieval tasks, V4 can construct a relevance pool.

For non-general memory types, the relevance manager may provide highly relevant memory IDs for the requested type. Those records are converted into `CandidateRecord` objects and added to the candidate set.

If the relevance manager does not provide IDs, V4 can fall back to fetching records from the selected memory pool or type.

The relevance pool can independently satisfy a query when it already contains at least `TOP_K` candidates.

When that happens, retrieval workers are skipped for that query.

This creates a fast path:

```text
Typed Query
    ↓
Relevance Manager
    ↓
Enough candidates?
    ├── Yes → Candidate Set
    └── No  → Parallel Retrieval
```

This behavior is important when evaluating retrieval performance because not every query necessarily exercises every worker.

---

## Sharding

V4 supports optional type-based sharding.

When sharding is enabled, the shard manager selects the shards relevant to the query and memory type.

Otherwise, retrieval uses a single default shard.

The number of selected shards affects worker execution. Retrieval limits are distributed across the selected shards, including the graph candidate limit.

Conceptually:

```text
Query
  ↓
Shard Manager
  ├── Shard 0
  ├── Shard 1
  ├── ...
  └── Shard N
```

The current implementation therefore supports both:

* single-shard retrieval
* type-aware multi-shard retrieval

without changing the higher-level worker model.

---

## Retrieval Workers

V4 can execute several retrieval sources through the scheduler.

### FAISS

FAISS provides semantic/vector retrieval.

The query embedding is submitted to the FAISS worker along with the configured per-shard candidate limit.

FAISS is treated specially by the retrieval completion policy: when FAISS is submitted, it is a required source for policy satisfaction.

FAISS distances are converted into a similarity-like base score during candidate construction:

```text
base_score = 1 / (1 + distance)
```

Lower vector distance therefore corresponds to a higher retrieval score.

---

### BM25

BM25 provides lexical retrieval over tokenized memories.

The query's token representation is submitted to BM25 together with the per-shard candidate limit.

BM25 can operate alongside FAISS rather than replacing it.

The batch storage path can rebuild BM25 from the current token corpus when BM25 is enabled.

---

### Graph

The graph worker retrieves candidates through entity/relationship information.

Graph retrieval requires query entities. When entities are present and graph retrieval is selected, the worker receives:

* query entities
* candidate limit
* shard information
* configured graph depth

Graph results are auxiliary retrieval signals during candidate construction rather than a semantic similarity score.

---

### Phrase

Phrase retrieval is available when the inverted index is enabled and the query contains phrases.

The phrase worker receives the extracted query phrases and shard information.

This provides a lexical retrieval path that is distinct from ordinary BM25 matching.

---

### Attribute

Attribute retrieval operates when the query exposes both a subject and attribute.

The worker uses those structured query components to retrieve candidates from the database.

Like graph and phrase retrieval, attribute retrieval contributes candidates and/or auxiliary signals rather than functioning as the primary semantic score.

---

### Fusion

V4 can optionally register a fusion worker when fusion is enabled and BM25 is available.

The fusion worker receives both:

* the query embedding
* the query tokens

and combines semantic and lexical retrieval.

The configured `FUSION_SEMANTIC_WEIGHT` controls the semantic contribution.

Fusion therefore provides a retrieval source in its own right rather than merely being an operation performed after all workers finish.

---

## Parallel Retrieval

Selected workers are submitted independently to the scheduler.

For example:

```text
                ┌── FAISS
                ├── BM25
Query → Router ├── Graph
                ├── Phrase
                ├── Attribute
                └── Fusion
```

Not every query submits every worker.

Worker submission depends on routing and on the information available in the query. For example:

* Graph requires query entities.
* Phrase requires extracted phrases and an inverted index.
* Attribute requires subject + attribute information.
* Fusion requires the fusion configuration and BM25 availability.

This means the scheduler operates over the actual set of submitted retrieval tasks rather than a fixed list of workers.

---

## Completion Policy

V4 uses the scheduler's completion-policy system to determine when retrieval is considered complete.

The current retrieval policy:

1. requires all submitted sources to complete;
2. treats FAISS as mandatory when FAISS was submitted;
3. uses the configured retrieval deadline as a hard safety ceiling;
4. does not use the deadline as the normal completion mechanism.

Workers still running when the completion policy terminates are not included in the query's result set.

This produces explicit source-level retrieval state:

```text
submitted
completed
pending
failed
```

That state is retained in query diagnostics.

The distinction matters because a worker being pending does not mean its result was silently incorporated later. Only results completed by policy termination are consumed for that query.

---

## Candidate Construction

Completed worker results are converted into `CandidateRecord` objects.

The retrieval layer records the source associated with each candidate, together with source-specific score information.

Examples include:

* FAISS distance
* BM25 score
* Fusion score
* graph-hit status
* attribute distance
* phrase result information

FAISS and BM25 contribute directly usable retrieval scores.

Graph, phrase, and attribute results can instead act as auxiliary retrieval signals before ranking.

Candidate embeddings are recovered from the embedding cache or vector store when constructing the candidate.

---

## Candidate Deduplication

Candidates from the relevance pool and worker results are merged into one collection.

Duplicate memory IDs are removed.

The result is a unique candidate set suitable for downstream ranking.

```text
Relevance Pool
      +
Worker Results
      ↓
Candidate Merge
      ↓
Deduplicate by Memory ID
```

This keeps retrieval sources independent while giving the ranking stage a unified candidate representation.

---

## Candidate Limit

V4 caps the number of candidates before the database fetch and expensive downstream ranking stages.

The current retrieval handler uses:

```text
RANKING_CANDIDATE_LIMIT = 200
```

When the worker/relevance candidate set exceeds this limit, it is reduced before database-backed candidate construction continues.

The cap is deliberately placed before expensive ranking work.

---

## Type Filtering

After retrieval and candidate construction, V4 can apply a final memory-type filter.

When the query has a specific memory-type hint, candidates matching that type are preferred.

If matching candidates exist, the candidate set is restricted to that type.

If no candidates match the requested type, the system retains the broader candidate set rather than returning nothing.

This gives typed retrieval a fallback behavior:

```text
Typed Query
    ↓
Matching Candidates?
    ├── Yes → Keep matching type
    └── No  → Keep broader candidates
```

---

## Cross-Encoder Reranking

An optional cross-encoder can run between retrieval and the main ranking pipeline.

When enabled, Memoria:

1. selects the top configured number of candidates;
2. creates `(query, memory)` pairs;
3. scores them using the configured local cross-encoder;
4. replaces their base/final scores with the cross-encoder scores;
5. sorts the reranked candidates ahead of the remaining candidates.

The cross-encoder is therefore an optional intermediate reranking stage, not the primary retrieval mechanism.

---

## Ranking Boundary

Retrieval and ranking are explicitly separable.

With ranking enabled:

```text
Retrieval
   ↓
Candidate Set
   ↓
RankingPipeline
   ↓
Response
```

With ranking disabled:

```text
Retrieval
   ↓
Candidate Set
   ↓
Retrieval Scores
   ↓
Response
```

In retrieval-only mode, Memoria sorts candidates by the score already supplied by the retrieval source, falling back to an inverse-distance score when necessary.

This mode is useful for measuring retrieval quality independently from downstream ranking behavior.

---

## V3 Fallback

Memoria retains a V3 retrieval path for compatibility and fallback.

The V3 path performs the simpler sequence:

```text
Query
  ↓
Query Embedding
  ↓
FAISS Search
  ↓
Database Retrieval
  ↓
Candidate Set
  ↓
Ranking
```

It does not use the V4 scheduler/source-completion model.

V3 diagnostics therefore report FAISS as the submitted/completed retrieval source and do not expose V4 scheduler state.

This distinction should be preserved when comparing historical V3 results against V4 results.

---

## Retrieval Diagnostics

V4 exposes retrieval diagnostics including:

* candidate count
* returned count
* query-processing time
* embedding time
* retrieval time
* database time
* ranking time
* response time
* total query time
* retrieval policy
* retrieval finish reason
* completed task count
* pending task count
* failed task count
* retrieval wait time
* submitted sources
* completed sources
* pending sources
* failed sources
* whether ranking was skipped

These diagnostics make it possible to distinguish retrieval latency from downstream ranking and response construction.

They also make partial or deadline-limited retrieval observable instead of treating every query as an undifferentiated latency number.

---

## Retrieval Design Goals

The V4 retrieval architecture is designed around several separations:

### Retrieval vs. ranking

Retrieval finds candidates. Ranking determines their downstream ordering.

### Worker selection vs. worker execution

Routing determines which retrieval mechanisms should participate. The scheduler executes those selected tasks according to a completion policy.

### Source completion vs. deadline

Completion policy determines normal termination. The deadline is a safety ceiling.

### Semantic vs. lexical vs. structural retrieval

FAISS, BM25, phrase, graph, attribute, and fusion provide different retrieval signals rather than forcing every query through one retrieval mechanism.

### Retrieval experiments vs. full answer evaluation

Retrieval-only mode allows candidate recall and retrieval behavior to be measured without conflating those results with ranking or LLM answer generation.

---

## Current Scope

This document describes the current V4 retrieval path represented by the `MemorySystem` query handler.

It intentionally does not prescribe the internal implementation of:

* the Router's route-generation logic
* individual worker algorithms
* Scheduler internals
* RankingPipeline signal calculations
* MMR/finalizer internals
* benchmark-specific evaluation logic

Those components are separate implementation layers and should be documented from their respective source files.
