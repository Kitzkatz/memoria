## `docs/03_system_architecture.md` — System Architecture

```markdown
# System Architecture

Memory Daemon V4 is a layered system designed for local-first, LLM-agnostic memory.

Each layer is independent and composable.

---

## Layer Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                     ┌─────────────────┐                        │
│                     │   Generation    │                        │
│                     │  (LLM output)   │                        │
│                     └────────┬────────┘                        │
│                              │                                 │
│                     ┌────────▼────────┐                        │
│                     │    Reasoning    │                        │
│                     │  (goals, plans) │                        │
│                     └────────┬────────┘                        │
│                              │                                 │
│                     ┌────────▼────────┐                        │
│                     │    Ranking      │                        │
│                     │ (signals, MMR)  │                        │
│                     └────────┬────────┘                        │
│                              │                                 │
│                     ┌────────▼────────┐                        │
│                     │   Retrieval     │                        │
│                     │ (FAISS, BM25)   │                        │
│                     └────────┬────────┘                        │
│                              │                                 │
│                     ┌────────▼────────┐                        │
│                     │     Memory      │                        │
│                     │  (storage, DB)  │                        │
│                     └─────────────────┘                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Memory Layer

**Responsibility:** Store, retrieve, and manage memory objects.

**Components:**

| Component | Purpose |
|-----------|---------|
| `MemoryDB` | SQLite backend with WAL and full-text search |
| `VectorStore` | FAISS index for semantic search |
| `EmbeddingCache` | In-memory cache for vectors |
| `Graph` | Numpy-based graph for entity relationships |
| `MemoryPruner` | Background cleanup of stale memories |
| `RelevanceManager` | Tracks query frequency and feedback |

**Key Interfaces:**

```python
db.insert(record)          # Store a memory
db.fetch_many(ids)         # Batch retrieve
vector_store.search(vec)   # Semantic search
graph.neighbors(entity)    # Relationship traversal
```

**Design Decisions:**

- SQLite for portability and local-first
- FAISS for fast vector search on CPU
- Numpy graph for fast entity relationships
- Soft delete (tombstone) instead of hard delete

---

## Layer 2: Retrieval Layer

**Responsibility:** Find relevant memories for a query.

**Components:**

| Component | Purpose |
|-----------|---------|
| `QueryProcessor` | Normalize, tokenize, extract entities |
| `RetrievalEngine` | Orchestrates multi-strategy retrieval |
| `InvertedIndex` | Keyword and phrase search |
| `BM25` | Lexical ranking |
| `ShardManager` | Type-based sharding |
| `Router` | Routes query to appropriate workers |

**Retrieval Strategies:**

| Strategy | When Used | Source |
|----------|-----------|--------|
| FAISS | Semantic similarity | Vector embeddings |
| BM25 | Keyword overlap | Inverted index |
| Graph | Entity relationships | Numpy graph |
| Phrase | Exact phrase matches | Positional index |
| Attribute | Subject-attribute facts | Metadata extraction |

**Execution Model:**

- All strategies run in parallel
- Results are merged and deduplicated
- Type routing selects which strategies to use per query

---

## Layer 3: Ranking Layer

**Responsibility:** Score and rank retrieved candidates.

**Components:**

| Component | Purpose |
|-----------|---------|
| `MemoryRanker` | Core scoring with 10+ signals |
| `ScoreNormalizer` | Z-score normalization |
| `AttributeBooster` | Boost based on attribute matches |
| `MMRReranker` | Diversity re-ranking |
| `ScoreFinalizer` | Final confidence score |

**Signals:**

| Signal | Weight | Description |
|--------|--------|-------------|
| Semantic | 0.21 | Cosine similarity from FAISS |
| Entity | 0.18 | Entity overlap score |
| Subject | 0.16 | Subject match |
| Attribute | 0.12 | Attribute match |
| Token | 0.07 | Token overlap |
| TF-IDF | 0.06 | Term frequency score |
| BM25 | 0.09 | BM25 relevance |
| Importance | 0.05 | Memory importance |
| Recency | 0.03 | Age decay |
| Feedback | 0.03 | User feedback |

**Pipeline:**

```
Raw Candidates
    ↓
Ranker (signal scores)
    ↓
Normalizer (Z-score)
    ↓
Attribute Booster (boost)
    ↓
BM25 Scoring (lexical)
    ↓
Finalizer (confidence)
    ↓
Context Builder (token budget)
    ↓
MMR (diversity)
    ↓
Ranked Results
```

---

## Layer 4: Reasoning Layer

**Responsibility:** Active reasoning over memory. V4 feature.

**Components:**

| Component | Status | Purpose |
|-----------|--------|---------|
| Goals | ✅ Complete | Active objectives to pursue |
| Blackboard | ✅ Complete | Shared state for reasoning |
| Task Scheduler | ✅ Complete | Parallel execution |
| Reasoning Nodes | 🔄 In Progress | Atomic reasoning units |
| Computation Graph | 🔄 In Progress | Flow of reasoning |
| Planner | 🔜 Planned | Maps goals to graphs |
| Execution Queue | 🔜 Planned | Manages pending work |

**Reasoning Flow:**

```
User Prompt
    ↓
Planner (maps goal → graph)
    ↓
Execution Queue (schedules work)
    ↓
Task Scheduler (parallel execution)
    ↓
Reasoning Nodes (atomic operations)
    ↓
Blackboard (shared results)
    ↓
Response (synthesized output)
```

---

## Layer 5: Generation Layer

**Responsibility:** Generate human-readable responses.

**Components:**

| Component | Purpose |
|-----------|---------|
| `MemoryController` | Unified API for all operations |
| `LLMAdapter` | LLM-agnostic chat interface |
| `ContextBuilder` | Token-aware context assembly |
| `MemoryInterface` | Shared interface for all clients |

**Features:**

- LLM-agnostic — swap models via config
- Context-aware — uses retrieved memories
- Token budgeting — respects context window
- Chat history — maintains conversation state

---

## Data Flow Summary

```
1. User asks a question
2. QueryProcessor normalizes and extracts entities
3. Router selects retrieval strategy
4. Workers run in parallel (FAISS, BM25, Graph)
5. Results merged and deduplicated
6. Ranker scores all candidates
7. MMR reranks for diversity
8. ContextBuilder selects within token budget
9. LLM generates response
10. Feedback loop records behavior
```

---

## Cross-Cutting Concerns

| Concern | Implementation |
|---------|----------------|
| Performance | Parallel workers, batching, caching |
| Memory | LRU cache, pruning, compaction |
| Observability | Flight recorder, timing logs |
| Thread Safety | RLock, atomic operations |
| Persistence | Atomic writes, WAL |
| Testing | Benchmark suite, regression tests |

---

## See Also

- `02_project_overview.md` — High-level project intro
- `04_Dataflow.md` — Detailed data flow diagrams
- `05_Design_Principles.md` — Design decisions explained
- `v4.md` — V4 reasoning infrastructure
```

---

## Next?

Want to tackle `project_manifesto.md` next? That's the vision doc.
