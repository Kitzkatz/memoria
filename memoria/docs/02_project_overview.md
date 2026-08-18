## `docs/02_project_overview.md` — Project Overview

```markdown
# Project Overview

Memory Daemon is a fully local cognitive memory architecture for language models.

It provides long-term memory that is:
- Independent of any specific LLM
- Fast (150ms queries on 4GB RAM)
- Extensible
- Observable
- Local-first

---

## Core Subsystems

### Memory
Stores and manages memory records.

- `MemoryDB` — SQLite backend
- `MemoryRecord` — The memory object
- `MemoryPruner` — Automatic cleanup
- `RelevanceManager` — Query frequency tracking

**Responsibility:** Durability and retrieval of raw memory.

---

### Retrieval
Finds relevant memories for a query.

- `RetrievalEngine` — Orchestrates retrieval
- `QueryProcessor` — Query preprocessing
- `InvertedIndex` — Keyword and phrase search
- `BM25` — Lexical ranking

**Responsibility:** Candidate generation.

---

### Ranking
Scores and orders candidates.

- `MemoryRanker` — 10+ ranking signals
- `ScoreNormalizer` — Z-score normalization
- `AttributeBooster` — Attribute-based boosting
- `MMRReranker` — Diversity reranking
- `ScoreFinalizer` — Final confidence score
- `RankingPipeline` — Orchestrates ranking

**Responsibility:** Candidate prioritization.

---

### Graph
Entity relationships and graph traversal.

- `NumpyGraph` — Fast in-memory graph
- `EntityStore` — Entity storage and resolution
- `EdgeStore` — Graph edge management
- `GraphSearch` — Graph traversal and search
- `RelationshipBuilder` — Builds relationships from memory

**Responsibility:** Relationship-aware retrieval.

---

### Cache
Fast access to computed values.

- `EmbeddingCache` — In-memory vector cache
- `Embedder` — SentenceTransformer wrapper

**Responsibility:** Performance optimization.

---

### Database
Persistence layer.

- `MemoryDB` — SQLite connection and operations
- Schema management
- Type-specific tables
- Soft delete (tombstone)

**Responsibility:** Data persistence.

---

### Embedding
Vector representation of text.

- `Embedder` — HuggingFace SentenceTransformer
- `VectorStore` — FAISS index
- `EmbeddingCache` — Memory cache for vectors

**Responsibility:** Semantic representation.

---

### Context
Token-aware context assembly.

- `ContextBuilder` — Token budgeting
- `TokenEstimator` — Token counting
- `Context` — Memory selection within budget

**Responsibility:** Context preparation for LLM.

---

### LLM
Language model interface.

- `LLMAdapter` — LLM-agnostic communication
- `MemoryController` — Unified API
- `Chat` — Conversation management

**Responsibility:** Generation and reasoning.

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ┌──────────┐                                                      │
│  │   User   │                                                      │
│  └────┬─────┘                                                      │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────┐                                                      │
│  │  Query   │                                                      │
│  │Processor │  Normalize, tokenize, extract entities               │
│  └────┬─────┘                                                      │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────┐                                                      │
│  │Embedding │  Convert query to vector                             │
│  └────┬─────┘                                                      │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────┐                                                      │
│  │Retrieval │  FAISS, BM25, Graph, Phrase, Attribute              │
│  └────┬─────┘                                                      │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────┐                                                      │
│  │ Ranking  │  10+ signals + MMR                                  │
│  └────┬─────┘                                                      │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────┐                                                      │
│  │ Context  │  Token budgeting                                     │
│  │ Builder  │                                                      │
│  └────┬─────┘                                                      │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────┐                                                      │
│  │   LLM    │  Generate response                                   │
│  └────┬─────┘                                                      │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────┐                                                      │
│  │Response  │  Return to user                                      │
│  └──────────┘                                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow in Detail

### 1. User Input
User submits a query via CLI, TUI, GUI, or API.

### 2. Query Processor
- Normalizes text (case folding, Unicode)
- Tokenizes
- Extracts entities (rule-based or LLM)
- Extracts phrases (quoted text)
- Detects attributes
- Routes memory type (semantic, episodic, procedural, code, science)

### 3. Embedding
- Generates vector embedding for the query
- Uses SentenceTransformer model
- Cached for repeat queries

### 4. Retrieval
- Runs in parallel:
  - FAISS (semantic search)
  - BM25 (lexical search)
  - Graph (entity relationship search)
  - Phrase (exact phrase search)
  - Attribute (subject-attribute search)
- Results are merged and deduplicated

### 5. Ranking
- 10 signals applied:
  - Semantic, Entity, Subject, Attribute
  - Token, TF-IDF, BM25
  - Importance, Recency, Feedback
- Z-score normalization
- Attribute boosting
- MMR reranking for diversity
- Final score computed

### 6. Context Builder
- Filters by score threshold
- Selects within token budget
- Preserves order from ranking
- Returns context-ready memories

### 7. LLM
- Builds prompt with context
- Sends to configured LLM
- Returns response

### 8. Response
- Returns to user
- Records feedback (clicks, dwell, skips)
- Updates query history
- Updates relevance scores

---

## Key Metrics

| Metric | Target |
|--------|--------|
| Query latency | ~150ms |
| Memory footprint | ~4GB RAM |
| CPU | No GPU required |
| Storage | SQLite + FAISS |
| Languages | Python 3.12+ |

---

## See Also

- `03_system_architecture.md` — Detailed architecture
- `04_Dataflow.md` — Data flow diagrams
- `05_Design_Principles.md` — Design principles
- `project_manifesto.md` — The vision
```

---

## Next?

`04_Dataflow.md` next?
