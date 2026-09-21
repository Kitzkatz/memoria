## `docs/04_Dataflow.md` — Data Flow

```markdown
# Data Flow

Memory Daemon has two primary data pipelines:

1. **STORE** — Ingesting new memories
2. **RECALL** — Retrieving and generating responses

---

## STORE Pipeline

### Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ┌──────────┐                                                      │
│  │   Text   │  Raw input from user or ingestion                    │
│  └────┬─────┘                                                      │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────┐                                                      │
│  │Extractor │  MemoryExtractor extracts structured data            │
│  └────┬─────┘                                                      │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────┐                                                      │
│  │ Metadata │  Length, word count, flags (URL, email, code)       │
│  └────┬─────┘                                                      │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────┐                                                      │
│  │ Entities │  Named entities (people, places, things)             │
│  └────┬─────┘                                                      │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────┐                                                      │
│  │Relations │  Entity relationships (source → relation → target)   │
│  └────┬─────┘                                                      │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────┐                                                      │
│  │Importance│  Heuristic importance score (0.0 - 1.0)             │
│  └────┬─────┘                                                      │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────┐                                                      │
│  │Embedding │  Generate vector via SentenceTransformer            │
│  └────┬─────┘                                                      │
│       │                                                            │
│       ├──────────────────────────────────────────────────┐         │
│       │                                                  │         │
│       ▼                                                  ▼         │
│  ┌──────────┐                                      ┌──────────┐  │
│  │ Database │  Insert into SQLite                   │  FAISS   │  │
│  │          │  - Main table                         │  Index   │  │
│  │          │  - Type tables                       │          │  │
│  │          │  - Entities                          │          │  │
│  └────┬─────┘                                      └────┬─────┘  │
│       │                                                  │         │
│       ▼                                                  ▼         │
│  ┌──────────┐                                      ┌──────────┐  │
│  │  Graph   │  Build relationships                   │  Cache  │  │
│  │          │  - Entity links                       │          │  │
│  │          │  - Edge store                         │          │  │
│  └──────────┘                                      └──────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### STORE Pipeline — Detailed Steps

#### 1. Text
- Raw input text from user or ingestion (PDF, code, etc.)
- Can be single line or multi-paragraph

#### 2. Extractor (`MemoryExtractor`)
- Normalizes text (case folding, whitespace)
- Tokenizes
- Extracts structured data

#### 3. Metadata
- Length, word count
- Contains number, URL, email, code, question
- Type detection (semantic, episodic, procedural, code, science)

#### 4. Entities (`extract_entities`)
- Rule-based extraction
- Handles capitalized words and acronyms
- Deduplicates entities

#### 5. Relationships (`extract_relationships`)
- Extracts source → relation → target triples
- Uses heuristics and patterns
- If no explicit relationship, connects consecutive entities

#### 6. Importance (`ImportanceScorer`)
- Heuristic scoring:
  - Presence of importance cues ("remember this")
  - Text length
  - Metadata signals
  - Recency
  - Access frequency

#### 7. Embedding (`Embedder`)
- Uses SentenceTransformer model
- Generates vector representation (384 dims)
- Cached for future use

#### 8. Database (`MemoryDB`)
- Inserts into main `memories` table
- Inserts into type-specific table (semantic, episodic, etc.)
- Stores entities and relationships

#### 9. Graph (`RelationshipBuilder`)
- Builds entity → entity edges
- Stores in `graph` table
- Updates Numpy graph for fast queries

#### 10. FAISS (`VectorStore`)
- Adds vector to FAISS index
- Supports batch insert for performance
- Persists to disk

#### 11. Cache (`EmbeddingCache`)
- Stores embedding in memory
- Used for fast retrieval on subsequent queries

---

## RECALL Pipeline

### Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ┌──────────┐                                                      │
│  │  Query   │  User question or prompt                             │
│  └────┬─────┘                                                      │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────┐                                                      │
│  │Normalize │  QueryProcessor: normalize, tokenize, entities      │
│  └────┬─────┘                                                      │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────┐                                                      │
│  │  Embed   │  Generate query vector                               │
│  └────┬─────┘                                                      │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────┐                                                      │
│  │  FAISS   │  Semantic search (top K candidates)                 │
│  └────┬─────┘                                                      │
│       │                                                            │
│       ├──────────────────────────────────────────────────┐         │
│       │                                                  │         │
│       ▼                                                  ▼         │
│  ┌──────────┐                                      ┌──────────┐  │
│  │  Graph   │  Entity expansion                     │   BM25   │  │
│  │  Search  │  (neighbors up to depth)              │  Search  │  │
│  └────┬─────┘                                      └────┬─────┘  │
│       │                                                  │         │
│       └──────────────────┬───────────────────────────────┘         │
│                          │                                         │
│                          ▼                                         │
│  ┌──────────┐                                                      │
│  │Candidate │  Merge, deduplicate, enrich                         │
│  │Construct │  Build CandidateRecord list                         │
│  └────┬─────┘                                                      │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────┐                                                      │
│  │ Ranking  │  10+ signals + scoring                              │
│  │  (Ranker)│  - Semantic, Entity, Subject, Attribute             │
│  │          │  - Token, TF-IDF, BM25                              │
│  │          │  - Importance, Recency, Feedback                    │
│  └────┬─────┘                                                      │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────┐                                                      │
│  │   MMR    │  Diversity reranking                                │
│  │Reranker  │  (Maximal Marginal Relevance)                       │
│  └────┬─────┘                                                      │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────┐                                                      │
│  │  Final   │  Combined confidence score                          │
│  │  Score   │  (Relevance + Importance + Recency + Diversity)     │
│  └────┬─────┘                                                      │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────┐                                                      │
│  │ Context  │  Token budget filtering                             │
│  │ Builder  │  Select top N within budget                         │
│  └────┬─────┘                                                      │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────┐                                                      │
│  │   LLM    │  Generate response with context                     │
│  └──────────┘                                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### RECALL Pipeline — Detailed Steps

#### 1. Query
- Raw user input
- Can be question, command, or prompt

#### 2. Normalize (`QueryProcessor`)
- Case folding (NFKD)
- Tokenization
- Entity extraction
- Phrase extraction (quoted text)
- Attribute detection
- Memory type routing

#### 3. Embed (`Embedder`)
- Generate query vector
- Uses same model as STORE
- Returns 384-dim vector

#### 4. FAISS (`VectorStore`)
- Semantic search using cosine similarity
- Returns top K memory IDs with distances
- Configurable `TOP_K`

#### 5. Graph Search (`GraphSearch` / `NumpyGraph`)
- Entity expansion (neighbors up to depth)
- Returns memory IDs connected to entities
- Uses Numpy graph for speed

#### 6. Candidate Construction (`RetrievalEngine`)
- Merge FAISS + Graph + BM25 + Phrase + Attribute results
- Deduplicate candidates
- Build `CandidateRecord` objects

#### 7. Ranking (`MemoryRanker`)
- Apply 10+ signals:
  - Semantic, Entity, Subject, Attribute
  - Token, TF-IDF, BM25
  - Importance, Recency, Feedback
- Compute base score
- Normalize with Z-score

#### 8. MMR Reranker (`MMRReranker`)
- Re-rank for diversity
- Balances relevance and diversity
- Configurable lambda (0.5 default)

#### 9. Final Score (`ScoreFinalizer`)
- Combine signals:
  - Relevance (normalized score)
  - Importance
  - Recency
  - Diversity (1 - diversity)
  - Attribute
  - BM25
- Produces final confidence score

#### 10. Context Builder (`ContextBuilder`)
- Filter by score threshold
- Select within token budget
- Preserve ranking order

#### 11. LLM (`LLMAdapter`)
- Build prompt with context
- Generate response
- Return to user

---

## Store vs Recall — Comparison

| Aspect | STORE | RECALL |
|--------|-------|--------|
| Input | Raw text | Query |
| Output | Memory ID | Response + diagnostics |
| Key Operations | Extraction, embedding, storage | Search, ranking, generation |
| Latency | ~100ms | ~150ms |
| Threading | Single thread | Parallel workers |
| Dependencies | Extractor, DB, FAISS, Graph | QueryProcessor, FAISS, Ranker, LLM |

---

## See Also

- `03_system_architecture.md` — System architecture
- `02_project_overview.md` — Project overview
- `05_Design_Principles.md` — Design principles
- `v4.md` — V4 reasoning infrastructure
```

---
