# Design Principles

Ten immutable rules that govern the Memory Daemon architecture.

---

## 1. Retrieval retrieves. Nothing else.

The retrieval layer is responsible for one thing: finding candidate memories.

- It does not rank.
- It does not filter.
- It does not generate.
- It returns candidates, nothing more.

**Why:** Separation of concerns. Retrieval is a search problem. Ranking is a decision problem. Mixing them makes both worse.

**Violation:** Doing anything other than candidate retrieval in `RetrievalEngine`.

---

## 2. Ranking ranks. Nothing else.

The ranking layer is responsible for one thing: scoring and ordering candidates.

- It does not retrieve.
- It does not filter beyond scoring.
- It does not generate.
- It returns ranked candidates, nothing more.

**Why:** Ranking is about decision-making. It should be free to change weights, signals, and strategies without affecting how memories are found.

**Violation:** Doing retrieval in `MemoryRanker` or filtering in `ScoreFinalizer`.

---

## 3. Memory never generates.

Memory stores facts, context, and history. It never invents, infers, or imagines.

- It returns what was stored.
- It does not summarize.
- It does not paraphrase.
- It does not complete.

**Why:** Memory is ground truth. Generation is a separate cognitive function. The moment memory generates, it becomes unreliable.

**Violation:** Any LLM call inside the memory layer.

---

## 4. Everything communicates through records.

All data passed between layers uses defined record types.

- `MemoryRecord` for stored memories
- `CandidateRecord` for retrieved candidates
- `QueryRecord` for processed queries
- `GraphRecord` for graph edges
- `EntityRecord` for entities

**Why:** Strong typing prevents leakage and ensures each layer gets exactly what it expects. Records are self-documenting.

**Violation:** Passing raw dicts or tuples between layers.

---

## 5. Subsystems remain replaceable.

Every major component can be swapped out without breaking the system.

- Vector store can be replaced (FAISS → other)
- LLM can be swapped (Mistral → Llama)
- Database can be changed (SQLite → Postgres)
- Ranking signals can be reweighted

**Why:** The system should evolve. Locking into one implementation is a death sentence.

**Violation:** Hardcoding dependencies or assuming specific implementations.

---

## 6. Diagnostics are mandatory.

Every component has built-in observability.

- Flight recorder for query execution
- Timing logs for each stage
- Candidate diagnostics in ranking
- Health checks for all subsystems

**Why:** You can't fix what you can't see. The system must be transparent about its decisions.

**Violation:** Code that's opaque about what it's doing.

---

## 7. Benchmarks are required before merge.

Every change that affects performance or accuracy must be benchmarked.

- Accuracy benchmarks (recall@K)
- Latency benchmarks (ms per query)
- Memory usage benchmarks
- Regression tests

**Why:** The system is optimized for 150ms on 4GB RAM. Without benchmarks, we don't know when we break that.

**Violation:** Merging without running the benchmark suite.

---

## 8. Every subsystem owns one responsibility.

One thing, and one thing only.

| Subsystem | Responsibility |
|-----------|----------------|
| `MemoryDB` | Store and retrieve records |
| `VectorStore` | Semantic search |
| `RetrievalEngine` | Candidate retrieval |
| `MemoryRanker` | Candidate scoring |
| `RankingPipeline` | Orchestrate ranking |
| `ContextBuilder` | Token budgeting |
| `LLMAdapter` | LLM communication |

**Why:** Single responsibility makes code testable, debuggable, and understandable.

**Violation:** A module that does two unrelated things.

---

## 9. Architectural simplicity beats clever code.

Prefer simple, obvious solutions over clever, complex ones.

- If it's hard to explain, it's probably wrong
- If it requires a comment to understand, it's probably wrong
- If it's clever, it's probably wrong

**Why:** Simple code is easy to maintain, debug, and extend. Clever code is clever once, then a burden forever.

**Violation:** Complex optimizations without clear justification.

---

## 10. Future versions must remain backward understandable.

The codebase should always be understandable to someone who knows the principles.

- Names should be obvious
- Structure should match the architecture
- Every file should have a clear purpose
- No "magic" without comments

**Why:** The system is being built by multiple people over multiple years. If V5 can't understand V3, we've failed.

**Violation:** Breaking changes without documentation.

---

## Principles in Practice

| Principle | Implementation |
|-----------|----------------|
| 1. Retrieval retrieves | `RetrievalEngine` only does search |
| 2. Ranking ranks | `MemoryRanker` only scores |
| 3. Memory never generates | No LLM in `MemoryDB` |
| 4. Records everywhere | Pydantic models for all data |
| 5. Subsystems replaceable | Interfaces, not implementations |
| 6. Diagnostics mandatory | Flight recorder, timing, diagnostics |
| 7. Benchmarks required | `benchmark/` suite runs on every PR |
| 8. One responsibility | Each module does one thing |
| 9. Simplicity > clever | Obvious > clever |
| 10. Backward understandable | Clear naming, no magic |

---

## See Also

- `03_system_architecture.md` — Architecture overview
- `project_manifesto.md` — The vision
- `02_project_overview.md` — Project intro
