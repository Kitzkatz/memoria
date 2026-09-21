# Design Intent

Memory Daemon exists to provide:

- **Deterministic**
- **Local**
- **Inspectable**
- **Extensible**

long-term memory for language models.

---

## Core Principles

The project intentionally avoids cloud dependence.

**The user owns their memories.**

---

## Decision Framework

Every decision should support:

### Transparency
- Every component is observable
- Flight recorder for debugging
- Diagnostics are mandatory
- No black boxes

### Performance
- 150ms query latency target
- 4GB RAM footprint
- CPU-only by design
- Parallel execution where possible

### Modularity
- Subsystems are replaceable
- Clear interfaces between layers
- Single responsibility per component
- Plugins for future extension

### Deterministic Behavior
- Same query → same result
- No random sampling in retrieval
- Reproducible rankings
- Benchmark validated

### Simple Extension
- Add new signals without breaking ranking
- Add new retrievers without changing core
- Add new memory types without schema changes
- Swap LLMs without code changes

---

## What This Means in Practice

| Principle | Implementation |
|-----------|----------------|
| Deterministic | Fixed seed, no randomness in retrieval |
| Local | SQLite, FAISS on disk, no cloud calls |
| Inspectable | Diagnostics on every candidate |
| Extensible | Interfaces, not implementations |

---

## Tradeoffs We Accept

| Tradeoff | Why |
|----------|-----|
| CPU-only | GPU adds cost, complexity, and cloud dependence |
| SQLite | Local-first, portable, no network |
| Python | Accessibility, extensibility, community |
| No cloud sync | Privacy, ownership, simplicity |

---

## Anti-Goals

| We Don't | Because |
|----------|---------|
| Cloud sync | User ownership of data |
| GPU dependence | Local-first, accessible |
| Proprietary formats | Open, inspectable |
| Opaque ranking | Transparency |
| Vendor lock-in | Modular, replaceable |

---

## See Also

- `project_manifesto.md` — Vision and philosophy
- `05_Design_Principles.md` — Ten immutable rules
- `03_system_architecture.md` — Architecture overview
