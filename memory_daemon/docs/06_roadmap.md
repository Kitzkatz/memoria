# Roadmap

## V3 — Engineering Release

**Goal:** Release a documented, benchmarked local memory engine suitable for engineering review.

**Status:** ✅ Complete

### Checklist

| Item | Status |
|------|--------|
| Memory extraction | ✅ Complete |
| Embeddings | ✅ Complete |
| Vector search | ✅ Complete |
| Graph relationships | ✅ Complete |
| Ranking pipeline | ✅ Complete |
| Benchmarks | ✅ Complete |
| Synthetic World | ✅ Complete |
| Regression suite | ✅ Complete |
| CLI | ✅ Complete |
| README | ✅ Complete |
| API examples | ✅ Complete |
| Installation guide | ✅ Complete |
| Release package | ✅ Complete |

**V3 is archived.** The codebase now lives as V4.

---

## V4 — Architecture Release

**Goal:** Transform Memory Daemon from a memory engine into a cognitive architecture capable of coordinating multiple reasoning systems while preserving modularity and deterministic behavior.

**Status:** 🚧 In Progress (~90%)

**Release Date:** TBD (Current)

---

### Major Goals

#### Blackboard Architecture
Create a central reasoning workspace where independent modules collaborate without being tightly coupled.

**Purpose:**
- Shared reasoning state
- Decoupled components
- Cooperative processing
- Easier future expansion

**Status:** ✅ Complete

---

#### Scheduler
Introduce intelligent task scheduling.

**Responsibilities:**
- Background memory maintenance
- Deferred processing
- Memory aging
- Consolidation timing
- Pipeline orchestration

**Status:** ✅ Complete

---

#### Computation Graph
Replace linear execution with graph-based execution.

**Allows:**
- Dynamic execution paths
- Parallel processing
- Conditional routing
- Future optimization

**Status:** 🔄 In Progress

---

#### Memory Interaction
Expand interaction between memory types.

**Examples:**
- Semantic ↔ Episodic
- Goals ↔ Memories
- Structured ↔ Graph
- Graph ↔ Retrieval

**Status:** ✅ Complete

---

#### Memory Consolidation
Allow memories to evolve over time.

**Features:**
- Merge duplicates
- Reinforce important memories
- Create summaries
- Build higher-order concepts

**Status:** ✅ Complete

---

#### Memory Aging
Introduce lifecycle management.

**Stages:**
- Fresh
- Active
- Stable
- Dormant
- Archived

**Status:** ✅ Complete (via Pruner)

---

#### Relationship Intelligence
Replace placeholder relationships with richer extraction.

**Targets:**
- Better entity resolution
- Typed relationships
- Confidence scores
- Graph enrichment

**Status:** ✅ Complete

---

#### Planner
Introduce a planning layer capable of coordinating reasoning modules and future autonomous workflows.

**Status:** 🔜 Planned

---

#### Local Tool System
Provide a standardized interface for local tools.

**Examples:**
- File access
- Search
- Diagnostics
- External plugins

**Status:** 🔜 Planned

---

### Success Criteria

V4 is complete when Memory Daemon is no longer simply storing memories but coordinating multiple subsystems through a unified reasoning architecture.

**Progress:** ~90%

---

## V5 — Community Release

**Goal:** Deliver Memory Daemon as a stable, documented, extensible platform suitable for public adoption and long-term maintenance.

**Status:** 🔬 Research / Planning

**Release Date:** TBD

---

### Platform Goals

#### Stable API
Versioned interfaces with backwards compatibility.

**Status:** 🔜 Planned

---

#### Stable Data Schema
Long-term database compatibility with documented migration paths.

**Status:** 🔜 Planned

---

#### Plugin System
Allow developers to extend Memory Daemon without modifying the core.

**Potential plugin categories:**
- Retrieval
- Memory types
- Embeddings
- Tools
- Ranking
- Exporters

**Status:** 🔜 Planned

---

#### Documentation
Complete documentation set including:
- User Guide
- Developer Guide
- Architecture Guide
- API Reference
- Tutorials
- Examples

**Status:** 🟡 In Progress

---

#### Community Infrastructure
- Contribution Guidelines
- Issue Templates
- Discussion Channels
- Roadmaps
- Coding Standards

**Status:** 🔜 Planned

---

#### Packaging
Official release builds.

**Targets:**
- pip
- Docker
- Standalone binaries
- Optional installers

**Status:** 🟡 In Progress

---

#### Optional GUI
A graphical interface for users who prefer visual interaction over the CLI while maintaining feature parity.

**Status:** ✅ Complete (FastAPI GUI)

---

#### Performance Optimization
Focus areas:
- Query latency
- Memory usage
- Graph traversal
- Embedding cache
- Ranking efficiency

**Status:** 🔄 Ongoing

---

#### Memory Compression
Research and implement long-term storage optimization.

**Potential directions:**
- Hierarchical summaries
- Semantic compression
- Redundant memory consolidation
- Archive strategies

**Status:** 🔬 Research (V5 proposal)

---

### Release Vision

By the completion of V5, Memory Daemon should function as a mature local memory platform that is:

- ✅ Stable
- ✅ Extensible
- ✅ Well documented
- ✅ Benchmark validated
- ✅ Community maintainable
- ✅ Suitable for research, personal assistants, robotics, and local AI systems

V5 is the project's first true ecosystem release rather than simply another software version.

---

## Timeline Summary

| Version | Status | Focus |
|---------|--------|-------|
| V1 | ✅ Archived | Core memory |
| V2 | ✅ Archived | Ranking + feedback |
| V3 | ✅ Archived | Type routing + BM25 |
| V4 | 🚧 Current | Reasoning infrastructure |
| V5 | 🔜 Planned | Community release |

---

## See Also

- `01_project_status.md` — Current status
- `02_project_overview.md` — Project overview
- `09_release_strategy.md` — Release plan
- `v5.md` — V5 research proposal
