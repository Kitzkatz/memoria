# Release Strategy

Memory Daemon follows a phased release strategy aligned with architectural maturity.

Each version targets a specific audience and serves a distinct purpose.

---

## Version Overview

| Version | Type | Audience | Goal |
|---------|------|----------|------|
| V3 | Engineering Release | Recruiters, Hiring Managers, Engineers | Demonstrate architecture |
| V4 | Architecture Release | Developers, Researchers | Expand reasoning architecture |
| V5 | Community Release | Everyone | Stable public platform |

---

## V3 — Engineering Release

**Status:** ✅ Complete (Archived)

**Audience:**
- Recruiters
- Hiring managers
- Engineers

**Goal:** Demonstrate architecture.

**Key Deliverables:**
- Complete memory engine
- Retrieval + Ranking pipeline
- Benchmark suite
- Synthetic world testing
- Documentation

**Success Criteria:**
- Runs locally on 4GB RAM
- 150ms query latency
- Open source, inspectable

---

## V4 — Architecture Release

**Status:** 🚧 In Progress (Current)

**Audience:**
- Developers
- Researchers
- Early adopters

**Goal:** Expand reasoning architecture.

**Key Deliverables:**
- Blackboard architecture
- Task scheduler
- Goals and planning
- Computation graphs
- Reasoning nodes
- Full API

**Success Criteria:**
- Reasoning over memory
- Parallel task execution
- Goal-driven workflows
- Extensible reasoning nodes

---

## V5 — Community Release

**Status:** 🔜 Planned

**Audience:**
- Everyone
- Open source community
- Researchers
- Developers
- End users

**Goal:** Stable public platform.

**Key Deliverables:**
- Stable API
- Plugin system
- Complete documentation
- PyPI package
- Docker images
- Community infrastructure

**Success Criteria:**
- Public adoption
- Community contributions
- Stable API surface
- Production ready

---

## Release Cadence

| Version | Target Date | Status |
|---------|-------------|--------|
| V1 | Archived | ✅ Complete |
| V2 | Archived | ✅ Complete |
| V3 | Archived | ✅ Complete |
| V4 | Current | 🚧 In Progress |
| V5 | Future | 🔜 Planned |

---

## Versioning Philosophy

- **Major versions** (V3, V4, V5) — Architectural shifts
- **Minor versions** — Feature additions
- **Patches** — Bug fixes

No semantic versioning yet. Versions are architectural milestones.

---

## See Also

- `01_project_status.md` — Current status
- `06_roadmap.md` — Detailed roadmap
- `project_manifesto.md` — Vision and philosophy
