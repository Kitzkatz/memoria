---
title: Memoria
description: Local-first, LLM-agnostic memory substrate for persistent contextual retrieval.
---

# Memoria

**Local-first, LLM-agnostic memory substrate with parallel hybrid retrieval, multi-signal ranking, declarative type routing, and a plugin-based architecture.**

- **4GB RAM · CPU-only · No cloud · No API keys required · MIT licensed**
- Built solo. Designed to be extended rather than replaced.

Memoria is a configurable **memory/retrieval substrate**, not a chatbot-specific memory implementation. It stores memories, routes queries by memory type, retrieves candidates using parallel workers, ranks and finalizes results, constructs context, and exposes the system through CLI, TUI, GUI, and API interfaces.

---

## Where To Start

| I want to… | Read this |
|------------|-----------|
| Install Memoria and run my first query | [Getting Started](getting-started.md) |
| Understand every setting | [Configuration](configuration.md) |
| Build an adapter for a new dataset | [Adapters](adapters.md) |
| Understand how retrieval works | [Retrieval](retrieval.md) |
| See benchmark methodology and results | [Benchmarks](benchmarks.md) |
| Understand the architecture | [Architecture](ARCHITECTURE.md) |
| Write a plugin | [Plugins](PLUGINS.md) |

---

## What It Is

Memoria is a fully local memory system for LLMs and other applications that need persistent contextual retrieval.

Retrieval workers can include:

- **FAISS** — semantic retrieval
- **BM25** — lexical retrieval
- **Graph** — entity/relationship traversal
- **Phrase** — phrase matching
- **Attribute** — structured attribute retrieval
- **Fusion** — combined retrieval strategies

Retrieval workers are coordinated through a **blackboard/scheduler architecture** with declarative completion policies rather than requiring the query handler to synchronously wait for every worker.

**You own your data. No cloud. No subscription.**

---

## Design Goals

- Local-first
- LLM-agnostic at the memory layer
- Retrieval responsible for **finding** candidates; ranking responsible for **deciding usefulness**
- Independently replaceable retrieval workers
- Extensible through plugins, not forks

---

## Project History & Design

- [Project Overview](02_project_overview.md)
- [Project Status](01_project_status.md)
- [Dataflow](04_Dataflow.md)
- [Design Principles](05_Design_Principles.md)
- [Design Intent](07_design_intent.md)
- [Roadmap](06_roadmap.md)
- [Release Strategy](09_release_strategy.md)
- [Manifesto](project_manifesto.md)
- [V3](v3.md) · [V4](v4.md) · [V5](v5.md)

---

## License

MIT
