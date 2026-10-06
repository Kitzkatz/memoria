# Memoria 1.0.0

**Local-first, LLM-agnostic memory infrastructure with parallel hybrid retrieval, multi-signal ranking, declarative type routing, temporal retrieval, and a plugin-based architecture.**

Memoria provides a complete memory substrate without requiring a cloud service, API key, GPU, or even an LLM.

* **CPU-only** · **~4 GB RAM capable** · **No cloud dependency** · **No API keys required**
* **LLM optional** · **MIT licensed** · **Python**
* **CLI · TUI · GUI · API · MCP**
* **Pluggable retrieval and extension architecture**

**Memoria 1.0.0 is the first official public release.**

## Documentation

Full documentation, architecture notes, configuration reference, adapter documentation, retrieval details, and benchmark methodology:

**https://kitzkatz.github.io/memoria/**

---

## What Is Memoria?

Memoria is memory infrastructure first — built to store, retrieve, rank, route, and manage information independently of whichever language model happens to consume it.

Rather than treating memory as a thin vector-search layer, Memoria combines multiple retrieval and ranking signals, with temporal retrieval running as a separately evaluable path that merges in at the end:

```text
                    ┌─────────────────────┐
                    │       Query         │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
          Base Retrieval              Temporal Retrieval
                 │                           │
        ┌────────┼────────┐                  │
        │        │        │                  │
      FAISS    BM25    Graph             Temporal
        │        │        │                  │
        └────────┴────────┘                  │
                 │                           │
               Fusion                        │
                 │                           │
            Base Ranking                     │
                 │                           │
                 └─────────────┬─────────────┘
                               │
                         Final RRF Fusion
                               │
                         Ranked Results
```

The system remains useful without an LLM. An LLM can be connected later as a consumer, generator, or optional reasoning layer.

---

# Results

## LongMemEval-S

Evaluated against **LongMemEval-S** on 500 questions: 470 were retrieval-evaluable, 30 were intentional abstentions. Of the 470, **468 returned results** and **2 were retrieval failures**.

| Metric          |     Result |
| ---------------- | ---------: |
| Recall@1          |  **89.8%** |
| Recall@3          |  **96.4%** |
| Recall@5          |  **97.9%** |
| Recall@10         |  **98.9%** |
| Recall@30         |  **99.6%** |
| Recall@50         |  **99.6%** |
| Session NDCG@10   | **0.9257** |

Full benchmark configuration, reproduction steps, and dataset handling are documented separately.

## Synthetic Benchmark

A separate 4,632-question synthetic benchmark:

| Metric                    |        Result |
| --------------------------- | ------------: |
| Questions                    |     **4,632** |
| Queries returning results     |    **99.46%** |
| Recall@1                      |    **32.60%** |
| Recall@3                      |    **39.98%** |
| Recall@5                      |    **52.03%** |
| Recall@10                     |    **78.76%** |
| Average query time             | **~122.3 ms** |

These two benchmarks measure different things — full methodology and interpretation live in the benchmark documentation.

---

# Benchmark Environment

The published numbers above were run on deliberately modest hardware, not a dedicated workstation:

**Intel Celeron N4020 @ 1.10 GHz · ~3.7 GiB RAM · no GPU · Debian Linux · CPU-only**

### Memory Usage

Measured peak resident memory on this system:

| Workload                   |     Peak RSS |
| --------------------------- | -----------: |
| LongMemEval — cached           | **2.65 GiB** |
| LongMemEval — uncached         | **2.65 GiB** |
| Average individual query        |  **580 MiB** |

These are real measurements from the benchmark environment, not estimates.

---

# Architecture

Memoria is organized around independent subsystems rather than a monolithic retrieval pipeline.

```text
                         MemorySystem
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
       Storage             Routing             Query
          │                   │                   │
      Database          Type / Signal       Processing
      Vector Store          Routing              │
          │                   │                  │
          └───────────────────┼──────────────────┘
                              │
                         Blackboard
                              │
                         Scheduler
                              │
              ┌───────────────┼───────────────┐
              │               │               │
            FAISS           BM25           Graph
              │               │               │
           Phrase         Attribute        Temporal
              │               │               │
              └───────────────┴───────────────┘
                              │
                           Fusion
                              │
                           Ranking
                              │
                         Final Results
```

Major subsystems: FAISS vector retrieval, BM25 lexical retrieval, graph retrieval, phrase retrieval, attribute retrieval, temporal retrieval, multi-signal fusion, ranking/reranking, declarative type routing, signal routing, blackboard scheduling, persistent storage, plugin hooks, and external adapters.

---

# Retrieval

Memoria runs multiple retrieval strategies in parallel rather than relying on a single embedding search.

**FAISS** — Semantic vector retrieval using local embeddings.

**BM25** — Lexical retrieval for exact terms, identifiers, names, and textual matches embeddings may underweight.

**Graph** — Relationship-oriented retrieval through stored graph structure.

**Phrase** — Phrase-sensitive matching for exact or near-exact textual relationships.

**Attribute** — Metadata and attribute-based retrieval.

**Temporal** — Runs independently of the base retrieval path, identifying temporal intent and producing its own ranking signal without calling the base Fusion worker. The result merges with the base retrieval/ranking path through a final fusion stage — keeping temporal retrieval independently measurable and ablatable rather than folded into the core pipeline.

---

# Ranking

Retrieval candidates are combined and ranked using multiple signals rather than vector similarity alone: semantic similarity, lexical relevance, graph relationships, phrase matches, attribute matches, temporal relevance, any additional registered signals, and reranking.

Individual signals can be evaluated independently and extended through plugins.

---

# Blackboard and Scheduler

Memoria uses a blackboard-based execution model to coordinate independent retrieval and processing workers. Workers publish results to shared execution state while the scheduler controls task execution and completion policies.

This supports parallel execution, completion policies, time budgets, worker isolation, extensible scheduling, and deterministic orchestration.

---

# Storage

Persistent memory storage is kept separate from retrieval implementation: relational storage, vector indexes, metadata, graph relationships, retrieval-specific indexes, and query/memory management. Retrieval workers can evolve without the underlying memory representation becoming tied to one strategy.

---

# Integrations

## Obsidian

A source-ingestion adapter for Obsidian vaults — markdown notes, frontmatter, wikilinks, note metadata, and structured ingestion into Memoria. Tested against real vault data and a dedicated adapter test suite.

## MCP

Memoria includes a Model Context Protocol server:

```bash
memoria-mcp
```

Available operations: `memory_search`, `memory_store`, `memory_store_many`, `memory_fetch`, `memory_update`, `memory_delete` — letting MCP-compatible clients and agents use Memoria as an external memory system.

---

# Plugin System

A configuration-driven plugin architecture spanning analysis, evaluation, feedback, ingestion, lifecycle, query, ranking, retrieval, routing, scheduler, and storage.

Plugins can register retrieval workers, ranking signals, rerankers, query processors, type/signal routers, storage backends, vector stores, database backends, extractors, entity recognizers, benchmark adapters, feedback recorders, and scheduler policies.

Loading supports automatic discovery, explicit activation, allowlists, denylists, runtime enable/disable, local plugins, and entry-point plugins. The denylist takes precedence when both allow and deny rules apply.

### Plugin Generator

```bash
memory plugin create
```

Interactively creates a plugin module, optional tests, optional README documentation, and the corresponding hook configuration.

---

# Interfaces

Python API · CLI · TUI · GUI · HTTP API · MCP

### CLI

```bash
memory --help
```

```bash
memory info
memory store "Memoria stores this locally."
memory recall "What does Memoria store?"
```

---

# Installation

## PyPI

```bash
python -m pip install kitzkatz-memoria
```

## From Source

```bash
git clone https://github.com/KitzKatz/Memoria.git
cd memoria/memoria
python -m pip install -e .
```

No external LLM or API key required for core memory and retrieval functionality.

---

# Quick Start

```bash
memory store "Memoria is a local-first memory system."
memory recall "What is Memoria?"
memory info
memoria-mcp
memory plugin create
```

---

# Configuration

Layered: built-in defaults → user configuration → environment variables.

Covers database, embeddings, retrieval, ranking, scheduling, temporal retrieval, plugins, interfaces, logging, and API services. Environment variables use the `MEMORY_` prefix.

```toml
PLUGIN_ENABLED = true
PLUGIN_DIR = "plugins"
PLUGIN_AUTO_LOAD = true

PLUGIN_ENABLED_PLUGINS = []
PLUGIN_DISABLED_PLUGINS = []
```

```bash
export MEMORY_PLUGIN_AUTO_LOAD=true
```

See the configuration documentation for the complete settings reference.

---

# Benchmarking

Memoria includes infrastructure for evaluating retrieval, latency, ranking, and memory-system behavior — LongMemEval, LoCoMo, synthetic workloads, benchmark analysis, result formatting, batch loading, memory extraction, and question extraction.

Full reproduction commands, dataset preparation, cache behavior, and analysis tooling are documented separately rather than duplicated here.

---

# Performance

On the benchmark system described above, a representative query:

| Stage                 |       Average |
| ------------------------ | ------------: |
| Query processing          |      ~0.24 ms |
| Embedding                  |      ~79.8 ms |
| Retrieval                   |     ~101.9 ms |
| Scheduler wait                |      ~35.2 ms |
| Database                       |      ~30.9 ms |
| Ranking                         |      ~0.28 ms |
| Response construction            |       ~3.4 ms |
| **Total**                        | **~208.5 ms** |

Configured retrieval deadline: **~125 ms**. Actual latency varies with workload, cache state, memory size, candidate counts, and hardware — these numbers come from the Celeron N4020 / ~3.7 GiB RAM / CPU-only system above, not a GPU-equipped dev machine.

---

# Project Structure

```text
memoria/
├── blackboard/
├── benchmark/
│   ├── github/
│   └── obsidian/
├── cache/
├── core/
├── db/
├── graph/
├── ingestion/
├── memory/
├── memoria_mcp/
├── plugins/
├── ranking/
├── retrieval/
├── routing/
├── routes/
├── shared/
├── system/
├── tools/
├── cli.py
├── tui.py
├── demo.py
├── pyproject.toml
└── README.md
```

The repository also contains benchmark datasets, test infrastructure, development tooling, and documentation.

---

# Requirements

* Python, Linux recommended
* ~4 GiB RAM recommended, CPU-only supported, GPU optional
* No cloud service, no API key, LLM optional

### Tested Hardware

**Intel Celeron N4020 @ 1.10 GHz · ~3.7 GiB RAM · no GPU · Debian Linux**

Despite the modest hardware, the full LongMemEval workload reached a measured peak RSS of **2.65 GiB**, with an average individual query peaking around **580 MiB**.

---

# License

MIT License.

---

## Memoria

**Memory infrastructure first.**
Local. Composable. Measurable. Extensible. Independent of any particular LLM.
