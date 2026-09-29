# Memoria V4.5.1

**Local-first, LLM-agnostic memory system with parallel hybrid retrieval, multi-signal ranking, declarative type routing, temporal retrieval, and a plugin-based architecture.**

**~4 GB RAM · CPU-only · No cloud · No API keys required · MIT licensed**

Memoria is a memory substrate, not an LLM application. An LLM can be plugged in when needed, but retrieval, ranking, storage, routing, and evaluation run independently of any model.

**V4.5.1 is the current development release.** The upcoming v1.0.0 release is the first official public release of Memoria.

---

## Documentation

Full documentation: **https://kitzkatz.github.io/memoria/**

Covers installation, configuration, retrieval architecture (including Temporal), the Obsidian adapter, the MCP server, CLI/GUI/TUI usage, the quick-start demo, and performance tuning.

**A PyPI package is coming soon** — until then, install by cloning the repo (see Quick Start below).

---

## What is Memoria?

Memoria is a local memory system built to provide persistent, searchable memory without requiring a hosted model or cloud service.

Rather than relying on a single embedding search, the core combines several retrieval signals:

* Dense vector retrieval (FAISS)
* BM25 lexical retrieval
* Graph retrieval
* Phrase retrieval
* Attribute retrieval
* Temporal retrieval
* Reciprocal-rank fusion
* Multi-signal ranking
* MMR diversification
* Declarative type routing
* Blackboard-based scheduling

It's usable through CLI, TUI, GUI, API, Python interfaces, and MCP. An LLM is optional.

---

## Results

### LongMemEval-S

**500 questions · 470 retrieval-evaluable · 30 intentional abstentions**

| Metric                           |        Result |
| --------------------------------- | ------------: |
| Retrieval-evaluable questions      | **470 / 500** |
| Questions with retrieved results   | **468 / 470** |
| Actual retrieval failures          |   **2 / 470** |
| R@1                                |     **89.8%** |
| R@3                                |     **96.4%** |
| R@5                                |     **97.9%** |
| R@10                               |     **98.9%** |
| R@30                               |     **99.6%** |
| R@50                               |     **99.6%** |
| Session NDCG@10                    |    **0.9257** |

The 30 abstention questions are excluded from retrieval metrics by design. Of the remaining 470 retrieval-evaluable questions, 468 returned results and 2 were genuine retrieval failures.

### Synthetic Retrieval Benchmark

**4,632 questions · CPU-only · ~4 GB RAM**

| Metric                |        Result |
| ---------------------- | ------------: |
| Retrieved               |    **99.46%** |
| R@1                     |    **32.60%** |
| R@3                     |    **39.98%** |
| R@5                     |    **52.03%** |
| R@10                    |    **78.76%** |
| Average query latency   | **~122.3 ms** |

The two benchmarks measure different things: LongMemEval-S evaluates retrieval against a real long-context memory benchmark; the synthetic set is a larger workload built for system-level performance testing.

---

## Architecture

```text
Query
  ↓
Query Processing
  ↓
Routing
  ↓
Parallel Retrieval (FAISS · BM25 · Graph · Phrase · Attribute · Temporal)
  ↓
Scheduler / Blackboard
  ↓
Candidate Records
  ↓
Ranking
  ↓
MMR
  ↓
Finalization
  ↓
Context Construction
  ↓
Results
```

FAISS and BM25 are combined first through reciprocal-rank fusion; Temporal retrieval runs as a separate signal and, when enabled, is merged in through a second-stage RRF pass:

```text
FAISS ─┐
       ├──> Fusion ──┐
BM25 ──┘             ├──> second-stage RRF ──> final candidates
Temporal ────────────┘
```

This keeps Temporal retrieval evaluable on its own, without requiring changes to the existing FAISS/BM25 fusion path. Every retrieval mechanism runs independently; the scheduler coordinates when each has completed.

---

## Retrieval

**FAISS** — Dense semantic retrieval over vector embeddings.
Default embedding model: `all-MiniLM-L6-v2` · Vector dimension: `384`

**BM25** — Lexical retrieval for exact terms, names, identifiers, and vocabulary semantic embeddings may miss. Maintains its own scoring signal and contributes to fusion and ranking independently.

**Graph** — Retrieves memories through entity and relationship connections. Uses canonical integer entity IDs at the retrieval boundary while preserving name-based lookup.

**Phrase** — Targets phrase-level lexical matches and exact textual relationships.

**Attribute** — Retrieves memories through structured metadata and attribute matches.

**Fusion** — Combines FAISS and BM25 via reciprocal-rank fusion, letting dense and lexical retrieval contribute independently before ranking.

**Temporal** — Retrieves memories using temporal constraints and temporal intent extracted from queries: exact dates, years, before/after constraints, date ranges, and relative temporal queries. Runs independently of the FAISS/BM25 fusion path and can be enabled or disabled on its own, so it's evaluable separately from the rest of the pipeline. The current implementation is usable but under active refinement — treat it as an additional retrieval capability rather than a claim of mature temporal reasoning.

---

## Ranking

Candidate records pass through a multi-signal ranking stage after retrieval. Signals include:

* Semantic similarity
* Token similarity
* TF-IDF
* Entity overlap
* Subject similarity
* BM25
* Type/routing information
* Other candidate metadata

Current ranking weights:

| Signal   | Weight |
| -------- | -----: |
| Semantic | 0.1195 |
| Token    | 0.3107 |
| TF-IDF   | 0.2929 |
| Entity   | 0.0272 |
| Subject  | 0.0980 |
| BM25     | 0.0762 |

Ranking is independent from the retrieval workers, so the two can be ablated separately.

**MMR** runs after ranking to reduce redundancy while keeping highly relevant candidates. It records relevance score, diversity score, MMR score, selection order, and reordering diagnostics — making diversification measurable rather than an opaque post-processing step.

---

## Blackboard and Scheduler

Retrieval workers run through a shared blackboard/scheduler architecture. Workers publish candidate results and completion state rather than controlling the query lifecycle directly; completion policies determine when enough retrieval work has finished to proceed.

This provides parallel retrieval, worker isolation, explicit completion policies, timing diagnostics, extensible retrieval workers, and controlled query deadlines.

---

## Storage

Storage is kept separate from retrieval indexes. The pipeline handles:

1. Ingestion hooks
2. Extraction
3. Metadata processing
4. Scoring
5. Embedding
6. Database insertion
7. Relationship construction
8. Vector indexing
9. Cache updates
10. Post-storage hooks

Batch ingestion has matching bulk operations for embedding, database insertion, inverted-index construction, BM25 rebuilding, relationship construction, and vector-store updates.

---

## Integrations

### Obsidian

The Obsidian adapter imports an existing Obsidian vault into Memoria. Notes are parsed into structured sections — preserving frontmatter, wiki-links, tasks, and vault-relative paths — and ingested through Memoria's normal storage and indexing pipeline via `BatchLoader`.

```python
from memory.memory_controller import MemoryController
from adapters.obsidian_adapter import ObsidianAdapter

controller = MemoryController()
adapter = ObsidianAdapter("/path/to/your/obsidian-vault")

count = adapter.load_into_memory(controller)
print(f"Ingested {count} notes")
```

`adapter.stats()` returns a quick summary (record count, templates, linked notes, tasks) without touching storage. Pass `include_templates=False` to skip template notes during ingestion.

After ingestion, the imported notes can be queried through Memoria normally.

### MCP

Memoria includes an MCP server exposing the memory system to compatible AI agents and MCP clients. It's a thin interface over the existing `MemoryController` — no separate retrieval or storage system underneath.

Available operations: `memory_search`, `memory_store`, `memory_store_many`, `memory_fetch`, `memory_update`, `memory_delete`.

```bash
memoria-mcp
```

This lets any MCP-compatible client or agent use Memoria as a persistent local memory backend.

---

## Plugin Architecture

Built around explicit hook specifications — **10 plugin subsystems, 39 hook specifications**:

| Subsystem  | Hooks |
| ---------- | ----: |
| Lifecycle  |     6 |
| Ranking    |     4 |
| Storage    |     4 |
| Ingestion  |     4 |
| Scheduler  |     4 |
| Routing    |     4 |
| Evaluation |     4 |
| Retrieval  |     3 |
| Query      |     3 |
| Feedback   |     3 |

Plugins extend system behavior without changing the core memory pipeline.

As of V4.5.1, the plugin system is operational end-to-end — hooks fire correctly across the full pipeline. This has been verified as thoroughly as testing allows; if you hit a hook that misbehaves, please file an issue.
---

## Interfaces

CLI, TUI, GUI, API, Python, and MCP interfaces sit over the same underlying system. The memory substrate remains independent of whichever interface or integration is used.

---

## Quick Start

```bash
git clone https://github.com/KitzKatz/Memoria.git
cd Memoria
```

See the documentation for configuration and interface setup: **https://kitzkatz.github.io/memoria/**

### Demo

A small local demo exercises the basic workflow — storing memories, hybrid retrieval, a temporal query, and fetching a memory by ID — without requiring an external LLM or cloud service.

```bash
python demo.py
```

It's intentionally small: the goal is to verify the install and show the basic system, not reproduce a benchmark.

---

## Configuration

Memoria runs locally and is configured via settings and environment variables, covering embedding models, LLM integration, retrieval workers, ranking, MMR, candidate limits, scheduler behavior, context limits, caching, storage, and benchmark configuration.

Environment variables use the `MEMORY_` prefix.

---

## LongMemEval-S

Memoria includes an adapter for the LongMemEval-S benchmark. It:

* Isolates each question's haystack
* Stores every non-empty turn
* Preserves session and turn metadata
* Queries Memoria and preserves ranked candidates
* Calculates session- and turn-level retrieval metrics
* Excludes intentional abstentions from retrieval metrics
* Supports reproducible per-question caches
* Records dataset, model, settings, and commit metadata

It supports five retrieval modes for ablation testing — dense (FAISS only), bm25, raw (FAISS + BM25, no ranking), fusion, and full (FAISS + BM25 + Graph + Phrase + Attribute, ranking enabled):

```bash
python benchmark/longmemeval_adapter.py \
    --retrieval-mode full \
    --output benchmark_output/results/full.json
```

Swap `--retrieval-mode` for `dense`, `bm25`, `raw`, or `fusion` to run each ablation. Compare them once all five have run:

```bash
python benchmark/benchmark_analyzer.py --compare \
    benchmark_output/results/dense.json \
    benchmark_output/results/bm25.json \
    benchmark_output/results/raw.json \
    benchmark_output/results/fusion.json \
    benchmark_output/results/full.json
```

```text
Mode                     R@1     R@3     R@5    R@10   NDCG@10
----------------------------------------------------------------
...
```

The comparison uses the adapter's session-level `recall_any` and session-level NDCG@10 — tied to the benchmark's actual retrieval evaluation, not a "returned any candidate" diagnostic.

---

## Benchmark Analyzer

```bash
python benchmark/benchmark_analyzer.py results.json
python benchmark/benchmark_analyzer.py --compare a.json b.json ...
python benchmark/benchmark_analyzer.py --all
```

Reports retrieval statistics, timing, MMR diagnostics, Recall@K, and official session/turn metrics.

---

## Performance

A representative LongMemEval-S run on CPU-only hardware:

| Stage                 |       Average |
| ---------------------- | ------------: |
| Query processing        |      ~0.24 ms |
| Embedding                |      ~79.8 ms |
| Retrieval                |     ~101.9 ms |
| Scheduler wait            |      ~35.2 ms |
| Database                  |      ~30.9 ms |
| Ranking                   |      ~0.28 ms |
| Response construction     |       ~3.4 ms |
| **Total**                 | **~208.5 ms** |

Configured retrieval deadline: **125 ms**. Each stage is timed independently so regressions surface at the stage level rather than hiding inside one aggregate number.

---

## Synthetic Benchmark

**4,632 questions**, built primarily for retrieval and performance analysis, runnable on modest CPU-only hardware without a GPU or hosted inference service.

| Metric                |    Result |
| ---------------------- | --------: |
| Questions               |     4,632 |
| Retrieved                |    99.46% |
| R@1                      |    32.60% |
| R@3                      |    39.98% |
| R@5                      |    52.03% |
| R@10                     |    78.76% |
| Average query latency    | ~122.3 ms |

---

## Project Structure

```text
memoria/
├── benchmark/
│   ├── longmemeval_adapter.py
│   ├── benchmark_analyzer.py
│   └── ...
├── memoria_mcp/
│   ├── __init__.py
│   └── server.py
├── adapters/
│   └── obsidian_adapter.py
├── core/
│   ├── bootstrap.py
│   ├── context_builder.py
│   ├── llm_adapter.py
│   ├── logger.py
│   └── token_estimator.py
├── memory/
├── ranking/
├── retrieval/
├── system/
│   ├── blackboard/
│   │   ├── core/
│   │   ├── scheduler/
│   │   ├── workers/
│   │   └── consolidator/
│   └── ...
├── gui/
├── cli/
├── demo.py
└── ...
```

---

## Requirements

* ~4 GB RAM
* CPU-only
* Linux
* Python
* No cloud service
* No API key

An LLM is optional, not a prerequisite for the memory substrate.

---

## License

MIT License.

---

**Memoria is memory infrastructure first: local, composable, measurable, and independent of any particular LLM.**
