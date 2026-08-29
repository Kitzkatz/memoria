# Memoria V4.5.1

**Local-first, LLM-agnostic memory system with parallel hybrid retrieval, multi-signal ranking, declarative type routing, and a plugin-based architecture.**

**4GB RAM · CPU-only · No cloud · No API keys required · MIT licensed**

Memoria is designed as a configurable memory/retrieval substrate rather than a chatbot-specific memory implementation.

---

## What It Is

Memoria is a fully local memory system for LLMs and other applications that need persistent contextual retrieval.

It can store memories, route queries by memory type, retrieve candidates using parallel workers, rank and finalize results, construct context, and expose the system through CLI, TUI, GUI, and API interfaces.

Retrieval workers can include:

* **FAISS** — semantic retrieval
* **BM25** — lexical retrieval
* **Graph** — entity/relationship traversal
* **Phrase** — phrase matching
* **Attribute** — structured attribute retrieval
* **Fusion** — combined retrieval strategies

Retrieval workers are coordinated through a **blackboard/scheduler architecture** with declarative completion policies rather than requiring the query handler to synchronously wait for every worker.

**You own your data. No cloud. No subscription.**

---

## Quick Start

```bash
git clone https://github.com/Kitzkatz/memoria.git
cd memoria
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python cli.py store "Kevin Johnson likes ramen."
python cli.py recall "What does Kevin Johnson like?"
```

---

## Interfaces

| Interface | Command          | URL                        |
| --------- | ---------------- | -------------------------- |
| CLI       | `python cli.py`  | Terminal                   |
| TUI       | `python tui.py`  | Terminal                   |
| GUI       | `python gui.py`  | http://localhost:5000      |
| API       | `python main.py` | http://localhost:8000/docs |

---

## CLI Commands

```bash
python cli.py store "Your memory here"
python cli.py recall "What did I say?" --limit 5
python cli.py chat "What does Kevin Johnson like?"
python cli.py set-goal "Finish V4 release" --progress started
python cli.py list-goals --status active
python cli.py info
python cli.py doctor
python cli.py benchmark --limit 100
python cli.py serve --port 8000
python cli.py export memories.json
python cli.py import memories.json
python cli.py config
```

---

## TUI

```bash
python tui.py
```

Commands:

`store`, `recall`, `chat`, `set-goal`, `list-goals`, `graph`, `stats`, `doctor`, `export`, `import`, `quit`

---

## GUI

```bash
python gui.py
```

Open:

`http://localhost:5000`

---

## API

```bash
python main.py
```

Open:

`http://localhost:8000/docs`

Endpoints include:

`/memory/store` · `/memory/query` · `/memory/batch_store` · `/chat` · `/chat/raw` · `/debug/stats` · `/debug/health` · `/maintenance/rebuild_index` · `/benchmark/run`

---

# LongMemEval

Memoria includes an adapter for the **LongMemEval-S** benchmark.

The adapter works from the benchmark's native question/haystack structure rather than converting the dataset into Memoria's original database format.

For each question it can:

* isolate the relevant haystack
* construct the corresponding memory state
* cache embeddings
* reuse cached embeddings across runs
* execute the query
* identify expected answer sessions
* identify expected answer turns
* compare retrieved memories against expected IDs
* record session-level and turn-level retrieval metrics
* record retrieval and ranking diagnostics
* record per-stage timing

The adapter is designed to make repeated evaluation practical even on constrained hardware.

On a **4GB RAM CPU-only laptop**, caching the embeddings for the 500-question evaluation took roughly **a few hours on 4gbs, its faster on better systems**. Subsequent evaluation runs against the cached embeddings take roughly **3-4 minutes**.

### Evaluation Scope

The benchmark contains **500 questions**.

The current metric aggregation reports **470 evaluable questions**. Abstained/non-evaluable records are excluded from the official session- and turn-level metric aggregation.

The metrics below describe **retrieval performance only**. They measure whether expected sessions or turns appear within the retrieved candidate set and how they are ranked.

They do **not** measure:

* end-to-end answer-generation accuracy
* LLM answer quality
* factual correctness of a generated response
* conversational quality

### Current LongMemEval Retrieval Metrics

The current fusion configuration produced the following stored official metrics over the 470 evaluable questions:

|  K | Session Recall Any | Session Recall All | Session NDCG | Turn Recall Any | Turn Recall All | Turn NDCG |
| -: | -----------------: | -----------------: | -----------: | --------------: | --------------: | --------: |
|  1 |              89.6% |              31.7% |       0.8957 |           26.4% |            7.4% |    0.2638 |
|  3 |              96.2% |              80.4% |       0.8993 |           48.7% |           19.4% |    0.3021 |
|  5 |              97.7% |              86.4% |       0.9095 |           60.6% |           30.9% |    0.3551 |
| 10 |              98.7% |              93.2% |       0.9236 |           75.3% |           47.4% |    0.4156 |
| 30 |              99.4% |              98.5% |       0.9319 |           87.2% |           67.9% |    0.4664 |
| 50 |              99.4% |              98.5% |       0.9319 |           91.1% |           77.9% |    0.4824 |

**Recall Any** indicates whether at least one expected item was retrieved within K.

**Recall All** indicates whether all expected items were retrieved within K.

**NDCG** measures the ranking of expected items within the retrieved results.

Session-level metrics evaluate retrieval of the expected answer session. Turn-level metrics evaluate retrieval of the specific expected answer turn.

### Retrieval Ablation

The LongMemEval adapter also supports retrieval configuration ablations.

The following results were produced using the same evaluation dataset and retrieval metric definitions:

| Configuration |       R@1 |       R@3 |       R@5 |      R@10 |    NDCG@10 |
| ------------- | --------: | --------: | --------: | --------: | ---------: |
| Dense         |     87.0% |     94.0% |     97.2% |     98.5% |     0.9083 |
| BM25          |     81.1% |     91.5% |     93.6% |     96.8% |     0.8540 |
| Raw           |     62.8% |     72.8% |     74.7% |     77.2% |     0.5720 |
| **Fusion**    | **89.6%** | **96.2%** | **97.7%** | **98.7%** | **0.9236** |
| Full          |     34.0% |     49.6% |     55.7% |     63.6% |     0.3649 |

These values represent the retrieval evaluation for each configuration. They are not end-to-end LongMemEval answer-generation scores.

### Independent Desktop Run

The fusion configuration was also evaluated on a second system:

| Configuration |       R@1 |       R@3 |       R@5 |      R@10 |    NDCG@10 |
| ------------- | --------: | --------: | --------: | --------: | ---------: |
| Dense         |     87.0% |     94.0% |     97.2% |     98.5% |     0.9083 |
| BM25          |     81.1% |     91.5% |     93.6% |     96.8% |     0.8540 |
| Raw           |     62.8% |     72.8% |     74.7% |     77.2% |     0.5720 |
| **Fusion**    | **89.8%** | **96.4%** | **97.9%** | **98.9%** | **0.9257** |
| Full          |     34.0% |     49.6% |     55.7% |     63.6% |     0.3649 |

The second run is included as a separate measurement rather than being combined with the primary run.

---

# Synthetic Benchmark

Memoria also includes a larger synthetic benchmark for evaluating retrieval and ranking behavior across controlled workloads.

Full benchmark:

```text
Questions:       4632
Retrieved:       99.46%

Recall@1:         32.60%
Recall@3:         39.98%
Recall@5:         52.03%
Recall@10:        78.76%

Avg query latency: 122.3 ms
Hardware:           4GB RAM, CPU-only
```

The synthetic benchmark is primarily used for architectural regression testing, retrieval/ranking experiments, and performance analysis.

The LongMemEval adapter provides a separate evaluation path using a real-world conversational-memory benchmark.

---

# Architecture

The primary V4 boundary is:

```text
Query
  ↓
Query Processing
  ↓
Routing
  ↓
Parallel Retrieval
  ↓
Scheduler / Blackboard
  ↓
Candidate Records
  ↓
Ranking
  ↓
Finalization
  ↓
Context Construction
  ↓
MMR
  ↓
Results
```

The architectural goal is to keep **retrieval responsible for finding candidates** and **ranking responsible for determining which candidates are useful**.

Retrieval workers are independently replaceable and can be coordinated by declarative completion policies.

---

# Plugin Architecture

Memoria uses **Pluggy** to expose extension points across the system.

Current architecture:

* **10 plugin subsystems**
* **39 hook specifications**
* Plugin discovery through entry points and a local `plugins/` directory

Subsystems include:

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

Plugins can register components such as:

* retrieval workers
* ranking signals
* rerankers
* database backends
* vector stores
* ingestion extractors
* entity recognizers
* scheduler workers
* completion policies
* routers
* benchmark adapters
* analyzers
* feedback recorders
* query processors

The goal is to make major pieces of the memory substrate replaceable without requiring the core query pipeline to be rewritten.

---

# Configuration

Memoria exposes a large set of configurable parameters through Pydantic in `cache/config.py`.

Environment variables can override settings using the `MEMORY_` prefix.

Example:

```bash
export MEMORY_TOP_K=1000
export MEMORY_CONTEXT_TOKEN_BUDGET=20000
```

### Retrieval

* `TOP_K`
* `TOP_N`
* `TOP_K_PER_SHARD`
* `GRAPH_TOP_K`
* `GRAPH_SEARCH_LIMIT`
* `GRAPH_DEPTH`
* `USE_BM25`
* `USE_PHRASE_SEARCH`
* `USE_INVERTED_INDEX`
* `RETRIEVAL_MIN_CANDIDATES`
* `MIN_RETRIEVAL_SOURCES`
* `RETRIEVAL_DEADLINE`

### Routing

* `USE_ROUTING`
* `ROUTING_MATRIX_OVERRIDE`
* `ROUTING_FALLBACK_ENABLED`

### Ranking

* `RANKING_SEMANTIC`
* `RANKING_TOKEN`
* `RANKING_TFIDF`
* `RANKING_BM25`
* `RANKING_ENTITY`
* `RANKING_SUBJECT`
* `RANKING_ATTRIBUTE`
* `RANKING_IMPORTANCE`
* `RANKING_RECENCY`
* `RANKING_FEEDBACK`

### Memory

* `MEMORY_DECAY_DAYS`
* `MEMORY_DECAY_RATE`
* `CONSOLIDATE_THRESHOLD`
* `CONSOLIDATE_BATCH_SIZE`
* `PRUNE_THRESHOLD`
* `PRUNE_MAX_AGE_DAYS`
* `AUTO_STORE_MEMORIES`
* `AUTO_STORE_THRESHOLD`

### Architecture

* `USE_BLACKBOARD`
* `USE_SHARDING`
* `NUM_SHARDS`
* `MMR_ENABLED`
* `USE_ADAPTIVE_WEIGHTS`
* `RANKER_DIAGNOSTICS`
* `ENABLE_SIGNAL_REGISTRY`

---

## Ranking Weights

Default ranking weights:

| Signal   | Weight |
| -------- | -----: |
| Semantic | 0.1195 |
| Token    | 0.3107 |
| TF-IDF   | 0.2929 |
| Entity   | 0.0272 |
| Subject  | 0.0980 |
| BM25     | 0.0762 |

Additional ranking signals such as importance, recency, attribute, and feedback can also be configured.

---

## Performance Tuning

| Goal        | Action                                                |
| ----------- | ----------------------------------------------------- |
| **Speed**   | Use embedding cache / skip embedding when appropriate |
| **Recall**  | Increase `TOP_K_PER_SHARD` and `TOP_K`                |
| **Quality** | Tune retrieval and ranking configuration              |
| **Context** | Adjust `CONTEXT_TOKEN_BUDGET`                         |

Memoria is designed so retrieval, ranking, routing, scheduling, and storage behavior can be tuned independently.

---

## Chat Templates & LLM Configuration

Memoria is LLM-agnostic at the memory layer.

Chat templates can be configured through `CHAT_TEMPLATE_FILE`.

Templates use:

```text
{system}
{context}
{user}
{assistant}
```

The LLM endpoint and generation settings can also be configured through `LLM_URL`, `LLM_ENDPOINT`, `LLM_MAX_TOKENS`, `LLM_TEMPERATURE`, `LLM_TIMEOUT`, and `LLM_STOP_TOKENS`.

---

## Requirements

* Python 3.12+
* 4GB RAM minimum target
* CPU-compatible
* Local embedding model
* No cloud service required

---

## License

MIT

---

**Built solo. Local-first. LLM-agnostic. Designed to be extended rather than replaced.**
