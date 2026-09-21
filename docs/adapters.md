---
title: Adapters
description: How Memoria adapters wrap external datasets for retrieval evaluation, and how to write your own.
---

# Adapters

An adapter is a thin translation layer between an external dataset and Memoria. It:

1. Reads the dataset's native structure.
2. Isolates a memory state per question or conversation.
3. Stores the dataset's turns as raw memories with native identifiers attached as metadata.
4. Runs Memoria queries against that isolated state.
5. Evaluates retrieved candidates against the dataset's expected identifiers.
6. Emits a per-question record set plus a reproducibility manifest.

The adapter does **not** implement retrieval or ranking. It delegates to the same `MemoryInterface` the CLI, TUI, and API use.

---

## Storage Model

Adapters store the **raw turn text as the unit of record**. Metadata — `session_id`, `turn_id`, `turn_index`, `role`, gold-turn flags — is attached as annotations alongside the text. Extraction still produces entities, relationships, and embeddings, but those are derived indexes. The raw memory is canonical.

Storage goes through `BatchLoader.insert_batch()`:

```python
count = loader.insert_batch(
    texts,
    metadatas=metadatas,
    batch_size=args.batch_size,
    skip_embedding=args.skip_embedding,
    parallel_extract=not args.no_parallel,
    max_workers=args.workers,
    skip_embedding_build=skip_embedding_build,
)
```

`texts` and `metadatas` are positional — metadata at index *i* belongs to the memory stored at index *i*.

---

## Shipped Adapters

| Adapter | Dataset | File | Status |
|---------|---------|------|--------|
| LongMemEval-S | `longmemeval_s_cleaned.json` | `benchmark/longmemeval_adapter.py` | Stable |
| LoCoMo-S | `locomo10.json` | `benchmark/locomoeval_adapter.py` | In development |

The LongMemEval adapter is documented in full below. The LoCoMo adapter is under rewrite and is not yet at parity with the LongMemEval methodology.

---

## LongMemEval Adapter

### Evaluation flow

```
longmemeval_s_cleaned.json
        ↓
extract_questions.py   →  benchmark_output/longmemeval_questions.json
        ↓
longmemeval_adapter.py
        ↓
benchmark_output/results/longmemeval_{timestamp}.json
                       longmemeval_{timestamp}.manifest.json
        ↓
benchmark_analyzer.py  →  printed report / exported summary
```

### Question extraction

The adapter reads expected-answer text from an extracted companion file, produced by a separate script:

```bash
python benchmark/extract_questions.py \
    --input longmemeval_s_cleaned.json \
    --output benchmark_output/longmemeval_questions.json
```

The extractor writes one entry per question:

| Field | Source |
|-------|--------|
| `question_id` | `question_id` from the dataset, or `q_{idx}` fallback |
| `query` | `question` |
| `expected` | First turn content of the first answer session |
| `expected_ids` | `answer_session_ids` |

If this file is missing at run time, the adapter prints a warning and falls back to session-only gold — turn-level matching degrades to "first turn of any answer session."

### Retrieval modes

The adapter drives ablations through `configure_retrieval_mode()`. Each mode sets `WORKERS_TO_USE` and `RANKING_ENABLED`:

| Mode | Workers | Ranking |
|------|---------|---------|
| `dense` | `["faiss"]` | off |
| `bm25` | `["bm25"]` | off |
| `raw` | `["faiss", "bm25"]` | off |
| `fusion` | `["fusion"]` | off |
| `full` | `["faiss", "bm25", "graph", "phrase", "attribute"]` | on |

`fusion` submits a single worker that runs FAISS + BM25 internally and applies Reciprocal Rank Fusion. `raw` submits FAISS and BM25 as independent sources without fusing their outputs through the fusion worker. They are not the same configuration.

`full` enables every worker **and** the full ranking pipeline, including MMR, the signal registry, and the finalizer. It exists as a comparison point, not as a recommended configuration.

Select with `--retrieval-mode`:

```bash
python benchmark/longmemeval_adapter.py --retrieval-mode fusion
```

### Caches

Per-question caches make repeated evaluation practical on a 4GB CPU-only machine. Four artifacts live under `--cache-dir` for each question:

| File | Contents |
|------|----------|
| `db_{q_id}.sqlite` | Isolated SQLite database for that question's haystack |
| `faiss_{q_id}.index` | FAISS index over that haystack's embeddings |
| `bm25_{q_id}.pkl` | BM25 state: corpus tokens, doc IDs, IDF, avg doc length, k1, b |
| `{q_id}.manifest.json` | Validation record |

The manifest is checked against the current dataset checksum and embedding signature before any cached artifact is loaded:

```python
def get_cache_signature():
    return {
        "embedding_model": str(settings.EMBEDDING_MODEL),
        "embedding_dimension": int(settings.VECTOR_DIM),
    }
```

If the dataset hash, embedding model, or embedding dimension has changed since the cache was written, the cache is treated as invalid and the question is rebuilt. `--rebuild-cache` bypasses the cache entirely.

### Gold sessions vs. gold turns

Session-level gold comes directly from `answer_session_ids`. Turn-level gold is stricter:

> A turn is gold only if its content contains the expected answer text **and** it belongs to an answer session.

If the extracted questions file is unavailable, the adapter marks the **first turn** of each answer session as gold and leaves the rest unmarked. It does **not** treat every turn in an answer session as a gold turn.

### Abstention

Questions are excluded from retrieval metrics when any of these hold:

- `question["_abs"]` is truthy
- The expected text contains `"don't know"` (case-insensitive)
- Any value in `answer_session_ids` contains `_abs`

Excluded questions are recorded with `"abstention": true` and `"metrics": null`. They are counted separately and do not enter the evaluable denominator.

### Metrics

The adapter computes six metric families per question across `K = [1, 3, 5, 10, 30, 50]`:

| Level | Metric | Meaning |
|-------|--------|---------|
| Session | `recall_any` | At least one gold session appears in top-K |
| Session | `recall_all` | Every gold session appears in top-K |
| Session | `ndcg_any` | Ranking quality of gold sessions in top-K |
| Turn | `recall_any` | At least one gold turn appears in top-K |
| Turn | `recall_all` | Every gold turn appears in top-K |
| Turn | `ndcg_any` | Ranking quality of gold turns in top-K |

Session ranking is de-duplicated by first occurrence — a session that produces multiple retrieved candidates counts once, at its earliest position. Turn ranking preserves individual retrieved turns, because each turn is a distinct candidate.

NDCG uses standard discounted cumulative gain with binary relevance (gold / not gold).

### Running the adapter

```bash
python benchmark/longmemeval_adapter.py \
    --dataset longmemeval_s_cleaned.json \
    --retrieval-mode fusion \
    --cache-dir cache/faiss_indices \
    --workers 4
```

| Flag | Default | Purpose |
|------|---------|---------|
| `--dataset` | `longmemeval_s_cleaned.json` | Dataset file |
| `--limit` | none | Evaluate only the first N questions |
| `--batch-size` | `100` | Batch size passed to `BatchLoader` |
| `--workers` | `4` | Parallel extraction workers |
| `--skip-embedding` | off | Skip embedding computation |
| `--no-parallel` | off | Disable parallel extraction |
| `--optimize` | off | Run adaptive weight adjustment after evaluation |
| `--dry-run-weights` | off | With `--optimize`, report proposed changes without applying |
| `--step-size` | `0.02` | Weight adjustment step size |
| `--cache-dir` | `cache/faiss_indices` | Cache location |
| `--rebuild-cache` | off | Ignore existing caches and rebuild |
| `--output` | auto-timestamped | Output JSON path |
| `--retrieval-mode` | `fusion` | `dense`, `bm25`, `raw`, `fusion`, or `full` |

### Output

Two files are written.

**Results** — `benchmark_output/results/longmemeval_{timestamp}.json` by default, or the `--output` path. Structure:

```json
{
  "benchmark": "LongMemEval-S",
  "question_count": 500,
  "retrieval_evaluable": 470,
  "abstentions_excluded": 30,
  "records": [ /* one per question */ ],
  "dataset_checksum": "...",
  "embedder": "...",
  "embedding_dimension": 384,
  "commit": "...",
  "retrieval_mode": "fusion",
  "settings": { /* snapshot */ },
  "command": "python benchmark/longmemeval_adapter.py ...",
  "elapsed_seconds": 123.4
}
```

Each record has: `query`, `expected`, `expected_ids`, `expected_rank`, `retrieved`, `candidates`, `runtime_ms`, `diagnostics`, `metrics`, `abstention`. Abstention records have `metrics: null`.

**Manifest** — `{results_path}.manifest.json`. A fuller reproducibility record: dataset path and SHA-256, parsed arguments, git commit, vector store class name, retrieval block (mode, submitted sources, candidate limit), ranking block (enabled, MMR), cache block (directory, rebuild flag, signature), evaluation block (K values, abstention count, evaluable count, metric names), and the full settings snapshot.

### Cost

On a 4GB CPU-only laptop, first-run embedding for the 500-question evaluation takes a few hours. Subsequent runs against a warm cache complete in minutes. The manifest validation described above is what makes the second run safe — a stale cache produces a rebuild, never a silent mismatch.

---

## LoCoMo Adapter

`benchmark/locomoeval_adapter.py` is under rewrite. It currently does per-conversation isolation — one conversation stored once, all its QA pairs queried against that state — which matches the LoCoMo dataset shape better than per-question isolation would.

The rewrite is targeting parity with the LongMemEval adapter on:

- Cache manifest validation on dataset hash and embedding signature
- Gold-turn identification from expected-answer text
- Abstention handling
- Reproducibility manifest in the output
- The same session-level and turn-level metric shape

Until then, treat its output as preliminary. It will be documented in full here once it reaches parity.

---

## The Analysis Pipeline

Three modules sit between the raw dataset and the numbers in [Benchmarks](benchmarks.md):

| Module | Role |
|--------|------|
| `benchmark/batch_loader.py` | Batch storage, parallel extraction, batch embedding |
| `benchmark/result_formatter.py` | Standard record/output shape used by every adapter |
| `benchmark/benchmark_analyzer.py` | Reads result JSON, aggregates metrics, prints reports |

### Result formatter

`build_record()` normalizes a single question's data into the standard shape. `build_output()` assembles the full result file. `format_candidate()` normalizes each retrieved candidate with `rank`, `id`, `text`, `metadata`, `score`, `final_score`, and `diagnostics`.

Optional reproducibility fields on `build_output()` — `dataset_hash`, `embedder`, `retrieval_mode`, `commit`, `settings_snapshot` — are included when the adapter supplies them.

### Analyzer

```bash
python benchmark/benchmark_analyzer.py benchmark_output/results/longmemeval_*.json
```

The analyzer reads result files only. It does not touch the memory system, run queries, or load models.

Sections printed:

- Overall retrieved / top-1 / top-3 rates
- Final score distribution
- Rank distribution and rank buckets
- Average candidate count
- Timing breakdown with accounted vs. unaccounted milliseconds
- Retrieval policy and finish-reason counts
- Submitted, completed, pending, and failed source counts
- Scheduler wait times
- MMR reordering statistics
- Recall@K
- Official session-level and turn-level metrics (from the adapter's `metrics` field)

### Comparing ablations

```bash
python benchmark/benchmark_analyzer.py --compare \
    benchmark_output/results/dense_*.json \
    benchmark_output/results/bm25_*.json \
    benchmark_output/results/raw_*.json \
    benchmark_output/results/fusion_*.json \
    benchmark_output/results/full_*.json
```

Prints a table of session-level R@1, R@3, R@5, R@10, and NDCG@10 across each file.

### Explaining failures

```bash
python benchmark/benchmark_analyzer.py results.json \
    --questions benchmark_output/longmemeval_questions.json \
    --explain 20
```

Prints per-question detail for questions that were never retrieved or were retrieved but outranked. For near misses, it prints a signal-by-signal comparison between the winner and the expected candidate (`semantic`, `importance`, `recency`, `token`, `feedback`, plus `attribute_boost`, `score`, `final_score`, `mmr_score`), and an aggregate "average signal advantage" across all near misses.

### Analyzing a directory

```bash
python benchmark/benchmark_analyzer.py --all
```

Runs the analyzer over every JSON in `benchmark_output/results/` and prints a JSON summary.

---

## Writing Your Own Adapter

The shipped adapters are plain Python modules with no enforced base class. The pattern:

```python
from shared.memory_interface import MemoryInterface
from benchmark.batch_loader import BatchLoader
from benchmark.result_formatter import build_record, build_output, write_output

def build_records(dataset_item):
    """Return (texts, metadatas) for one question or conversation."""
    texts = []
    metadatas = []
    # ... populate from dataset_item ...
    return texts, metadatas

def evaluate(ranked_results, gold_session_ids, gold_turn_ids):
    """Return a metrics dict in the standard shape."""
    # ... session-level and turn-level metrics ...
    return metrics

def main():
    memory = MemoryInterface()
    loader = BatchLoader(memory)

    for item in dataset:
        texts, metadatas = build_records(item)
        loader.insert_batch(texts, metadatas=metadatas)
        response = memory.controller.recall(item["question"])
        metrics = evaluate(
            response["results"],
            item["gold_session_ids"],
            item["gold_turn_ids"],
        )
        record = build_record(
            query=item["question"],
            expected=item.get("expected", ""),
            expected_ids=item["gold_session_ids"],
            candidates=response["results"],
            metrics=metrics,
        )
        # ... append record, isolate state for next item ...
```

### Isolation

Each dataset item gets a fresh memory state. Two strategies:

- **Truncate-and-reuse** — delete all rows from every table, reset the vector store, clear the embedding cache. Fast, but requires a working DB connection.
- **Cache-and-restore** — snapshot the SQLite file and FAISS index after each item and reload for the next. Slower per item, but makes re-runs trivial.

The LongMemEval adapter uses both: cache-and-restore when the cache validates, truncate-and-reuse when it doesn't.

### Reproducibility

Every adapter should record at minimum:

- SHA-256 of the dataset file
- The embedding signature (`EMBEDDING_MODEL`, `VECTOR_DIM`)
- Git commit at run time
- Full command line and parsed arguments

If the cache manifest validates only the dataset hash but not the embedding identity, you will silently mix vectors from two different models. Both hashes belong in the manifest.

---

## Best Practices

- **Store raw text as the memory.** Attach identifiers as metadata. Do not pre-decompose turns into atomic facts — the raw turn is the unit that retrieval returns, and the metric shape depends on that.
- **Preserve native identifiers in metadata.** `session_id` and `turn_id` are what evaluation reads back out. If they don't survive storage, session-level and turn-level recall are unrecoverable.
- **Validate caches on two axes.** Dataset hash and embedding signature. Either one alone is insufficient.
- **Exclude abstentions explicitly.** Record them with `metrics: null` and count them separately. Do not silently drop them — the evaluable denominator needs to be derivable from the output.
- **Report retrieval metrics as retrieval metrics.** Session-level and turn-level recall are not answer-generation accuracy.
- **Write the manifest alongside the results.** The results file is for analysis; the manifest is for reproduction. They serve different readers.
