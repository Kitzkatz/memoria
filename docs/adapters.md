---
title: Adapters
description: How Memoria adapters wrap external datasets for evaluation, and how to write your own.
---

# Adapters

Memoria ships with adapters for external benchmarks and datasets. An adapter is a thin translation layer that:

1. Reads the dataset's native structure.
2. Constructs an isolated memory state for each question or conversation.
3. Runs Memoria queries.
4. Compares retrieved results against the dataset's expected IDs.
5. Emits evaluation metrics in a standard format.

The adapter does **not** implement retrieval or ranking logic. It delegates to Memoria.

---

## Shipped Adapters

| Adapter | Dataset | File |
|---------|---------|------|
| LongMemEval-S | `longmemeval_s_cleaned.json` | `benchmark/longmemeval_adapter.py` |
| LoCoMo-S | `locomo10.json` | `benchmark/locomoeval_adapter.py` |

---

## Adapter Responsibilities

Every adapter should:

- **Isolate** the retrieval state per question or conversation (either by clearing the DB or by loading an isolated cache).
- **Preserve** the dataset's native identifiers (`dia_id`, `session_id`, `turn_id`, `question_id`) in memory metadata.
- **Cache** embeddings and indexes per question or conversation to make repeated evaluation practical on constrained hardware.
- **Record** the following per question:
    - retrieved candidate IDs
    - expected IDs
    - session-level and turn-level metrics
    - retrieval diagnostics
    - per-stage timing
- **Exclude** abstained or non-evaluable records from official metric aggregation.

---

## Example: LongMemEval Adapter

The LongMemEval adapter:

- Works from the benchmark's native question/haystack structure rather than converting the dataset into Memoria's original database format.
- Isolates the haystack per question.
- Caches embeddings.
- Reuses cached embeddings across runs.
- Identifies expected answer sessions and expected answer turns separately.
- Computes session-level and turn-level retrieval metrics.

On a **4GB RAM CPU-only laptop**, caching embeddings for the 500-question evaluation takes a few hours on 4 GB; subsequent evaluation runs against the cached embeddings take roughly 3–4 minutes.

Run it:

```bash
python benchmark/longmemeval_adapter.py \
    --dataset longmemeval_s_cleaned.json \
    --cache-dir cache/faiss_indices \
    --workers 4 \
    --output benchmark_output/results/longmemeval.json
```

---

## Example: LoCoMo Adapter

The LoCoMo adapter:

- Isolates per conversation.
- Stores every conversation turn.
- Preserves `dia_id`, `session_idx`, `turn_idx`, and `session_key` in memory metadata.
- Routes temporal queries to the standalone temporal worker.
- Falls back to fusion when temporal returns no candidates.
- Excludes adversarial (Category 5) questions from scoring.

Run it:

```bash
python benchmark/locomoeval_adapter.py \
    --dataset locomo10.json \
    --cache-dir cache/faiss_indices \
    --workers 4 \
    --output benchmark_output/results/locomo.json
```

---

## Writing Your Own Adapter

An adapter typically:

1. **Loads the dataset** into a list of entries (questions, conversations, or sessions).
2. **Builds memory metadata** from each entry's native structure.
3. **Stores memories** using `BatchLoader` or `MemoryInterface`.
4. **Runs queries** against the controller.
5. **Compares results** against the dataset's expected identifiers.
6. **Emits metrics** in the same shape the bundled adapters emit.

There is no enforced base class. The adapters are intentionally plain Python modules so you can wire them to any dataset schema.

---

## Best Practices

- **Cache aggressively.** The dominant cost is embedding. Cache the DB, the FAISS index, and BM25 state per question or conversation.
- **Preserve metadata.** The `dia_id`, `session_id`, and `turn_id` fields are used by the temporal worker and the evaluation layer.
- **Separate retrieval metrics from answer-generation metrics.** Retrieval-only results should not be presented as end-to-end answer accuracy.
- **Exclude abstention records.** Mark them, skip them, and report them separately.
