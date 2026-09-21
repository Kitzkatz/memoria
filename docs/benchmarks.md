---
title: Benchmarks
description: Retrieval-only benchmark methodology and results for LongMemEval-S, the synthetic suite, and LoCoMo.
---

# Benchmarks

This page documents Memoria's benchmark methodology, hardware, and results.

**Important:** Memoria reports **retrieval performance only**. These metrics measure whether expected sessions or turns appear within the retrieved candidate set and how they are ranked. They do **not** measure:

- end-to-end answer-generation accuracy
- LLM answer quality
- factual correctness of a generated response
- conversational quality

---

## Hardware

All primary results were produced on a **3.7GB RAM CPU-only laptop** (Intel Celeron N4020 CPU @ 1.1Ghz x 2). The system is target-designed to run under these constraints.

An independent desktop run is included as a separate measurement.

---

## Metrics

**Recall Any** indicates whether at least one expected item was retrieved within K.

**Recall All** indicates whether all expected items were retrieved within K.

**NDCG** measures the ranking of expected items within the retrieved results.

Session-level metrics evaluate retrieval of the expected answer session. Turn-level metrics evaluate retrieval of the specific expected answer turn.

---

## LongMemEval-S

### Evaluation Scope

- Dataset: `longmemeval_s_cleaned.json`
- Total questions: 500
- Evaluable questions (excluding abstained/non-evaluable): 470

### Current Retrieval Metrics

Fusion configuration, 470 evaluable questions:

| K | Session Recall Any | Session Recall All | Session NDCG | Turn Recall Any | Turn Recall All | Turn NDCG |
|--:|-------------------:|-------------------:|-------------:|----------------:|----------------:|----------:|
| 1 | 89.6% | 31.7% | 0.8957 | 26.4% | 7.4% | 0.2638 |
| 3 | 96.2% | 80.4% | 0.8993 | 48.7% | 19.4% | 0.3021 |
| 5 | 97.7% | 86.4% | 0.9095 | 60.6% | 30.9% | 0.3551 |
| 10 | 98.7% | 93.2% | 0.9236 | 75.3% | 47.4% | 0.4156 |
| 30 | 99.4% | 98.5% | 0.9319 | 87.2% | 67.9% | 0.4664 |
| 50 | 99.4% | 98.5% | 0.9319 | 91.1% | 77.9% | 0.4824 |

### Retrieval Ablation

Same evaluation dataset and metric definitions:

| Configuration | R@1 | R@3 | R@5 | R@10 | NDCG@10 |
|---------------|----:|----:|----:|-----:|--------:|
| Dense | 87.0% | 94.0% | 97.2% | 98.5% | 0.9083 |
| BM25 | 81.1% | 91.5% | 93.6% | 96.8% | 0.8540 |
| Raw | 62.8% | 72.8% | 74.7% | 77.2% | 0.5720 |
| **Fusion** | **89.6%** | **96.2%** | **97.7%** | **98.7%** | **0.9236** |
| Full (all workers) | 34.0% | 49.6% | 55.7% | 63.6% | 0.3649 |

> **Note on "Full":** the Full configuration submits every retrieval worker simultaneously — FAISS, BM25, graph, phrase, attribute, and temporal all at once. Flooding the candidate pool with weakly-related hits from every source dilutes the ranking signal and pushes the correct session down the list. It is retained as a **negative control**: it shows that adding more retrieval sources without coordination is worse than selecting the right ones. Coordination through the scheduler is the point.

### Independent Desktop Run

| Configuration | R@1 | R@3 | R@5 | R@10 | NDCG@10 |
|---------------|----:|----:|----:|-----:|--------:|
| Dense | 87.0% | 94.0% | 97.2% | 98.5% | 0.9083 |
| BM25 | 81.1% | 91.5% | 93.6% | 96.8% | 0.8540 |
| Raw | 62.8% | 72.8% | 74.7% | 77.2% | 0.5720 |
| **Fusion** | **89.8%** | **96.4%** | **97.9%** | **98.9%** | **0.9257** |
| Full (all workers) | 34.0% | 49.6% | 55.7% | 63.6% | 0.3649 |

The second run is reported separately and is not combined with the primary run.

### Cost

Embedding caching for the 500-question evaluation took a few hours on 4 GB RAM. Subsequent evaluation runs against the cached embeddings take roughly **3–4 minutes**.

---

## Synthetic Benchmark

Memoria includes a larger synthetic benchmark for evaluating retrieval and ranking behavior across controlled workloads.

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

**Note:** These are legacy results from before the retrieval stabilization work. They are kept for architectural regression testing and are not representative of the current LongMemEval-level retrieval quality.

---

## LoCoMo

LoCoMo evaluation is ongoing. The adapter is under active development and targets the same retrieval-only reporting standard as the LongMemEval adapter.

Key design decisions:

- Adversarial (Category 5) questions are excluded from scoring.
- Temporal queries route to the standalone temporal worker.
- Fusion fallback is allowed when temporal returns no candidates.

LoCoMo results will be published here when the adapter and temporal worker reach parity with the LongMemEval methodology.

---

## Reproducibility

Every benchmark result on this page is produced by an adapter in `benchmark/`. Each adapter:

- is deterministic given the same cache
- preserves the dataset's native identifiers
- emits per-question records with diagnostics and timing
- can be re-run against a warm cache in minutes

To reproduce:

```bash
python benchmark/longmemeval_adapter.py \
    --dataset longmemeval_s_cleaned.json \
    --cache-dir cache/faiss_indices \
    --workers 4 \
    --output benchmark_output/results/longmemeval.json

python benchmark/benchmark_analyzer.py \
    benchmark_output/results/longmemeval.json
```

---

## What These Numbers Are Not

Memoria's published metrics are **retrieval-only**. They should not be compared against end-to-end answer-generation scores from other systems without stating that distinction.

When referencing Memoria's LongMemEval results, use language like:

> "Memoria achieves **89.6% R@1 session-level retrieval** on the 470 evaluable questions of LongMemEval-S."

Not:

> "Memoria scores 89.6% on LongMemEval."

The distinction matters, and it's how Memoria stays honest about what it actually measures.
