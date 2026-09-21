# Benchmarks

Memoria's benchmark tooling is intended to measure retrieval behavior, ranking behavior, latency, and end-to-end memory-system behavior separately where possible.

The most important distinction is between **retrieval evaluation** and **full answer-generation evaluation**.

A retrieval result showing that Memoria found the expected evidence is not equivalent to a LongMemEval leaderboard score. Full LongMemEval performance also depends on the answer-generation and evaluation stages.

---

## Benchmark Scope

Memoria supports benchmark workflows that exercise the actual memory system rather than testing an isolated vector-search implementation.

Depending on the benchmark configuration, a query can exercise:

```text
Query Processing
      ↓
Routing
      ↓
Parallel Retrieval
      ↓
Candidate Construction
      ↓
Ranking
      ↓
Context / Response Construction
      ↓
Optional LLM Answer Generation
```

For retrieval-focused experiments, ranking can be disabled so that the retrieved candidates are evaluated without the downstream ranking pipeline.

This makes it possible to ask separate questions:

* Did Memoria retrieve the expected evidence?
* How did ranking reorder the retrieved candidates?
* How much latency comes from retrieval?
* How much latency comes from ranking?
* How much additional performance comes from answer generation?

---

## LongMemEval Retrieval Evaluation

Memoria includes an adapter for LongMemEval-style evaluation.

The retrieval-focused workflow treats each question/haystack independently:

```text
Load one question
      ↓
Load its associated sessions
      ↓
Ingest into Memoria
      ↓
Query Memoria
      ↓
Compare retrieved memories against expected evidence
      ↓
Record diagnostics
      ↓
Clear database
      ↓
Next question
```

This isolation is important.

It prevents memories from one LongMemEval question from contaminating retrieval for another question.

The benchmark should therefore be interpreted as a series of isolated retrieval experiments rather than as one continuously growing memory database.

---

## Retrieval-Only Results

The current LongMemEval work has primarily focused on **retrieval recall**.

The retrieval experiment asks whether Memoria returns the expected evidence/session identifiers among its retrieved results.

This is intentionally different from asking an LLM to produce the final natural-language answer.

Therefore:

> A retrieval percentage is a retrieval result, not a LongMemEval leaderboard score.

This distinction should remain explicit in benchmark reports.

---

## Retrieval Metrics

The primary retrieval metrics include:

### Recall@1

Whether the expected evidence appears at rank 1.

### Recall@5

Whether the expected evidence appears within the first five retrieved candidates.

### Recall@10

Whether the expected evidence appears within the first ten retrieved candidates.

### NDCG@10

A ranking-sensitive metric measuring how effectively relevant evidence is positioned within the top ten results.

These metrics can be reported with ranking enabled or disabled depending on the experiment.

When ranking is disabled, the experiment is closer to measuring the retrieval layer itself.

When ranking is enabled, the result reflects the combined effect of retrieval and downstream ranking.

---

## Retrieval Coverage

The benchmark also tracks whether a question successfully produced a retrievable result at all.

This is useful because a system can have a high recall among successful queries while still failing to produce usable candidates for some questions.

For that reason, benchmark reports should distinguish:

```text
Questions evaluated
Questions with retrieval results
Questions with expected evidence retrieved
Recall@K
```

These are different measurements.

---

## Current LongMemEval Retrieval Work

The current development work has been focused on improving the retrieval experiment itself before spending substantial resources on full answer generation.

The workflow has included:

* native LongMemEval-format parsing
* question/session isolation
* expected-evidence identification
* Memoria ingestion
* retrieval-only evaluation
* per-question diagnostics
* database clearing between questions
* embedding-cache construction
* batch ingestion
* retrieval benchmarking on constrained hardware

A recent retrieval-focused run reached approximately **96% retrieval coverage/recall with document embeddings disabled**.

That number should be treated as a retrieval-development result, not as a full LongMemEval score.

The current embedding-backed experiment is intended to provide a comparable measurement with the normal semantic retrieval path enabled.

---

## Embedding Cost

Document embedding has been a significant part of benchmark runtime.

An earlier implementation performed embedding work during ingestion at a scale that made a full LongMemEval run impractical.

The current batch path changes this in two important ways:

1. `store_many()` can batch document embedding through `embed_many()`.
2. `skip_embedding_build=True` can be used when the benchmark is operating against an already-loaded/cached vector index.

This allows ingestion and embedding work to be measured separately rather than forcing every benchmark run to regenerate every embedding.

The benchmark infrastructure therefore supports experiments where the embedding-generation cost is excluded from the retrieval measurement.

That should be reported explicitly whenever used.

---

## Hardware

Memoria has been tested on constrained CPU-only hardware.

One of the documented development environments is approximately:

```text
CPU: Intel Celeron N4020
CPU clock: ~1.1 GHz
RAM: ~4 GB
GPU: none
```

The purpose of reporting this environment is reproducibility and context.

Latency measured on this hardware should not be interpreted as a universal performance claim for Memoria.

Likewise, faster hardware can change ingestion and query timings substantially without changing retrieval quality.

---

## Query Latency

The V4 query handler exposes separate timing boundaries for:

* query processing
* query embedding
* retrieval
* database access
* ranking
* response construction
* feedback
* auto-store
* total query time

This allows benchmark analysis to distinguish:

```text
Total query latency
├── Query processing
├── Embedding
├── Retrieval
│   ├── Scheduler wait
│   └── Database/candidate work
├── Ranking
├── Response construction
├── Feedback
└── Auto-store
```

V4 also records scheduler/source state, including:

* submitted sources
* completed sources
* pending sources
* failed sources
* completion policy
* completion reason
* retrieval wait time

This is important when investigating tail latency or deadline-limited retrieval.

---

## Historical Results

Memoria has accumulated benchmark results across multiple architecture revisions.

These historical results are useful for studying architectural changes, but they should not automatically be presented as measurements of the current release.

Historical benchmark records can differ in:

* hardware
* corpus
* database contents
* embedding configuration
* worker configuration
* ranking configuration
* scheduler behavior
* benchmark adapter version
* query isolation
* metric definitions

Therefore historical results should be labeled with the architecture/configuration they actually represent.

A number from an earlier V3/V4 experiment should not be silently presented as a current V4.5.x benchmark.

---

## Ablation Testing

The retrieval architecture is designed to support component-level ablations.

Useful retrieval ablations include:

```text
FAISS
BM25
Graph
Phrase
Attribute
Fusion
FAISS + BM25
FAISS + BM25 + Graph
Full configured worker set
```

The purpose of these experiments is to determine which retrieval mechanisms contribute to evidence recall and how the combination changes candidate coverage.

A proper ablation should hold the benchmark corpus, question set, evaluation logic, and other relevant configuration constant while changing the component under test.

---

## Ranking Ablation

Ranking can be disabled independently of retrieval.

With:

```text
RANKING_ENABLED = False
```

the system sorts candidates using the retrieval score already available from the retrieval layer rather than running the full ranking pipeline.

This provides a useful comparison:

```text
Retrieval only
vs.
Retrieval + Ranking
```

The difference should not automatically be interpreted as a pure ranking improvement unless the rest of the experiment is controlled.

---

## Scheduler Ablation

The scheduler itself can also be studied through retrieval configuration.

Potential comparisons include:

* different worker sets
* different completion policies
* different retrieval deadlines
* blackboard path vs V3 fallback
* single-source retrieval vs multi-source retrieval

The important diagnostic fields are the source-level completion fields:

```text
retrieval_submitted_sources
retrieval_completed_sources
retrieval_pending_sources
retrieval_failed_sources
```

These make it possible to determine whether a retrieval result was produced by the intended worker set or whether some sources failed to complete.

---

## Benchmark Isolation

LongMemEval-style experiments should isolate questions whenever the benchmark is intended to measure retrieval against a known haystack.

The preferred conceptual model is:

```text
Question A
  └── Haystack A
       └── Query A
            └── Results A

Clear

Question B
  └── Haystack B
       └── Query B
            └── Results B
```

This prevents Question A's memories from becoming candidates for Question B.

It also makes the expected-evidence comparison much easier to interpret.

---

## Embedding Cache Experiments

Embedding-cache work is useful for separating:

* document ingestion
* embedding generation
* vector persistence
* query embedding
* retrieval

The cache can also avoid unnecessarily recomputing document vectors when the benchmark is operating against an already-built vector store.

When an embedding cache or prebuilt FAISS index is used, benchmark reports should state that fact.

---

## Full Answer Evaluation

Retrieval-only evaluation is only the first stage of a full memory benchmark.

A complete evaluation can be decomposed into:

```text
A. Evidence Retrieval
       ↓
B. Candidate Ranking
       ↓
C. Context Construction
       ↓
D. Answer Generation
       ↓
E. Answer Evaluation
```

This decomposition is useful because it prevents a weak answer-generation result from being incorrectly attributed to retrieval, or vice versa.

For LongMemEval specifically, a high retrieval recall means that Memoria is finding the expected evidence frequently. It does **not** establish that a downstream LLM will always produce the correct answer from that evidence.

Likewise, a poor final answer does not necessarily imply that the evidence was absent.

---

## Recommended Reporting Format

For public benchmark results, a useful report should include:

| Field                   | Description                                |
| ----------------------- | ------------------------------------------ |
| Benchmark               | Dataset/evaluation suite                   |
| Evaluation mode         | Retrieval-only or full answer evaluation   |
| Questions               | Number actually evaluated                  |
| Retrieval configuration | Workers and routing configuration          |
| Ranking                 | Enabled/disabled                           |
| Embeddings              | Enabled, cached, or skipped                |
| Hardware                | CPU/RAM/GPU                                |
| Recall@1                | Retrieval recall at rank 1                 |
| Recall@5                | Retrieval recall at rank 5                 |
| Recall@10               | Retrieval recall at rank 10                |
| NDCG@10                 | Ranking-sensitive retrieval metric         |
| Avg query time          | Mean query latency                         |
| Retrieval time          | Retrieval-stage latency                    |
| Ranking time            | Ranking-stage latency                      |
| Failures                | Questions that failed evaluation           |
| Notes                   | Relevant configuration/version information |

This format makes benchmark results substantially easier to reproduce and compare.

---

## What Current Results Do and Do Not Establish

Current retrieval experiments establish useful evidence about Memoria's retrieval behavior under the tested configuration.

They do **not**, by themselves, establish:

* a leaderboard position
* superiority over another memory system
* a universal latency figure
* full LongMemEval answer accuracy
* performance on hardware other than the tested environment
* performance under configurations not included in the experiment

Those claims require controlled comparisons and the corresponding evaluation stages.

---

## Benchmark Philosophy

Memoria's benchmark work is intended to answer engineering questions, not merely produce a single headline number.

The useful questions are:

* What evidence can the retrieval system recover?
* Which retrieval sources contribute?
* Does ranking improve evidence ordering?
* What does the scheduler actually wait for?
* Which sources fail or remain pending?
* How much does embedding generation cost?
* How much latency comes from retrieval versus ranking?
* Does the architecture remain useful under constrained hardware?
* Which architectural changes improve measurable behavior?

The benchmark system should make those questions observable rather than collapsing everything into one score.
