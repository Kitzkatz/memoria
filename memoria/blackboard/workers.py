import time
from typing import Dict, Any

from cache.config import settings
from core.logger import debug


# ----------------------------------------------------------------------
# Sharding helpers
# ----------------------------------------------------------------------

def _get_shard_config(payload: Dict[str, Any]):
    """Resolve shard configuration from the task payload/settings."""
    shard_id = int(payload.get("shard_id", 0))
    num_shards = int(
        payload.get(
            "num_shards",
            getattr(settings, "NUM_SHARDS", 1),
        )
    )

    if num_shards < 1:
        num_shards = 1

    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(
            f"Invalid shard_id={shard_id} for num_shards={num_shards}"
        )

    return shard_id, num_shards


def _memory_id(value):
    """
    Normalize a memory/document ID to the integer identity used by
    Memoria's SQLite-backed retrieval indexes.

    Returns None for invalid IDs.
    """
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _belongs_to_shard(memory_id, shard_id, num_shards):
    """Return whether a memory ID belongs to this shard."""
    memory_id = _memory_id(memory_id)

    if memory_id is None:
        return False

    if num_shards <= 1:
        return True

    return memory_id % num_shards == shard_id


def _shard_filter(
    items,
    key_fn,
    shard_id,
    num_shards,
):
    """
    Filter retrieval results to the deterministic memory-ID shard.

    The identity used for sharding is always the real memory/document ID,
    never a corpus position or result-list position.
    """
    if num_shards <= 1:
        return list(items)

    filtered = []

    for item in items:
        memory_id = _memory_id(key_fn(item))

        if memory_id is None:
            continue

        if _belongs_to_shard(
            memory_id,
            shard_id,
            num_shards,
        ):
            filtered.append(item)

    return filtered


def _dedupe_candidates(candidates):
    """
    Deduplicate candidate tuples by real memory ID.

    Keeps the highest score encountered for each memory.
    """
    unique = {}

    for memory_id, score in candidates:
        memory_id = _memory_id(memory_id)

        if memory_id is None:
            continue

        score = float(score)

        previous = unique.get(memory_id)

        if previous is None or score > previous:
            unique[memory_id] = score

    return list(unique.items())


# ----------------------------------------------------------------------
# Base worker
# ----------------------------------------------------------------------

class Worker:
    """Base class for all retrieval workers."""

    def process(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        raise NotImplementedError


# ----------------------------------------------------------------------
# FAISS
# ----------------------------------------------------------------------

class FAISSWorker(Worker):
    """Worker that performs FAISS vector search with deterministic sharding."""

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def process(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        start = time.perf_counter()

        query_vec = payload.get("vector")
        top_k = payload.get(
            "top_k",
            settings.TOP_K,
        )

        shard_id, num_shards = _get_shard_config(
            payload
        )

        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = int(settings.TOP_K)

        if top_k <= 0:
            return {
                "source": "faiss",
                "candidates": [],
                "count": 0,
            }

        if query_vec is None:
            return {
                "error": "No vector provided",
                "source": "faiss",
                "candidates": [],
                "count": 0,
            }

        # ------------------------------------------------------------------
        # Global over-fetch.
        #
        # We need enough global neighbors for deterministic ID-based shard
        # filtering. This is an over-fetch strategy, not an independent
        # per-shard FAISS index.
        # ------------------------------------------------------------------

        search_k = (
            top_k * num_shards
            if num_shards > 1
            else top_k
        )

        ids, distances = self.vector_store.search(
            query_vec,
            k=search_k,
        )

        candidates = []

        for memory_id, distance in zip(
            ids,
            distances,
        ):
            memory_id = _memory_id(memory_id)

            if memory_id is None:
                continue

            candidates.append(
                (
                    memory_id,
                    float(distance),
                )
            )

        candidates = _dedupe_candidates(
            candidates
        )

        candidates = _shard_filter(
            candidates,
            lambda candidate: candidate[0],
            shard_id,
            num_shards,
        )

        # FAISS distances are already ordered by the vector store.
        candidates = candidates[:top_k]

        search_time = (
            time.perf_counter() - start
        ) * 1000

        debug(
            f"FAISSWorker "
            f"(shard {shard_id}/{num_shards}): "
            f"search={search_time:.2f}ms, "
            f"count={len(candidates)}"
        )

        return {
            "source": "faiss",
            "candidates": candidates,
            "count": len(candidates),
            "diagnostics": {
                "search_k": search_k,
                "requested_k": top_k,
                "shard_id": shard_id,
                "num_shards": num_shards,
            },
        }


# ----------------------------------------------------------------------
# BM25
# ----------------------------------------------------------------------

class BM25Worker(Worker):
    """
    Worker that performs BM25 lexical retrieval with deterministic sharding.

    BM25 owns the mapping between real memory IDs and internal corpus
    positions. This worker never treats a memory ID as a corpus position.
    """

    def __init__(
        self,
        bm25_ranker,
        inverted_index,
    ):
        self.bm25_ranker = bm25_ranker
        self.inverted_index = inverted_index

    def process(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        start = time.perf_counter()

        query_tokens = payload.get(
            "tokens",
            [],
        )

        limit = payload.get(
            "limit",
            settings.TOP_K,
        )

        shard_id, num_shards = _get_shard_config(
            payload
        )

        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = int(settings.TOP_K)

        if limit <= 0:
            return {
                "source": "bm25",
                "candidates": [],
                "count": 0,
            }

        if not query_tokens:
            return {
                "error": "No tokens provided",
                "source": "bm25",
                "candidates": [],
                "count": 0,
            }

        if self.bm25_ranker is None:
            return {
                "error": "BM25 ranker not available",
                "source": "bm25",
                "candidates": [],
                "count": 0,
            }

        if not self.inverted_index:
            return {
                "error": "Inverted index not available",
                "source": "bm25",
                "candidates": [],
                "count": 0,
            }

        # ------------------------------------------------------------------
        # Candidate generation.
        #
        # The inverted index returns real memory IDs.
        # ------------------------------------------------------------------

        candidate_ids = set()

        for token in query_tokens:
            ids = self.inverted_index.search(token)

            if ids:
                candidate_ids.update(ids)

        candidate_ids = [
            memory_id
            for memory_id in (
                _memory_id(value)
                for value in candidate_ids
            )
            if memory_id is not None
        ]

        if not candidate_ids:
            return {
                "error": "No candidates found",
                "source": "bm25",
                "candidates": [],
                "count": 0,
            }

        # ------------------------------------------------------------------
        # Score ALL candidate IDs first (before shard filtering)
        #
        # BM25 scores are computed once and carried forward with the candidate.
        # ------------------------------------------------------------------

        scores = self.bm25_ranker.score_ids(
            query_tokens,
            candidate_ids,
        )

        candidates = [
            (
                memory_id,
                float(scores.get(memory_id, 0.0)),
            )
            for memory_id in candidate_ids
        ]

        # ------------------------------------------------------------------
        # Filter by shard AFTER scoring.
        #
        # This ensures we have scores for all IDs before shard filtering,
        # not just the ones that happen to belong to this shard.
        # ------------------------------------------------------------------

        candidates = _shard_filter(
            candidates,
            lambda candidate: candidate[0],
            shard_id,
            num_shards,
        )

        candidates.sort(
            key=lambda item: (
                -item[1],
                item[0],
            )
        )

        candidates = candidates[:limit]

        build_time = (
            time.perf_counter() - start
        ) * 1000

        debug(
            f"BM25Worker "
            f"(shard {shard_id}/{num_shards}): "
            f"score={build_time:.2f}ms, "
            f"count={len(candidates)}"
        )

        return {
            "source": "bm25",
            "candidates": candidates,
            "count": len(candidates),
            "diagnostics": {
                "candidate_count": len(candidate_ids),
                "scored_count": len(candidates),
                "shard_id": shard_id,
                "num_shards": num_shards,
            },
        }


# ----------------------------------------------------------------------
# Graph
# ----------------------------------------------------------------------

class GraphWorker(Worker):
    """
    Worker that performs graph-based retrieval using the numpy graph.

    Graph scores remain source-specific. The worker returns the graph
    candidate identity; downstream ranking decides how that signal is used.
    """

    def __init__(self, numpy_graph):
        self.numpy_graph = numpy_graph

    def process(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        start = time.perf_counter()

        entities = payload.get(
            "entities",
            [],
        )

        depth = payload.get(
            "depth",
            2,
        )

        limit = payload.get(
            "limit",
            settings.GRAPH_TOP_K,
        )

        shard_id, num_shards = _get_shard_config(
            payload
        )

        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = int(settings.GRAPH_TOP_K)

        if limit <= 0:
            return {
                "source": "graph",
                "candidates": [],
                "count": 0,
            }

        if not entities or self.numpy_graph is None:
            return {
                "error": "No entities or graph",
                "source": "graph",
                "candidates": [],
                "count": 0,
            }

        search_limit = (
            limit * num_shards
            if num_shards > 1
            else limit
        )

        memory_ids = self.numpy_graph.multi_hop_search(
            entities,
            depth=depth,
            limit=search_limit,
        )

        memory_ids = [
            memory_id
            for memory_id in (
                _memory_id(value)
                for value in memory_ids
            )
            if memory_id is not None
        ]

        # Preserve graph traversal order while removing duplicates.
        seen = set()
        memory_ids = [
            memory_id
            for memory_id in memory_ids
            if not (
                memory_id in seen
                or seen.add(memory_id)
            )
        ]

        memory_ids = [
            memory_id
            for memory_id in memory_ids
            if _belongs_to_shard(
                memory_id,
                shard_id,
                num_shards,
            )
        ]

        memory_ids = memory_ids[:limit]

        candidates = [
            (
                memory_id,
                0.0,
            )
            for memory_id in memory_ids
        ]

        search_time = (
            time.perf_counter() - start
        ) * 1000

        debug(
            f"GraphWorker "
            f"(shard {shard_id}/{num_shards}): "
            f"search={search_time:.2f}ms, "
            f"count={len(candidates)}"
        )

        return {
            "source": "graph",
            "candidates": candidates,
            "count": len(candidates),
            "diagnostics": {
                "search_limit": search_limit,
                "requested_limit": limit,
                "shard_id": shard_id,
                "num_shards": num_shards,
            },
        }


# ----------------------------------------------------------------------
# Phrase
# ----------------------------------------------------------------------

class PhraseWorker(Worker):
    """Worker that performs phrase retrieval using the inverted index."""

    def __init__(self, inverted_index):
        self.inverted_index = inverted_index

    def process(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        start = time.perf_counter()

        phrases = payload.get(
            "phrases",
            [],
        )

        shard_id, num_shards = _get_shard_config(
            payload
        )

        if not phrases or not self.inverted_index:
            return {
                "error": (
                    "No phrases or inverted index missing"
                ),
                "source": "phrase",
                "candidates": [],
                "count": 0,
            }

        unique = {}

        for phrase_tokens in phrases:
            doc_ids = self.inverted_index.phrase_search(
                phrase_tokens
            )

            for doc_id in doc_ids:
                doc_id = _memory_id(doc_id)

                if doc_id is None:
                    continue

                if not _belongs_to_shard(
                    doc_id,
                    shard_id,
                    num_shards,
                ):
                    continue

                # Phrase match is currently binary.
                # Preserve the highest source score if multiple phrases
                # produce the same document.
                previous = unique.get(
                    doc_id,
                    0.0,
                )

                unique[doc_id] = max(
                    previous,
                    1.0,
                )

        candidates = list(
            unique.items()
        )

        candidates.sort(
            key=lambda item: (
                -item[1],
                item[0],
            )
        )

        # Keep existing phrase-worker ceiling.
        candidates = candidates[:100]

        phrase_time = (
            time.perf_counter() - start
        ) * 1000

        debug(
            f"PhraseWorker "
            f"(shard {shard_id}/{num_shards}): "
            f"phrase_time={phrase_time:.2f}ms, "
            f"count={len(candidates)}"
        )

        return {
            "source": "phrase",
            "candidates": candidates,
            "count": len(candidates),
            "diagnostics": {
                "shard_id": shard_id,
                "num_shards": num_shards,
            },
        }


# ----------------------------------------------------------------------
# Attribute
# ----------------------------------------------------------------------

class AttributeWorker(Worker):
    """Worker that performs attribute-based retrieval."""

    def __init__(self, db):
        self.db = db

    def process(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        start = time.perf_counter()

        subject = payload.get(
            "subject"
        )

        attribute = payload.get(
            "attribute"
        )

        shard_id, num_shards = _get_shard_config(
            payload
        )

        if not subject or not attribute:
            return {
                "error": "No subject or attribute",
                "source": "attribute",
                "candidates": [],
                "count": 0,
            }

        rows = self.db.search_attribute(
            subject,
            attribute,
        )

        candidates = []

        for row in rows:
            memory_id = _memory_id(
                row.get("id")
            )

            if memory_id is None:
                continue

            if not _belongs_to_shard(
                memory_id,
                shard_id,
                num_shards,
            ):
                continue

            candidates.append(
                (
                    memory_id,
                    0.0,
                )
            )

        attr_time = (
            time.perf_counter() - start
        ) * 1000

        debug(
            f"AttributeWorker "
            f"(shard {shard_id}/{num_shards}): "
            f"attr_time={attr_time:.2f}ms, "
            f"count={len(candidates)}"
        )

        return {
            "source": "attribute",
            "candidates": candidates,
            "count": len(candidates),
            "diagnostics": {
                "shard_id": shard_id,
                "num_shards": num_shards,
            },
        }


# ----------------------------------------------------------------------
# Fusion
# ----------------------------------------------------------------------

class FusionWorker(Worker):
    """
    Retrieval-level fusion of FAISS and BM25 using Reciprocal Rank Fusion.

    Despite the historical class name/documentation, this worker does NOT
    perform min-max score normalization or weighted-score fusion.

    It performs RRF:

        RRF(d) = sum(1 / (k + rank_i(d)))

    A document appearing in both sources therefore receives contributions
    from both rankings.

    This worker is retained as an explicit retrieval-level fusion primitive.
    The normal V4 router may instead submit FAISS/BM25 independently and
    perform fusion downstream.
    """

    def __init__(
        self,
        faiss_worker,
        bm25_worker,
        semantic_weight=0.5,
    ):
        self.faiss_worker = faiss_worker
        self.bm25_worker = bm25_worker

        # Retained for backwards compatibility with callers that provide it.
        # RRF itself does not use a semantic_weight.
        self.semantic_weight = semantic_weight

    def process(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        start = time.perf_counter()

        top_k = payload.get(
            "top_k",
            settings.TOP_K,
        )

        shard_id, num_shards = _get_shard_config(
            payload
        )

        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = int(settings.TOP_K)

        if top_k <= 0:
            return {
                "source": "fusion",
                "candidates": [],
                "count": 0,
            }

        # ------------------------------------------------------------------
        # Over-fetch both retrieval sources.
        #
        # We fuse rankings, not raw scores, so source-specific score scales
        # do not need normalization.
        # ------------------------------------------------------------------

        source_limit = top_k * 2

        faiss_result = self.faiss_worker.process({
            "vector": payload.get("vector"),
            "top_k": source_limit,
            "shard_id": shard_id,
            "num_shards": num_shards,
        })

        bm25_result = self.bm25_worker.process({
            "tokens": payload.get("tokens", []),
            "limit": source_limit,
            "shard_id": shard_id,
            "num_shards": num_shards,
        })

        faiss_candidates = (
            faiss_result.get(
                "candidates",
                [],
            )
        )

        bm25_candidates = (
            bm25_result.get(
                "candidates",
                [],
            )
        )

        # ------------------------------------------------------------------
        # Build 1-indexed source rankings.
        #
        # Candidate lists are already deduplicated by their workers, but
        # enumerate unique IDs defensively here so fusion cannot assign
        # multiple ranks to one memory.
        # ------------------------------------------------------------------

        faiss_rank = {}
        faiss_score = {}

        for memory_id, distance in faiss_candidates:
            memory_id = _memory_id(memory_id)

            if memory_id is None:
                continue

            if memory_id not in faiss_rank:
                faiss_rank[memory_id] = (
                    len(faiss_rank) + 1
                )
                faiss_score[memory_id] = float(
                    distance
                )

        bm25_rank = {}
        bm25_score = {}

        for memory_id, score in bm25_candidates:
            memory_id = _memory_id(memory_id)

            if memory_id is None:
                continue

            if memory_id not in bm25_rank:
                bm25_rank[memory_id] = (
                    len(bm25_rank) + 1
                )
                bm25_score[memory_id] = float(
                    score
                )

        # ------------------------------------------------------------------
        # Reciprocal Rank Fusion.
        # ------------------------------------------------------------------

        rrf_k = int(
            getattr(
                settings,
                "RRF_K",
                60,
            )
        )

        if rrf_k < 0:
            rrf_k = 60

        all_ids = (
            set(faiss_rank)
            | set(bm25_rank)
        )

        rrf_scores = {}

        for memory_id in all_ids:
            score = 0.0

            rank = faiss_rank.get(
                memory_id
            )

            if rank is not None:
                score += 1.0 / (
                    rank + rrf_k
                )

            rank = bm25_rank.get(
                memory_id
            )

            if rank is not None:
                score += 1.0 / (
                    rank + rrf_k
                )

            rrf_scores[memory_id] = score

        sorted_candidates = sorted(
            rrf_scores.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )

        top_candidates = sorted_candidates[
            :top_k
        ]

        fusion_time = (
            time.perf_counter() - start
        ) * 1000

        debug(
            f"FusionWorker "
            f"(RRF, k={rrf_k}): "
            f"fusion={fusion_time:.2f}ms, "
            f"count={len(top_candidates)}"
        )

        return {
            "source": "fusion",
            "candidates": top_candidates,
            "count": len(top_candidates),
            "diagnostics": {
                "rrf_k": rrf_k,
                "source_limit": source_limit,
                "faiss_count": len(faiss_rank),
                "bm25_count": len(bm25_rank),
                "overlap_count": len(
                    set(faiss_rank)
                    & set(bm25_rank)
                ),
                "shard_id": shard_id,
                "num_shards": num_shards,
                "source_scores": {
                    str(memory_id): {
                        "faiss_rank": faiss_rank.get(
                            memory_id
                        ),
                        "faiss_distance": faiss_score.get(
                            memory_id
                        ),
                        "bm25_rank": bm25_rank.get(
                            memory_id
                        ),
                        "bm25_score": bm25_score.get(
                            memory_id
                        ),
                    }
                    for memory_id in (
                        memory_id
                        for memory_id, _ in top_candidates
                    )
                },
            },
        }


##class TemporalWorker(Worker):
##    """
##    Worker that retrieves memories based on temporal constraints.
##    
##    Parses query for temporal expressions:
##    - before, after, during, between, since, until
##    - most recent, first, last, previous, next
##    - how long, how many times, in the past N days
##    """
##
##    def __init__(self, db):
##        self.db = db
##
##    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
##        start = time.perf_counter()
##        query_text = payload.get("query_text", "")
##        query_tokens = payload.get("tokens", [])
##        shard_id, num_shards = _get_shard_config(payload)
##
##        # 1. Parse temporal constraints
##        constraints = self._parse_temporal(query_text, query_tokens)
##
##        if not constraints:
##            return {"source": "temporal", "candidates": [], "count": 0}
##
##        # 2. Query DB for memories that satisfy temporal constraints
##        candidates = self.db.search_temporal(constraints)
##
##        # 3. Score candidates by how well they satisfy constraints
##        scored = self._score_temporal(candidates, constraints)
##
##        # 4. Filter by shard
##        scored = _shard_filter(scored, lambda c: c[0], shard_id, num_shards)
##        scored.sort(key=lambda x: x[1], reverse=True)
##
##        temporal_time = (time.perf_counter() - start) * 1000
##        debug(f"TemporalWorker: {temporal_time:.2f}ms, count={len(scored)}")
##
##        return {
##            "source": "temporal",
##            "candidates": scored,
##            "count": len(scored),
##            "diagnostics": {"constraints": constraints},
##        }
##
##    def _parse_temporal(self, query_text, query_tokens):
##        """Extract temporal constraints from query."""
##        constraints = {}
##
##        # Pattern matching for temporal expressions
##        if "before" in query_text:
##            constraints["before"] = self._extract_date(query_text, "before")
##        if "after" in query_text:
##            constraints["after"] = self._extract_date(query_text, "after")
##        if "between" in query_text:
##            constraints["between"] = self._extract_range(query_text)
##        if "most recent" in query_text or "last" in query_text:
##            constraints["most_recent"] = True
##        if "first" in query_text:
##            constraints["first"] = True
##        if "how long" in query_text or "how many times" in query_text:
##            constraints["aggregation"] = True
##
##        return constraints
##
##    def _score_temporal(self, candidates, constraints):
##        """Score candidates by temporal constraint satisfaction."""
##        scored = []
##        for memory_id, _ in candidates:
##            score = 0.0
##            # Apply temporal scoring logic
##            scored.append((memory_id, score))
##        return scored


class ContradictionWorker(Worker):
    """
    Worker that detects and resolves contradictory facts across turns.
    
    Identifies same entity, same attribute, different value across memories.
    Returns candidates with contradiction metadata.
    """

    def __init__(self, db, entity_store):
        self.db = db
        self.entity_store = entity_store

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        query_text = payload.get("query_text", "")
        query_tokens = payload.get("tokens", [])
        shard_id, num_shards = _get_shard_config(payload)

        # 1. Extract entities from query
        entities = self._extract_entities(query_text)

        # 2. Find memories with those entities
        candidates = self.db.search_entities(entities)

        # 3. Detect contradictions
        contradictions = self._detect_contradictions(candidates)

        # 4. Score and resolve
        scored = self._resolve_contradictions(candidates, contradictions)

        scored = _shard_filter(scored, lambda c: c[0], shard_id, num_shards)
        scored.sort(key=lambda x: x[1], reverse=True)

        return {
            "source": "contradiction",
            "candidates": scored,
            "count": len(scored),
            "diagnostics": {"contradictions": contradictions},
        }

    def _detect_contradictions(self, candidates):
        """Detect conflicting facts across candidates."""
        # Group by entity + attribute
        # Look for different values
        # Track temporal ordering
        pass
