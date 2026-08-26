import time
from typing import Dict, Any, List, Optional
from cache.config import settings

from core.logger import debug


def _shard_filter(items, key_fn, shard_id, num_shards):
    """Filter a list of items to only those belonging to this shard."""
    if num_shards <= 1:
        return items
    return [item for item in items if key_fn(item) % num_shards == shard_id]


class Worker:
    """Base class for all workers."""
    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class FAISSWorker(Worker):
    """Worker that performs FAISS vector search with shard support."""
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        query_vec = payload.get("vector")
        top_k = payload.get("top_k", settings.TOP_K)
        shard_id = payload.get("shard_id", 0)
        num_shards = payload.get("num_shards", getattr(settings, "NUM_SHARDS", 1))

        if query_vec is None:
            return {"error": "No vector provided", "source": "faiss", "candidates": []}

        ids, distances = self.vector_store.search(query_vec, k=top_k * num_shards)
        candidates = list(zip(ids, distances))

        candidates = _shard_filter(candidates, lambda c: c[0], shard_id, num_shards)
        candidates = candidates[:top_k]

        search_time = (time.perf_counter() - start) * 1000
        debug(f"FAISSWorker (shard {shard_id}): search={search_time:.2f}ms, count={len(candidates)}")
        return {
            "source": "faiss",
            "candidates": candidates,
            "count": len(candidates)
        }


class BM25Worker(Worker):
    """Worker that performs BM25 lexical search with shard support."""
    def __init__(self, bm25_ranker, inverted_index):
        self.bm25_ranker = bm25_ranker
        self.inverted_index = inverted_index

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        query_tokens = payload.get("tokens", [])
        limit = payload.get("limit", settings.TOP_K)
        shard_id = payload.get("shard_id", 0)
        num_shards = payload.get("num_shards", getattr(settings, "NUM_SHARDS", 1))

        if not isinstance(limit, int):
            limit = int(limit) if limit else settings.TOP_K

        if not query_tokens:
            return {"error": "No tokens provided", "source": "bm25", "candidates": []}

        if not self.inverted_index:
            return {"error": "Inverted index not available", "source": "bm25", "candidates": []}

        candidate_ids = set()
        for token in query_tokens:
            candidate_ids.update(self.inverted_index.search(token))
        candidate_ids = [int(x) for x in candidate_ids if x is not None]

        if not candidate_ids:
            return {"error": "No candidates found", "source": "bm25", "candidates": []}

        # Filter by shard BEFORE scoring — don't spend time scoring
        # candidates this shard is going to throw away anyway.
        candidate_ids = _shard_filter(candidate_ids, lambda cid: cid, shard_id, num_shards)

        candidates = [
            (doc_id, self.bm25_ranker.score(query_tokens, doc_id))
            for doc_id in candidate_ids
        ]

        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:limit]

        build_time = (time.perf_counter() - start) * 1000
        debug(f"BM25Worker (shard {shard_id}): build={build_time:.2f}ms, count={len(candidates)}")
        return {
            "source": "bm25",
            "candidates": candidates,
            "count": len(candidates)
        }


class GraphWorker(Worker):
    """Worker that performs graph-based retrieval using Numpy graph with shard support."""
    def __init__(self, numpy_graph):
        self.numpy_graph = numpy_graph

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        entities = payload.get("entities", [])
        depth = payload.get("depth", 2)
        limit = payload.get("limit", settings.GRAPH_TOP_K)
        shard_id = payload.get("shard_id", 0)
        num_shards = payload.get("num_shards", getattr(settings, "NUM_SHARDS", 1))

        if not entities or not self.numpy_graph:
            return {"error": "No entities or graph", "source": "graph", "candidates": []}

        memory_ids = self.numpy_graph.multi_hop_search(entities, depth=depth, limit=limit * num_shards)
        memory_ids = [int(x) for x in memory_ids if x is not None]

        memory_ids = _shard_filter(memory_ids, lambda mid: mid, shard_id, num_shards)
        memory_ids = memory_ids[:limit]

        candidates = [(mem_id, 0.0) for mem_id in memory_ids]
        search_time = (time.perf_counter() - start) * 1000

        debug(f"GraphWorker (shard {shard_id}): search={search_time:.2f}ms, count={len(candidates)}")
        return {
            "source": "graph",
            "candidates": candidates,
            "count": len(candidates)
        }


class PhraseWorker(Worker):
    """Worker that performs phrase search using the inverted index with shard support."""
    def __init__(self, inverted_index):
        self.inverted_index = inverted_index

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        phrases = payload.get("phrases", [])
        shard_id = payload.get("shard_id", 0)
        num_shards = payload.get("num_shards", getattr(settings, "NUM_SHARDS", 1))

        if not phrases or not self.inverted_index:
            return {"error": "No phrases or inverted index missing", "source": "phrase", "candidates": []}

        results = []
        for phrase_tokens in phrases:
            doc_ids = self.inverted_index.phrase_search(phrase_tokens)
            for doc_id in doc_ids:
                results.append((doc_id, 1.0))

        unique = {}
        for doc_id, score in results:
            if doc_id not in unique or score > unique[doc_id]:
                unique[doc_id] = score

        candidates = list(unique.items())
        candidates.sort(key=lambda x: x[1], reverse=True)

        candidates = _shard_filter(candidates, lambda c: c[0], shard_id, num_shards)
        candidates = candidates[:100]

        phrase_time = (time.perf_counter() - start) * 1000
        debug(f"PhraseWorker (shard {shard_id}): phrase_time={phrase_time:.2f}ms, count={len(candidates)}")
        return {
            "source": "phrase",
            "candidates": candidates,
            "count": len(candidates)
        }


class AttributeWorker(Worker):
    """Worker that performs attribute-based retrieval with shard support."""
    def __init__(self, db):
        self.db = db

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        subject = payload.get("subject")
        attribute = payload.get("attribute")
        shard_id = payload.get("shard_id", 0)
        num_shards = payload.get("num_shards", getattr(settings, "NUM_SHARDS", 1))

        if not subject or not attribute:
            return {"error": "No subject or attribute", "source": "attribute", "candidates": []}

        rows = self.db.search_attribute(subject, attribute)
        candidates = [(row["id"], 0.0) for row in rows if row["id"] is not None]

        candidates = _shard_filter(candidates, lambda c: c[0], shard_id, num_shards)

        attr_time = (time.perf_counter() - start) * 1000
        debug(f"AttributeWorker (shard {shard_id}): attr_time={attr_time:.2f}ms, count={len(candidates)}")
        return {
            "source": "attribute",
            "candidates": candidates,
            "count": len(candidates)
        }




class FusionWorker(Worker):
    """
    Fuses FAISS (semantic) and BM25 (lexical) results using min‑max normalization
    and a weighted sum. Scores are normalized to [0,1] before fusion.
    """
    def __init__(self, faiss_worker, bm25_worker, semantic_weight=0.5):
        self.faiss_worker = faiss_worker
        self.bm25_worker = bm25_worker
        self.semantic_weight = semantic_weight  # 0.0 = BM25 only, 1.0 = FAISS only

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        top_k = payload.get("top_k", settings.TOP_K)
        shard_id = payload.get("shard_id", 0)
        num_shards = payload.get("num_shards", getattr(settings, "NUM_SHARDS", 1))

        # 1. Run FAISS and BM25
        faiss_result = self.faiss_worker.process({
            "vector": payload.get("vector"),
            "top_k": top_k * 2,
            "shard_id": shard_id,
            "num_shards": num_shards
        })
        bm25_result = self.bm25_worker.process({
            "tokens": payload.get("tokens", []),
            "limit": top_k * 2,
            "shard_id": shard_id,
            "num_shards": num_shards
        })

        faiss_candidates = faiss_result.get("candidates", [])   # list of (mem_id, distance)
        bm25_candidates = bm25_result.get("candidates", [])     # list of (mem_id, score)

        # 2. Build rank dictionaries (1‑indexed, lower rank = better)
        faiss_rank = {mem_id: idx+1 for idx, (mem_id, _) in enumerate(faiss_candidates)}
        bm25_rank   = {mem_id: idx+1 for idx, (mem_id, _) in enumerate(bm25_candidates)}

        # 3. RRF fusion
        k = getattr(settings, "RRF_K", 60)  # configurable
        rrf_scores = {}
        all_ids = set(faiss_rank.keys()) | set(bm25_rank.keys())

        for mem_id in all_ids:
            faiss_r = faiss_rank.get(mem_id, float('inf'))  # missing = infinity
            bm25_r   = bm25_rank.get(mem_id, float('inf'))
            # RRF: 1/(rank + k) – missing ranks get 0
            score = 0.0
            if faiss_r != float('inf'):
                score += 1.0 / (faiss_r + k)
            if bm25_r != float('inf'):
                score += 1.0 / (bm25_r + k)
            rrf_scores[mem_id] = score

        # 4. Sort and truncate
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_candidates = sorted_candidates[:top_k]

        fusion_time = (time.perf_counter() - start) * 1000
        debug(f"FusionWorker (RRF, k={k}): fusion={fusion_time:.2f}ms, count={len(top_candidates)}")

        return {
            "source": "fusion",
            "candidates": top_candidates,
            "count": len(top_candidates)
        }
