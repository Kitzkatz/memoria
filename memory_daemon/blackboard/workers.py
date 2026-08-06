import time
from typing import Dict, Any, List, Optional
from cache.config import settings

try:
    from core.logger import debug
except ImportError:
    def debug(*args):
        print("[DEBUG]", *args)


class Worker:
    """Base class for all workers."""
    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class FAISSWorker(Worker):
    """Worker that performs FAISS vector search."""
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        query_vec = payload.get("vector")
        top_k = payload.get("top_k", settings.TOP_K)
        if query_vec is None:
            return {"error": "No vector provided", "source": "faiss", "candidates": []}

        ids, distances = self.vector_store.search(query_vec, k=top_k)
        candidates = list(zip(ids, distances))
        search_time = (time.perf_counter() - start) * 1000

        debug(f"FAISSWorker: search={search_time:.2f}ms, count={len(ids)}")
        return {
            "source": "faiss",
            "candidates": candidates,
            "count": len(ids)
        }


class BM25Worker(Worker):
    """Worker that performs BM25 lexical search."""
    def __init__(self, bm25_ranker, inverted_index):
        self.bm25_ranker = bm25_ranker
        self.inverted_index = inverted_index

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        query_tokens = payload.get("tokens", [])
        limit = payload.get("limit", settings.TOP_K)
        
        # --- FIX: ensure limit is an int ---
        if not isinstance(limit, int):
            limit = int(limit) if limit else settings.TOP_K

        if not query_tokens or not self.inverted_index:
            return {"error": "No tokens or inverted index", "source": "bm25", "candidates": []}

        candidate_ids = set()
        for token in query_tokens:
            candidate_ids.update(self.inverted_index.search(token))
        candidate_ids = list(candidate_ids)

        # --- FIX: ensure candidate_ids are ints ---
        candidate_ids = [int(x) for x in candidate_ids if x is not None]

        if not candidate_ids:
            return {"error": "No candidates found", "source": "bm25", "candidates": []}

        candidates = [
            (doc_id, self.bm25_ranker.score(query_tokens, doc_id))
            for doc_id in candidate_ids[:limit]
        ]

        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:limit]

        build_time = (time.perf_counter() - start) * 1000
        debug(f"BM25Worker: build={build_time:.2f}ms, count={len(candidates)}")

        return {
            "source": "bm25",
            "candidates": candidates,
            "count": len(candidates)
        }


class GraphWorker(Worker):
    """Worker that performs graph-based retrieval using Numpy graph."""
    def __init__(self, numpy_graph):
        self.numpy_graph = numpy_graph

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        entities = payload.get("entities", [])
        depth = payload.get("depth", 2)
        limit = payload.get("limit", settings.GRAPH_TOP_K)
        if not entities or not self.numpy_graph:
            return {"error": "No entities or graph", "source": "graph", "candidates": []}

        memory_ids = self.numpy_graph.multi_hop_search(entities, depth=depth, limit=limit)
        # Ensure memory_ids are ints
        memory_ids = [int(x) for x in memory_ids if x is not None]
        candidates = [(mem_id, 0.0) for mem_id in memory_ids]
        search_time = (time.perf_counter() - start) * 1000

        debug(f"GraphWorker: search={search_time:.2f}ms, count={len(candidates)}")
        return {
            "source": "graph",
            "candidates": candidates,
            "count": len(candidates)
        }


class PhraseWorker(Worker):
    """Worker that performs phrase search using the inverted index."""
    def __init__(self, inverted_index):
        self.inverted_index = inverted_index

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        phrases = payload.get("phrases", [])
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
        candidates = candidates[:100]

        phrase_time = (time.perf_counter() - start) * 1000
        debug(f"PhraseWorker: phrase_time={phrase_time:.2f}ms, count={len(candidates)}")

        return {
            "source": "phrase",
            "candidates": candidates,
            "count": len(candidates)
        }


class AttributeWorker(Worker):
    """Worker that performs attribute-based retrieval."""
    def __init__(self, db):
        self.db = db

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        subject = payload.get("subject")
        attribute = payload.get("attribute")
        if not subject or not attribute:
            return {"error": "No subject or attribute", "source": "attribute", "candidates": []}

        rows = self.db.search_attribute(subject, attribute)
        candidates = [(row["id"], 0.0) for row in rows]

        return {
            "source": "attribute",
            "candidates": candidates,
            "count": len(candidates)
        }
