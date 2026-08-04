import time
from typing import Dict, Any, List, Optional
from .core import Blackboard, BlackboardEntry

class Worker:
    """Base class for all workers."""
    def __init__(self, blackboard: Blackboard):
        self.blackboard = blackboard

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class FAISSWorker(Worker):
    """Worker that performs FAISS vector search."""
    def __init__(self, blackboard: Blackboard, vector_store):
        super().__init__(blackboard)
        self.vector_store = vector_store

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query_vec = payload.get("vector")
        top_k = payload.get("top_k", 100)
        if query_vec is None:
            return {"error": "No vector provided", "candidates": []}

        ids, distances = self.vector_store.search(query_vec, k=top_k)
        return {
            "source": "faiss",
            "candidates": list(zip(ids, distances)),
            "count": len(ids)
        }


class BM25Worker(Worker):
    """Worker that performs BM25 lexical search."""
    def __init__(self, blackboard: Blackboard, bm25_ranker, inverted_index):
        super().__init__(blackboard)
        self.bm25_ranker = bm25_ranker
        self.inverted_index = inverted_index

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query_tokens = payload.get("tokens", [])
        if not query_tokens:
            return {"error": "No tokens provided", "candidates": []}

        # Get candidate IDs from inverted index (or use all docs if index is empty)
        candidate_ids = self.inverted_index.search(query_tokens) if self.inverted_index else None
        if candidate_ids is None:
            # Fallback: score all documents
            scores = self.bm25_ranker.get_scores(query_tokens)
            candidates = [(i, scores[i]) for i in range(len(scores)) if scores[i] > 0]
            candidates.sort(key=lambda x: x[1], reverse=True)
        else:
            # Score only candidates from inverted index
            scores = self.bm25_ranker.get_scores(query_tokens)
            candidates = [(doc_id, scores[doc_id]) for doc_id in candidate_ids if scores[doc_id] > 0]
            candidates.sort(key=lambda x: x[1], reverse=True)

        return {
            "source": "bm25",
            "candidates": candidates,
            "count": len(candidates)
        }


class GraphWorker(Worker):
    """Worker that performs graph-based retrieval."""
    def __init__(self, blackboard: Blackboard, graph_search, entity_store):
        super().__init__(blackboard)
        self.graph_search = graph_search
        self.entity_store = entity_store

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        entities = payload.get("entities", [])
        if not entities:
            return {"error": "No entities provided", "candidates": []}

        memory_ids = self.graph_search.search(entities, depth=1)
        return {
            "source": "graph",
            "candidates": memory_ids,
            "count": len(memory_ids)
        }


class RankingWorker(Worker):
    """Worker that ranks candidates using BM25 (or fallback)."""
    def __init__(self, blackboard: Blackboard, ranker, bm25_ranker=None):
        super().__init__(blackboard)
        self.ranker = ranker
        self.bm25_ranker = bm25_ranker

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        candidates = payload.get("candidates", [])
        query = payload.get("query")
        if not candidates or not query:
            return {"error": "Missing candidates or query", "ranked": []}

        # Use BM25 if available, otherwise fallback to the base ranker
        if self.bm25_ranker:
            # Here you'd integrate BM25 scores into the ranking
            # For now, we just pass through
            ranked = self.ranker.rank(candidates, query)
        else:
            ranked = self.ranker.rank(candidates, query)

        return {
            "source": "ranker",
            "ranked": ranked,
            "count": len(ranked)
        }
