import time
from typing import Dict, Any, List, Optional
from .core import Blackboard, BlackboardEntry
from cache.config import settings

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
        top_k = payload.get("top_k", settings.TOP_K)
        if query_vec is None:
            return {"error": "No vector provided", "candidates": []}

        ids, distances = self.vector_store.search(query_vec, k=top_k)
        candidates = list(zip(ids, distances))
        
        self.blackboard.post(BlackboardEntry(
            type="candidates",
            content={"source": "faiss", "candidates": candidates},
            source="faiss_worker"
        ))
        
        return {
            "source": "faiss",
            "candidates": candidates,
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
        limit = payload.get("limit", settings.TOP_K)
        if not query_tokens or not self.bm25_ranker:
            return {"error": "No tokens or BM25", "candidates": []}

        # Get candidate IDs from inverted index (union of all token postings)
        if self.inverted_index:
            candidate_ids = set()
            for token in query_tokens:
                candidate_ids.update(self.inverted_index.search(token))
            candidate_ids = list(candidate_ids)
        else:
            candidate_ids = None

        if candidate_ids is None:
            # Fallback: score all documents (slow)
            scores = self.bm25_ranker.get_scores(query_tokens)
            candidates = [(i, scores[i]) for i in range(len(scores)) if scores[i] > 0]
        else:
            # Score only candidates from inverted index (fast)
            scores = self.bm25_ranker.get_scores(query_tokens)
            candidates = [(doc_id, scores[doc_id]) for doc_id in candidate_ids if scores[doc_id] > 0]

        candidates.sort(key=lambda x: x[1], reverse=True)
        # Limit to TOP_K to match FAISS
        candidates = candidates[:limit]

        self.blackboard.post(BlackboardEntry(
            type="candidates",
            content={"source": "bm25", "candidates": candidates},
            source="bm25_worker"
        ))

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
        limit = payload.get("limit", settings.GRAPH_TOP_K)
        if not entities:
            return {"error": "No entities provided", "candidates": []}

        memory_ids = self.graph_search.search(entities, depth=1, limit=limit)
        
        self.blackboard.post(BlackboardEntry(
            type="candidates",
            content={"source": "graph", "candidates": memory_ids},
            source="graph_worker"
        ))

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

        if self.bm25_ranker:
            # Pass BM25 scores to ranking (already attached during pipeline)
            ranked = self.ranker.rank(candidates, query)
        else:
            ranked = self.ranker.rank(candidates, query)

        self.blackboard.post(BlackboardEntry(
            type="ranked",
            content={"source": "ranker", "ranked": ranked},
            source="ranking_worker"
        ))

        return {
            "source": "ranker",
            "ranked": ranked,
            "count": len(ranked)
        }


class PhraseWorker(Worker):
    """Worker that performs phrase search using the inverted index."""
    def __init__(self, blackboard: Blackboard, inverted_index):
        super().__init__(blackboard)
        self.inverted_index = inverted_index

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        phrases = payload.get("phrases", [])
        if not phrases or not self.inverted_index:
            return {"error": "No phrases or inverted index missing", "candidates": []}

        results = []
        for phrase_tokens in phrases:
            doc_ids = self.inverted_index.phrase_search(phrase_tokens)
            for doc_id in doc_ids:
                results.append((doc_id, 1.0))

        # Deduplicate
        unique = {}
        for doc_id, score in results:
            if doc_id not in unique or score > unique[doc_id]:
                unique[doc_id] = score

        candidates = list(unique.items())
        candidates.sort(key=lambda x: x[1], reverse=True)
        # Limit to e.g. 100 phrase matches
        candidates = candidates[:100]

        self.blackboard.post(BlackboardEntry(
            type="candidates",
            content={"source": "phrase", "candidates": candidates},
            source="phrase_worker"
        ))

        return {
            "source": "phrase",
            "candidates": candidates,
            "count": len(candidates)
        }
