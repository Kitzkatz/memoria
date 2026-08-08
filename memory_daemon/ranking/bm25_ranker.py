import math
from collections import Counter
from typing import List, Dict, Any, Optional


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.idf = {}
        self._built = False

    def build(self, corpus_tokens: List[List[str]]):
        """Build the BM25 index from a list of tokenized documents."""
        if not corpus_tokens:
            self._built = True
            return

        self.corpus = corpus_tokens
        self.doc_lengths = [len(doc) for doc in corpus_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)

        # Term frequency in corpus
        doc_freq = Counter()
        for doc in corpus_tokens:
            for term in set(doc):
                doc_freq[term] += 1

        # Compute IDF
        N = len(corpus_tokens)
        for term, df in doc_freq.items():
            self.idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

        self._built = True

    def score(self, query_tokens: List[str], doc_id: int) -> float:
        """Compute BM25 score for a single document."""
        if not self._built or doc_id >= len(self.corpus) or not query_tokens:
            return 0.0

        doc = self.corpus[doc_id]
        doc_len = self.doc_lengths[doc_id]

        score = 0.0
        term_freq = Counter(doc)

        for term in query_tokens:
            if term not in self.idf:
                continue
            tf = term_freq.get(term, 0)
            if tf == 0:
                continue

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_length))
            score += self.idf[term] * (numerator / denominator)

        return score

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        """Return scores for all documents. DEPRECATED: Use score_ids() instead."""
        if not self._built or not query_tokens:
            return [0.0] * len(self.corpus)
        return [self.score(query_tokens, i) for i in range(len(self.corpus))]

    def score_ids(self, query_tokens: List[str], doc_ids: List[int]) -> Dict[int, float]:
        """
        Compute BM25 scores for specific document IDs only.
        This is much faster than get_scores() for candidate scoring.
        """
        if not self._built or not query_tokens or not doc_ids:
            return {doc_id: 0.0 for doc_id in doc_ids}

        results = {}
        for doc_id in doc_ids:
            if 0 <= doc_id < len(self.corpus):
                results[doc_id] = self.score(query_tokens, doc_id)
            else:
                results[doc_id] = 0.0

        return results
