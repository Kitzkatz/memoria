import math
from collections import Counter
from typing import List, Dict, Optional, Set


class TFIDF:
    def __init__(self):
        self.idf: Dict[str, float] = {}
        self.corpus_size = 0
        self._built = False

    def build(self, corpus_tokens: List[List[str]]):
        """Build TF/IDF from a list of tokenized documents."""
        if not corpus_tokens:
            self._built = True
            return

        self.corpus_size = len(corpus_tokens)
        doc_freq = Counter()
        for doc in corpus_tokens:
            for term in set(doc):
                doc_freq[term] += 1

        for term, df in doc_freq.items():
            self.idf[term] = math.log((self.corpus_size + 1) / (df + 1)) + 1

        self._built = True

    def tf(self, term: str, doc_tokens: List[str]) -> float:
        """Term frequency in a document."""
        if not doc_tokens:
            return 0.0
        return doc_tokens.count(term) / len(doc_tokens)

    def score(self, term: str, doc_tokens: List[str]) -> float:
        """TF/IDF score for a single term in a document."""
        if not self._built or term not in self.idf:
            return 0.0
        return self.tf(term, doc_tokens) * self.idf[term]

    def document_score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """TF/IDF score for a document against a query."""
        if not self._built or not query_tokens or not doc_tokens:
            return 0.0

        # Pre-compute query terms set once
        query_terms = set(query_tokens)
        if not query_terms:
            return 0.0

        score = 0.0
        for term in query_terms:
            score += self.score(term, doc_tokens)

        return score / len(query_terms)

    def batch_document_scores(
        self,
        query_tokens: List[str],
        doc_tokens_list: List[List[str]]
    ) -> List[float]:
        """Compute TF/IDF scores for multiple documents at once."""
        if not self._built or not query_tokens or not doc_tokens_list:
            return [0.0] * len(doc_tokens_list)

        query_terms = set(query_tokens)
        if not query_terms:
            return [0.0] * len(doc_tokens_list)

        results = []
        for doc_tokens in doc_tokens_list:
            score = 0.0
            for term in query_terms:
                score += self.score(term, doc_tokens)
            results.append(score / len(query_terms))

        return results

    def get_idf(self, term: str) -> float:
        """Get IDF for a single term."""
        return self.idf.get(term, 0.0)

    @property
    def built(self) -> bool:
        return self._built

    @property
    def num_terms(self) -> int:
        return len(self.idf)
