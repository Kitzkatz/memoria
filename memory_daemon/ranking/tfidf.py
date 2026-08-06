# ranking/tfidf.py
import math
from collections import Counter
from typing import List, Dict

class TFIDF:
    def __init__(self):
        self.idf: Dict[str, float] = {}
        self.corpus_size = 0

    def build(self, corpus_tokens: List[List[str]]):
        """Build TF/IDF from a list of tokenized documents."""
        self.corpus_size = len(corpus_tokens)
        doc_freq = Counter()
        for doc in corpus_tokens:
            for term in set(doc):
                doc_freq[term] += 1
        for term, df in doc_freq.items():
            self.idf[term] = math.log((self.corpus_size + 1) / (df + 1)) + 1

    def tf(self, term: str, doc_tokens: List[str]) -> float:
        """Term frequency in a document."""
        return doc_tokens.count(term) / max(len(doc_tokens), 1)

    def score(self, term: str, doc_tokens: List[str]) -> float:
        """TF/IDF score for a single term in a document."""
        if term not in self.idf:
            return 0.0
        return self.tf(term, doc_tokens) * self.idf[term]

    def document_score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """TF/IDF score for a document against a query."""
        if not query_tokens or not doc_tokens:
            return 0.0
        score = 0.0
        for term in set(query_tokens):
            score += self.score(term, doc_tokens)
        return score / len(set(query_tokens))
