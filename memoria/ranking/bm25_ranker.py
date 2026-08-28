import math
from collections import Counter
from typing import List, Dict, Any, Optional


class BM25:
    """
    BM25 ranker with an explicit mapping between corpus positions and
    real memory/document IDs.

    BM25 internally stores documents by positional index because that is
    how the scoring arrays are organized. External callers should use
    memory IDs through score_ids().
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        # BM25 corpus is positional internally.
        self.corpus: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.idf: Dict[str, float] = {}

        # Explicit identity mapping:
        #
        # corpus position -> real memory/document ID
        # real memory/document ID -> corpus position
        #
        # This prevents BM25's positional indexes from being mistaken
        # for database memory IDs.
        self.doc_ids: List[Any] = []
        self.doc_id_to_index: Dict[Any, int] = {}

        self._built = False

    def build(
        self,
        corpus_tokens: List[List[str]],
        doc_ids: Optional[List[Any]] = None,
    ):
        """
        Build the BM25 index from tokenized documents.

        Args:
            corpus_tokens:
                Tokenized documents in corpus order.

            doc_ids:
                Optional real memory/document IDs corresponding to each
                corpus document.

                If omitted, corpus positions (0..N-1) are used as IDs
                for backward compatibility.
        """
        # Always reset the previous index completely before rebuilding.
        self.corpus = []
        self.doc_lengths = []
        self.avg_doc_length = 0.0
        self.idf = {}
        self.doc_ids = []
        self.doc_id_to_index = {}
        self._built = False

        if not corpus_tokens:
            self._built = True
            return

        if doc_ids is None:
            doc_ids = list(range(len(corpus_tokens)))

        if len(doc_ids) != len(corpus_tokens):
            raise ValueError(
                "BM25 build requires doc_ids to have the same length "
                "as corpus_tokens "
                f"(got {len(doc_ids)} IDs for {len(corpus_tokens)} documents)"
            )

        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("BM25 build requires unique document IDs")

        self.corpus = corpus_tokens
        self.doc_ids = list(doc_ids)
        self.doc_id_to_index = {
            doc_id: index
            for index, doc_id in enumerate(self.doc_ids)
        }

        self.doc_lengths = [len(doc) for doc in corpus_tokens]

        self.avg_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths)
            if self.doc_lengths
            else 0.0
        )

        # Term frequency in corpus.
        doc_freq = Counter()

        for doc in corpus_tokens:
            for term in set(doc):
                doc_freq[term] += 1

        # Compute IDF.
        N = len(corpus_tokens)

        for term, df in doc_freq.items():
            self.idf[term] = math.log(
                (N - df + 0.5) / (df + 0.5) + 1.0
            )

        self._built = True

    def score(self, query_tokens: List[str], doc_id: Any) -> float:
        """
        Compute BM25 score for a real memory/document ID.

        doc_id is resolved through the explicit ID -> corpus-position
        mapping. It is NOT assumed to be a corpus index.
        """
        if not self._built or not query_tokens:
            return 0.0

        corpus_index = self.doc_id_to_index.get(doc_id)

        if corpus_index is None:
            return 0.0

        doc = self.corpus[corpus_index]
        doc_len = self.doc_lengths[corpus_index]

        score = 0.0
        term_freq = Counter(doc)

        for term in query_tokens:
            idf = self.idf.get(term)

            if idf is None:
                continue

            tf = term_freq.get(term, 0)

            if tf == 0:
                continue

            numerator = tf * (self.k1 + 1.0)

            if self.avg_doc_length > 0.0:
                length_norm = (
                    1.0 - self.b
                    + self.b * (doc_len / self.avg_doc_length)
                )
            else:
                length_norm = 1.0

            denominator = tf + self.k1 * length_norm

            score += idf * (numerator / denominator)

        return score

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        """
        Return BM25 scores for all documents in corpus order.

        This preserves the legacy positional return format.

        For retrieval/candidate scoring using real memory IDs, prefer
        score_ids().
        """
        if not self._built or not query_tokens:
            return [0.0] * len(self.corpus)

        return [
            self._score_index(query_tokens, index)
            for index in range(len(self.corpus))
        ]

    def score_ids(
        self,
        query_tokens: List[str],
        doc_ids: List[Any],
    ) -> Dict[Any, float]:
        """
        Compute BM25 scores for specific real memory/document IDs only.

        This avoids scoring the entire corpus and preserves the distinction
        between database IDs and BM25's internal corpus positions.
        """
        if not doc_ids:
            return {}

        if not self._built or not query_tokens:
            return {
                doc_id: 0.0
                for doc_id in doc_ids
            }

        results = {}

        for doc_id in doc_ids:
            corpus_index = self.doc_id_to_index.get(doc_id)

            if corpus_index is None:
                results[doc_id] = 0.0
                continue

            results[doc_id] = self._score_index(
                query_tokens,
                corpus_index,
            )

        return results

    def _score_index(
        self,
        query_tokens: List[str],
        corpus_index: int,
    ) -> float:
        """
        Score a document by its internal corpus position.

        This is deliberately private. External callers should use score()
        or score_ids() with real document IDs.
        """
        if (
            not self._built
            or not query_tokens
            or corpus_index < 0
            or corpus_index >= len(self.corpus)
        ):
            return 0.0

        doc = self.corpus[corpus_index]
        doc_len = self.doc_lengths[corpus_index]

        score = 0.0
        term_freq = Counter(doc)

        for term in query_tokens:
            idf = self.idf.get(term)

            if idf is None:
                continue

            tf = term_freq.get(term, 0)

            if tf == 0:
                continue

            numerator = tf * (self.k1 + 1.0)

            if self.avg_doc_length > 0.0:
                length_norm = (
                    1.0 - self.b
                    + self.b * (doc_len / self.avg_doc_length)
                )
            else:
                length_norm = 1.0

            denominator = tf + self.k1 * length_norm

            score += idf * (numerator / denominator)

        return score
