"""
score_finalizer.py
Combines ranking signals into final memory confidence.
"""
import math
from cache.config import settings


class ScoreFinalizer:
    def __init__(
        self,
        relevance_weight=None,
        importance_weight=None,
        recency_weight=None,
        diversity_weight=None,
        attribute_weight=None,
        bm25_weight=None,          # ← new
        debug=False
    ):
        self.weights = {
            "relevance": relevance_weight or settings.FINALIZER_RELEVANCE,
            "importance": importance_weight or settings.FINALIZER_IMPORTANCE,
            "recency": recency_weight or settings.FINALIZER_RECENCY,
            "diversity": diversity_weight or settings.FINALIZER_DIVERSITY,
            "attribute": attribute_weight or settings.FINALIZER_ATTRIBUTE,
            "bm25": bm25_weight or getattr(settings, "FINALIZER_BM25", 0.10)  # default 0.10
        }
        self.debug = debug
        self.last_diagnostics = {}

    def finalize(self, candidates, weights=None):
        active_weights = weights or self.weights

        for candidate in candidates:
            relevance = 1 / (1 + math.exp(-candidate.normalized_score))
            importance = candidate.importance_score
            recency = candidate.recency_score
            diversity = candidate.diversity_score
            attribute = candidate.attribute_score

            # Get BM25 score if present
            bm25 = getattr(candidate, 'bm25_score', 0.0)
            bm25_squashed = 1 / (1 + math.exp(-bm25))

            # Squash attribute
            attribute_squashed = 1 / (1 + math.exp(-attribute * 2))

            final = (
                relevance * active_weights["relevance"]
                + importance * active_weights["importance"]
                + recency * active_weights["recency"]
                + (1 - diversity) * active_weights["diversity"]
                + attribute_squashed * active_weights["attribute"]
                + bm25_squashed * active_weights["bm25"]   # ← added
            )

            candidate.final_score = final
            candidate.diagnostics["squashed_relevance"] = relevance
            candidate.diagnostics["squashed_bm25"] = bm25_squashed
            candidate.diagnostics["bm25_raw"] = bm25

        candidates.sort(key=lambda c: c.final_score, reverse=True)
        return candidates
