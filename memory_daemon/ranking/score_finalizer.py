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
        bm25_weight=None,
        debug=False
    ):
        self.weights = {
            "relevance": relevance_weight or settings.FINALIZER_RELEVANCE,
            "importance": importance_weight or settings.FINALIZER_IMPORTANCE,
            "recency": recency_weight or settings.FINALIZER_RECENCY,
            "diversity": diversity_weight or settings.FINALIZER_DIVERSITY,
            "attribute": attribute_weight or settings.FINALIZER_ATTRIBUTE,
            "bm25": bm25_weight or getattr(settings, "FINALIZER_BM25", 0.10)
        }
        self.debug = debug

    def finalize(self, candidates, weights=None):
        active_weights = weights or self.weights

        # Pre-compute which weights are non-zero for speed
        use_bm25 = active_weights.get("bm25", 0.0) > 0.001
        use_attribute = active_weights.get("attribute", 0.0) > 0.001
        use_diversity = active_weights.get("diversity", 0.0) > 0.001

        for candidate in candidates:
            relevance = 1 / (1 + math.exp(-candidate.normalized_score))
            importance = candidate.importance_score
            recency = candidate.recency_score

            # --- Diversity (skip if weight is near zero) ---
            if use_diversity:
                diversity = candidate.diversity_score
            else:
                diversity = 0.0

            # --- Attribute (skip if weight is near zero) ---
            if use_attribute:
                attribute = candidate.attribute_score
                attribute_squashed = 1 / (1 + math.exp(-attribute * 2))
            else:
                attribute_squashed = 0.0

            # --- BM25 (skip if weight is near zero) ---
            if use_bm25:
                bm25 = getattr(candidate, 'bm25_score', 0.0)
                bm25_squashed = 1 / (1 + math.exp(-bm25))
                candidate.diagnostics["squashed_bm25"] = bm25_squashed
                candidate.diagnostics["bm25_raw"] = bm25
            else:
                bm25_squashed = 0.0

            final = (
                relevance * active_weights["relevance"]
                + importance * active_weights["importance"]
                + recency * active_weights["recency"]
                + (1 - diversity) * active_weights["diversity"]
                + attribute_squashed * active_weights["attribute"]
                + bm25_squashed * active_weights["bm25"]
            )

            candidate.final_score = final
            candidate.diagnostics["squashed_relevance"] = relevance

        candidates.sort(key=lambda c: c.final_score, reverse=True)
        return candidates
