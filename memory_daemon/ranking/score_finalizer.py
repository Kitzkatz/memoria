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
        debug=False,
        sigmoid_scale: float = 1.0,
        use_sigmoid: bool = True,
    ):
        # Use `is not None` so explicit 0.0 values work correctly.
        self.weights = {
            "relevance": (
                relevance_weight
                if relevance_weight is not None
                else settings.FINALIZER_RELEVANCE
            ),
            "importance": (
                importance_weight
                if importance_weight is not None
                else settings.FINALIZER_IMPORTANCE
            ),
            "recency": (
                recency_weight
                if recency_weight is not None
                else settings.FINALIZER_RECENCY
            ),
            "diversity": (
                diversity_weight
                if diversity_weight is not None
                else settings.FINALIZER_DIVERSITY
            ),
            "attribute": (
                attribute_weight
                if attribute_weight is not None
                else settings.FINALIZER_ATTRIBUTE
            ),
            "bm25": (
                bm25_weight
                if bm25_weight is not None
                else getattr(settings, "FINALIZER_BM25", 0.10)
            ),
        }

        self.debug = debug
        self.sigmoid_scale = sigmoid_scale
        self.use_sigmoid = use_sigmoid

    def finalize(self, candidates, weights=None):
        """
        Combine ranking signals into final_score.

        AttributeBooster stores its signals in candidate.diagnostics:

            entity_boost
            attribute_boost
            total_boost

        The finalizer uses `attribute_boost` as the attribute signal.
        """

        active_weights = weights or self.weights

        # Pre-compute enabled signals for speed.
        use_bm25 = active_weights.get("bm25", 0.0) > 0.001
        use_attribute = active_weights.get("attribute", 0.0) > 0.001
        use_diversity = active_weights.get("diversity", 0.0) > 0.001

        for candidate in candidates:
            # --------------------------------------------------
            # Relevance
            # --------------------------------------------------

            if self.use_sigmoid:
                scale = max(self.sigmoid_scale, 1e-6)

                relevance = 1.0 / (
                    1.0
                    + math.exp(
                        -candidate.normalized_score / scale
                    )
                )
            else:
                relevance = max(
                    0.0,
                    min(
                        1.0,
                        candidate.normalized_score / 5.0,
                    ),
                )

            # --------------------------------------------------
            # Core signals
            # --------------------------------------------------

            importance = candidate.importance_score
            recency = candidate.recency_score

            # --------------------------------------------------
            # Diversity
            # --------------------------------------------------

            if use_diversity:
                diversity = candidate.diversity_score
            else:
                diversity = 0.0

            # --------------------------------------------------
            # Attribute
            #
            # AttributeBooster now stores this in diagnostics.
            # Do NOT read candidate.attribute_score directly.
            # --------------------------------------------------

            if use_attribute:
                attribute = candidate.diagnostics.get(
                    "attribute_boost",
                    0.0,
                )

                attribute_squashed = 1.0 / (
                    1.0 + math.exp(-attribute * 2.0)
                )
            else:
                attribute = 0.0
                attribute_squashed = 0.0

            # --------------------------------------------------
            # BM25
            # --------------------------------------------------

            if use_bm25:
                bm25 = getattr(
                    candidate,
                    "bm25_score",
                    0.0,
                )

                bm25_squashed = 1.0 / (
                    1.0 + math.exp(-bm25)
                )

                candidate.diagnostics["bm25_raw"] = bm25
                candidate.diagnostics[
                    "squashed_bm25"
                ] = bm25_squashed
            else:
                bm25 = 0.0
                bm25_squashed = 0.0

            # --------------------------------------------------
            # Final score
            # --------------------------------------------------

            final = (
                relevance
                * active_weights["relevance"]
                + importance
                * active_weights["importance"]
                + recency
                * active_weights["recency"]
                + (1.0 - diversity)
                * active_weights["diversity"]
                + attribute_squashed
                * active_weights["attribute"]
                + bm25_squashed
                * active_weights["bm25"]
            )

            candidate.final_score = final

            # --------------------------------------------------
            # Diagnostics
            # --------------------------------------------------

            candidate.diagnostics["attribute_raw"] = attribute
            candidate.diagnostics[
                "squashed_attribute"
            ] = attribute_squashed

            candidate.diagnostics[
                "squashed_relevance"
            ] = relevance

            candidate.diagnostics[
                "finalizer_weights"
            ] = dict(active_weights)

            candidate.diagnostics[
                "use_sigmoid"
            ] = self.use_sigmoid

            candidate.diagnostics[
                "sigmoid_scale"
            ] = self.sigmoid_scale

            if self.debug:
                candidate.diagnostics[
                    "finalizer_components"
                ] = {
                    "relevance": (
                        relevance
                        * active_weights["relevance"]
                    ),
                    "importance": (
                        importance
                        * active_weights["importance"]
                    ),
                    "recency": (
                        recency
                        * active_weights["recency"]
                    ),
                    "diversity": (
                        (1.0 - diversity)
                        * active_weights["diversity"]
                    ),
                    "attribute": (
                        attribute_squashed
                        * active_weights["attribute"]
                    ),
                    "bm25": (
                        bm25_squashed
                        * active_weights["bm25"]
                    ),
                }

        candidates.sort(
            key=lambda c: c.final_score,
            reverse=True,
        )

        return candidates
