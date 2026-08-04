from core.logger import debug
"""
mmr_reranker.py

Vector-space Maximal Marginal Relevance

Goals
-----
- Preserve relevance
- Reduce duplicate memories
- Produce diagnostics
- Stable ordering
"""

import math
from typing import Dict, List, Any

class MMRReranker:

    def __init__(
        self,
        lambda_param: float = 0.50,
        normalize_scores: bool = True,
        debug: bool = False,
        adaptive_lambda=True
    ):

        # Higher = relevance
        # Lower = diversity
        self.lambda_param = lambda_param

        self.normalize_scores = normalize_scores

        self.debug = debug

        self.adaptive_lambda = adaptive_lambda

        # cosine cache
        self._cache: Dict[tuple, float] = {}

        self.last_diagnostics = {}

        # ---------------------------------
    # Cosine Similarity
    # ---------------------------------

    def _cosine(self, a, b):

        if a is None or b is None:
            return 0.0

        key = tuple(sorted((id(a), id(b))))

        if key in self._cache:
            return self._cache[key]

        dot = sum(x * y for x, y in zip(a, b))

        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))

        if mag_a == 0 or mag_b == 0:
            sim = 0.0
        else:
            sim = dot / (mag_a * mag_b)

        self._cache[key] = sim

        return sim


    # ---------------------------------
    # Normalize Scores
    # ---------------------------------

    def _normalize_scores(self, candidates):

        if not candidates:
            return

        scores = [
            c.get("score", 0.0)
            for c in candidates
        ]

        low = min(scores)
        high = max(scores)

        if high == low:

            for c in candidates:
                c["_norm_score"] = 1.0

            return

        rng = high - low

        for c in candidates:

            c["_norm_score"] = (
                c.get("score", 0.0) - low
            ) / rng

    # ---------------------------------
    # Diagnostics
    # ---------------------------------

    def _build_diagnostics(
        self,
        before,
        after,
        lambda_used
    ):

        before_ids = [
            x.get(
                "memory_id",
                x.get("id")
            )
            for x in before
        ]

        after_ids = [
            x.get(
                "memory_id",
                x.get("id")
            )
            for x in after
        ]


        moves = 0

        for idx, item in enumerate(after_ids):

            if item in before_ids:

                old = before_ids.index(item)

                moves += abs(
                    old - idx
                )


        return {

            "candidate_count":
                len(before),

            "returned_count":
                len(after),

            "reordered":
                before_ids != after_ids,

            "total_moves":
                moves,

            "lambda":
                lambda_used,

            "avg_mmr_score":
                round(
                    sum(
                        x.get(
                            "_mmr_score",
                            0
                        )
                        for x in after
                    )
                    /
                    max(len(after),1),
                    4
                ),
            "avg_diversity":
                round(
                    sum(
                        x.get("_diversity",0)
                        for x in after
                    )
                    /
                    max(len(after),1),
                    4
                ),

        }
    # ---------------------------------
    # Maximal Marginal Relevance
    # ---------------------------------

    def rerank(self, results, k=5):

        #
        # Reset cosine cache every query
        #

        self._cache.clear()

        if not results:
            return []

        #
        # Copy candidates so we never mutate
        # the caller's list.
        #

        candidates = [dict(r) for r in results]

        #
        # Keep the original ordering for diagnostics.
        #

        before_order = [dict(r) for r in candidates]

        #
        # Optional normalization.
        #

        if self.normalize_scores:
            self._normalize_scores(candidates)

        else:
            for c in candidates:
                c["_norm_score"] = c.get("score", 0.0)

        #
        # If requesting more than available,
        # simply clamp.
        #

        k = min(k, len(candidates))

        #
        # Working containers.
        #

        selected = []
        remaining = candidates.copy()

        #
        # Diagnostics
        #

        mmr_changed = False
        mmr_moves = 0

        if self.debug:

            debug()

            debug("========== MMR ==========")

            debug("Candidates:", len(candidates))

            debug("Lambda:", self.lambda_param)

            debug("k:", k)

            

        #
        # ---------------------------------
        # Seed with highest relevance
        # ---------------------------------
        #

        first = max(
            remaining,
            key=lambda x: x["_norm_score"]
        )

        selected.append(first)

        remaining.remove(first)

        if self.debug:

            debug()

            debug(
                "Seed:",
                first.get(
                    "memory_id",
                    first.get("id")
                ),
                "score:",
                round(
                    first["_norm_score"],
                    4
                )
            )

        current_lambda = self.lambda_param


        if self.adaptive_lambda:

            candidate_count = len(candidates)


            if candidate_count <= 5:

                current_lambda = 0.85


            elif candidate_count <= 15:

                current_lambda = 0.65


            else:

                current_lambda = 0.50
        #
        # ---------------------------------
        # Greedy MMR Selection
        # ---------------------------------
        #

        while remaining and len(selected) < k:

            best_candidate = None

            best_score = float("-inf")


            for candidate in remaining:

                relevance = candidate["_norm_score"]


                #
                # Maximum similarity to
                # anything already selected.
                #

                similarities = []

                for chosen in selected:

                    similarity = self._cosine(

                        candidate.get("embedding"),

                        chosen.get("embedding")

                    )

                    similarities.append(similarity)

                if similarities:

                    diversity_penalty = (
                        max(similarities) * 0.7
                        +
                        (
                            sum(similarities)
                            /
                            len(similarities)
                        )
                        * 0.3
                    )

                else:

                    diversity_penalty = 0.0


                #
                # Standard MMR equation
                #

                mmr_score = (

                    current_lambda * relevance

                    -

                    (1.0 - current_lambda)
                    * diversity_penalty

                )


                candidate["_mmr_score"] = mmr_score

                candidate["_diversity"] = diversity_penalty


                if mmr_score > best_score:

                    best_score = mmr_score

                    best_candidate = candidate


            selected.append(best_candidate)

            remaining.remove(best_candidate)


            if self.debug:

                debug(

                    "Select:",

                    best_candidate.get(
                        "memory_id",
                        best_candidate.get("id")
                    ),

                    "rel=",

                    round(
                        best_candidate["_norm_score"],
                        4
                    ),

                    "div=",

                    round(
                        best_candidate["_diversity"],
                        4
                    ),

                    "mmr=",

                    round(
                        best_candidate["_mmr_score"],
                        4
                    )

                )

        self.last_diagnostics = self._build_diagnostics(
            before_order,
            selected,
            current_lambda
        )


        if self.debug:

            debug()
            debug("[MMR DIAGNOSTICS]")
            debug(
                self.last_diagnostics
            )


        return selected
