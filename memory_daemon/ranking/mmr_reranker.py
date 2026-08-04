from core.logger import debug
"""
mmr_reranker.py

Vector-space Maximal Marginal Relevance
"""

import math
from typing import Dict


class MMRReranker:

    def __init__(
        self,
        lambda_param: float = 0.50,
        debug: bool = False,
        adaptive_lambda=True
    ):

        # Higher = relevance
        # Lower = diversity
        self.lambda_param = lambda_param
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
    # Diagnostics
    # ---------------------------------

    def _build_diagnostics(self, before_ids, after, lambda_used):

        after_ids = [c.memory.id for c in after]

        moves = 0

        for idx, item in enumerate(after_ids):
            if item in before_ids:
                old = before_ids.index(item)
                moves += abs(old - idx)

        return {
            "candidate_count": len(before_ids),
            "returned_count": len(after),
            "reordered": before_ids != after_ids,
            "total_moves": moves,
            "lambda": lambda_used,
            "avg_mmr_score": round(
                sum(c.mmr_score for c in after) / max(len(after), 1),
                4
            ),
            "avg_diversity": round(
                sum(c.diversity_score for c in after) / max(len(after), 1),
                4
            ),
        }

    # ---------------------------------
    # Maximal Marginal Relevance
    # ---------------------------------

    def rerank(self, candidates, k=20):

        self._cache.clear()

        if not candidates:
            return []

        #
        # Copy candidates so we never mutate
        # the caller's list.
        #

        working = [c.model_copy() for c in candidates]
        working.sort(
            key=lambda c: c.normalized_score,
            reverse=True
        )

        working = working[:25]
        before_ids = [c.memory.id for c in working]

        k = min(k, len(working))

        selected = []
        remaining = working.copy()

        if self.debug:
            debug()
            debug("========== MMR ==========")
            debug("Candidates:", len(working))
            debug("Lambda:", self.lambda_param)
            debug("k:", k)

        #
        # Seed with highest relevance
        #

        

        

        current_lambda = self.lambda_param
        if self.adaptive_lambda:

            candidate_count = len(working)

            if candidate_count <= 10:
                current_lambda = 0.95

            elif candidate_count <= 30:
                current_lambda = 0.90

            elif candidate_count <= 75:
                current_lambda = 0.85

            else:
                current_lambda = 0.80

        first = max(remaining, key=lambda c: c.normalized_score)
        first.mmr_score = current_lambda * first.normalized_score
        first.diversity_score = 0.0
        selected.append(first)
        remaining.remove(first)

        if self.debug:
            debug()
            debug(
                "Seed:", first.memory.id,
                "score:", round(first.normalized_score, 4)
            )

        #
        # Greedy MMR Selection
        #

        while remaining and len(selected) < k:

            best_candidate = None
            best_score = float("-inf")

            for candidate in remaining:

                relevance = candidate.normalized_score

                similarities = [
                    self._cosine(candidate.embedding, chosen.embedding)
                    for chosen in selected
                ]

                if similarities:
                    diversity_penalty = max(similarities)
                    
                else:
                    diversity_penalty = 0.0

                mmr_score = (
                    (current_lambda + 0.10) * relevance
                    - (1.0 - current_lambda) * diversity_penalty
                )

                candidate.mmr_score = mmr_score
                candidate.diversity_score = diversity_penalty

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = candidate

            selected.append(best_candidate)
            remaining.remove(best_candidate)

            if self.debug:
                debug(
                    "Select:", best_candidate.memory.id,
                    "rel=", round(best_candidate.normalized_score, 4),
                    "div=", round(best_candidate.diversity_score, 4),
                    "mmr=", round(best_candidate.mmr_score, 4)
                )

        self.last_diagnostics = self._build_diagnostics(
            before_ids, selected, current_lambda
        )

        if self.debug:
            debug()
            debug("[MMR DIAGNOSTICS]")
            debug(self.last_diagnostics)

        return selected

