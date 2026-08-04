from core.logger import debug
from core.token_estimator import TokenEstimator
from cache.config import settings


class ContextBuilder:

    def __init__(
        self,
        max_memories=None,
        min_score=None,
        token_budget=None
    ):

        self.max_memories = max_memories or settings.CONTEXT_MAX_MEMORIES
        self.min_score = min_score or settings.CONTEXT_MIN_SCORE
        self.token_budget = token_budget or settings.CONTEXT_TOKEN_BUDGET
        self.tokens = TokenEstimator()


    # ----------------------------------
    # BUILD CONTEXT
    # ----------------------------------

    def build(self, candidates):

        if not candidates:
            return []

        debug("\n[CONTEXT BUILDER] START")
        debug("[INPUT] Total candidates:", len(candidates))

        #
        # Score filtering only
        #

        filtered = [
            c for c in candidates
            if c.base_score >= self.min_score
        ]

        debug("[FILTER] After threshold:", len(filtered))

        #
        # IMPORTANT:
        # Do NOT sort here.
        #
        # Ranking order belongs to:
        # MemoryRanker
        #
        # Diversity order belongs to:
        # MMR
        #

        selected = []
        used_tokens = 0

        for candidate in filtered:

            token_cost = candidate.memory.token_count

            if not token_cost:
                token_cost = self.tokens.estimate(
                    candidate.memory.text
                )

            if used_tokens + token_cost > self.token_budget:
                continue

            candidate.diagnostics["selected_reason"] = "score_budget"

            selected.append(candidate)
            used_tokens += token_cost

            if len(selected) >= self.max_memories:
                break

        debug("[BUDGET] Used tokens:", used_tokens)
        debug("[BUDGET] Selected:", len(selected))
        debug("[CONTEXT BUILDER] END\n")

        return selected

