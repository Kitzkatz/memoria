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

        # Score filtering
        filtered = [
            c for c in candidates
            if c.base_score >= self.min_score
        ]

        debug("[FILTER] After threshold:", len(filtered))

        selected = []
        used_tokens = 0
        skipped_by_budget = 0

        for candidate in filtered:
            # Get token cost safely
            token_cost = candidate.memory.token_count
            if token_cost is None or token_cost <= 0:
                token_cost = self.tokens.estimate(candidate.memory.text)

            # Ensure token_cost is an int
            token_cost = int(token_cost) if token_cost else 0

            if token_cost == 0:
                # Can't budget zero-token memories, but still include them
                selected.append(candidate)
                candidate.diagnostics["selected_reason"] = "score_budget_zero_token"
                continue

            if used_tokens + token_cost > self.token_budget:
                skipped_by_budget += 1
                candidate.diagnostics["selected_reason"] = "token_budget_exceeded"
                continue

            candidate.diagnostics["selected_reason"] = "score_budget"
            selected.append(candidate)
            used_tokens += token_cost

            if len(selected) >= self.max_memories:
                break

        if skipped_by_budget > 0:
            debug(f"[BUDGET] Skipped {skipped_by_budget} memories due to token budget")

        debug("[BUDGET] Used tokens:", used_tokens)
        debug("[BUDGET] Selected:", len(selected))
        debug("[CONTEXT BUILDER] END\n")

        return selected
