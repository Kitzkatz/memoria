import numpy as np
from core.logger import debug


class ScoreNormalizer:
    def normalize(self, candidates):
        if not candidates:
            return candidates

        scores = np.array([c.base_score for c in candidates], dtype=float)

        mean = scores.mean()
        std = scores.std()

        if std == 0:
            std = 1e-6
            debug("[ScoreNormalizer] Zero standard deviation, using epsilon")

        for candidate in candidates:
            candidate.normalized_score = (candidate.base_score - mean) / std
            candidate.diagnostics["normalizer"] = {
                "base": candidate.base_score,
                "normalized": candidate.normalized_score
            }

        debug(f"[ScoreNormalizer] Normalized {len(candidates)} candidates (mean={mean:.4f}, std={std:.4f})")
        return candidates
