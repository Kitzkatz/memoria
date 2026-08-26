import numpy as np
from core.logger import debug
from cache.config import settings


class ScoreNormalizer:
    def normalize(self, candidates):
        """
        Normalize base_scores using the method defined in settings.
        Options:
          - "zscore": Standardization (mean=0, std=1) - default
          - "minmax": Min-Max scaling to [0, 1]
        Set SCORE_NORMALIZER_METHOD in cache/config.py to toggle.
        """
        if not candidates:
            return candidates

        method = getattr(settings, "SCORE_NORMALIZER_METHOD", "zscore")

        if method == "minmax":
            return self._normalize_minmax(candidates)
        else:  # zscore (default)
            return self._normalize_zscore(candidates)

    def _normalize_zscore(self, candidates):
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
                "normalized": candidate.normalized_score,
                "method": "zscore",
                "mean": float(mean),
                "std": float(std),
            }

        debug(
            f"[ScoreNormalizer] Z-score normalized {len(candidates)} candidates "
            f"(mean={mean:.4f}, std={std:.4f})"
        )
        return candidates

    def _normalize_minmax(self, candidates):
        scores = np.array([c.base_score for c in candidates], dtype=float)

        min_score = scores.min()
        max_score = scores.max()

        # Avoid division by zero
        if max_score - min_score < 1e-12:
            for candidate in candidates:
                candidate.normalized_score = 0.5
                candidate.diagnostics["normalizer"] = {
                    "base": candidate.base_score,
                    "normalized": 0.5,
                    "method": "minmax_flat",
                    "min": float(min_score),
                    "max": float(max_score),
                }
            debug(
                f"[ScoreNormalizer] Min-Max: all scores equal ({min_score:.4f}), set all to 0.5"
            )
            return candidates

        for candidate in candidates:
            normalized = (candidate.base_score - min_score) / (max_score - min_score)
            candidate.normalized_score = normalized
            candidate.diagnostics["normalizer"] = {
                "base": candidate.base_score,
                "normalized": normalized,
                "min": float(min_score),
                "max": float(max_score),
                "method": "minmax",
            }

        debug(
            f"[ScoreNormalizer] Min-Max normalized {len(candidates)} candidates "
            f"(min={min_score:.4f}, max={max_score:.4f})"
        )
        return candidates
