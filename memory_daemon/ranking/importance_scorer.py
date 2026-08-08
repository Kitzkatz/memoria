import re
from datetime import datetime, timezone
from typing import Optional, Dict, Any


class ImportanceScorer:
    """
    Heuristic importance scorer for memories.
    Scores are based on:
    - Explicit importance cues in text
    - Text length
    - Metadata signals (goal, type, etc.)
    - Optional recency boost
    """

    def __init__(
        self,
        base_score: float = 0.3,
        max_score: float = 1.0,
        importance_phrases: Optional[list] = None
    ):
        self.base_score = base_score
        self.max_score = max_score

        # Default importance phrases
        self.importance_phrases = importance_phrases or [
            "remember this",
            "important",
            "critical",
            "essential",
            "key point",
            "vital",
            "crucial",
            "don't forget",
            "noteworthy",
            "significant",
            "priority",
            "urgent",
            "must remember",
            "need to know",
        ]

        # Compile regex pattern for faster matching
        self._pattern = re.compile(
            r'\b(' + '|'.join(re.escape(p) for p in self.importance_phrases) + r')\b',
            re.IGNORECASE
        )

    def score(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[str] = None,
        access_count: Optional[int] = None
    ) -> float:
        """
        Compute importance score for a memory.

        Args:
            text: Memory text
            metadata: Optional metadata dict
            created_at: ISO timestamp for recency boost
            access_count: Number of times accessed for frequency boost

        Returns:
            Float score between 0.0 and 1.0
        """
        score = self.base_score

        # 1. Text-based importance cues
        if self._pattern.search(text.lower()):
            score += 0.3

        # 2. Length bonus (longer texts often more important)
        if len(text) > 100:
            score += 0.1
        elif len(text) > 500:
            score += 0.15
        elif len(text) > 1000:
            score += 0.2

        # 3. Metadata signals
        if metadata:
            # Goal-related
            if metadata.get("goal"):
                score += 0.2

            # Type-specific boosts
            mem_type = metadata.get("memory_type", "")
            if mem_type in ("science", "code", "procedural"):
                score += 0.1

            # Explicit importance in metadata
            if metadata.get("importance", 0.0) > 0.5:
                score += 0.1

            # If metadata contains any key indicating importance
            if any(k in metadata for k in ["priority", "importance", "significance"]):
                score += 0.05

        # 4. Recency boost (memories from today)
        if created_at:
            try:
                created = datetime.fromisoformat(created_at)
                now = datetime.now(timezone.utc)
                days_old = (now - created).days
                if days_old == 0:
                    score += 0.05
                elif days_old < 7:
                    score += 0.02
            except (ValueError, TypeError):
                pass

        # 5. Access frequency boost
        if access_count and access_count > 5:
            score += min(0.1, access_count * 0.01)

        return min(score, self.max_score)

    def score_batch(
        self,
        texts: list,
        metadatas: Optional[list] = None,
        created_at_list: Optional[list] = None
    ) -> list:
        """Score multiple memories at once."""
        if metadatas is None:
            metadatas = [None] * len(texts)
        if created_at_list is None:
            created_at_list = [None] * len(texts)

        return [
            self.score(text, metadata, created_at)
            for text, metadata, created_at in zip(texts, metadatas, created_at_list)
        ]
