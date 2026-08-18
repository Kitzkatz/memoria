class TokenEstimator:

    # Average chars per token for different language families
    # English: ~4, Chinese/Japanese: ~2, code: ~3
    CHARS_PER_TOKEN = 4

    def estimate(self, text: str) -> int:
        """
        Estimate token count for a text string.

        Uses a rough heuristic: 1 token ≈ 4 characters (English average).
        This is a fast approximation for token budgeting.

        Args:
            text: The text to estimate

        Returns:
            int: Estimated token count
        """
        if not text:
            return 0

        # Basic estimation
        return max(1, len(text) // self.CHARS_PER_TOKEN)

    def estimate_many(self, texts: list) -> list:
        """Estimate token counts for multiple texts."""
        return [self.estimate(t) for t in texts]

    def estimate_batch(self, texts: list) -> int:
        """Estimate total tokens for a batch of texts."""
        return sum(self.estimate(t) for t in texts)

    def estimate_truncate(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within max_tokens.
        Uses character-based approximation.
        """
        if not text or max_tokens <= 0:
            return ""

        estimated_chars = max_tokens * self.CHARS_PER_TOKEN
        if len(text) <= estimated_chars:
            return text

        # Truncate at word boundary if possible
        truncated = text[:estimated_chars]
        # Find last space to avoid cutting in the middle of a word
        last_space = truncated.rfind(' ')
        if last_space > estimated_chars * 0.8:  # Only cut at space if it's not too early
            truncated = truncated[:last_space]

        return truncated + "..."
