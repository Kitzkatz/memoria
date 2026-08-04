class TokenEstimator:

    def estimate(self, text: str) -> int:

        # rough heuristic:
        # 1 token ≈ 4 characters (English average)

        if not text:
            return 0

        return len(text) // 4
