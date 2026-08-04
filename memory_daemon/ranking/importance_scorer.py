class ImportanceScorer:
    def score(self, text: str, metadata=None):
        score = 0.3
        t = text.lower()
        if "remember this" in t:
            score += 0.4
        if len(text) > 100:
            score += 0.1
        if metadata and metadata.get("goal"):
            score += 0.2
        return min(score, 1.0)
