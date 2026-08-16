from sentence_transformers import SentenceTransformer
from cache.config import settings
from core.logger import debug


class Embedder:
    def __init__(self, model_name: str = None):
        model_name = model_name or settings.EMBEDDING_MODEL
        try:
            self.model = SentenceTransformer(model_name)
            self._query_cache = {}  # ← NEW: Query embedding cache
            debug(f"[Embedder] Loaded model: {model_name}")
        except Exception as e:
            debug(f"[Embedder] Failed to load model: {e}")
            raise

    def embed(self, text: str):
        """Embed a single text string."""
        if not text:
            return []

        # ← NEW: Check cache
        if text in self._query_cache:
            return self._query_cache[text]

        # Compute and cache
        vec = self.model.encode(text).tolist()
        self._query_cache[text] = vec
        return vec

    def embed_many(self, texts: list):
        """Embed multiple texts in batch for efficiency, preserving order."""
        if not texts:
            return []

        result = [None] * len(texts)  # Pre-allocate
        to_encode = []
        to_encode_indices = []

        for i, text in enumerate(texts):
            if not text:
                result[i] = []
            elif text in self._query_cache:
                result[i] = self._query_cache[text]
            else:
                to_encode.append(text)
                to_encode_indices.append(i)

        if to_encode:
            embeddings = self.model.encode(to_encode).tolist()
            for idx, vec in zip(to_encode_indices, embeddings):
                result[idx] = vec
                self._query_cache[texts[idx]] = vec

        return result

    def clear_cache(self):
        """Clear the query embedding cache."""
        self._query_cache.clear()

    def __repr__(self) -> str:
        return f"Embedder(model={settings.EMBEDDING_MODEL}, cache_size={len(self._query_cache)})"
