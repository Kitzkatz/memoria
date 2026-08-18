from sentence_transformers import SentenceTransformer
from cache.config import settings
from core.logger import debug


class Embedder:
    def __init__(self, model_name: str = None, max_chars: int = 512):
        """
        Embedder with fast character‑based truncation and configurable skip.
        """
        model_name = model_name or settings.EMBEDDING_MODEL
        self.max_chars = max_chars
        # Read skip flag from config (default False)
        self.skip = getattr(settings, 'SKIP_EMBEDDING', False)
        self._query_cache = {}

        try:
            self.model = SentenceTransformer(model_name)
            debug(f"[Embedder] Loaded model: {model_name}")
        except Exception as e:
            debug(f"[Embedder] Failed to load model: {e}")
            raise

    def embed(self, text: str, max_chars: int = None):
        """Embed a single text, with optional truncation."""
        if self.skip or not text:
            return []
        if text in self._query_cache:
            return self._query_cache[text]

        length = max_chars if max_chars is not None else self.max_chars
        truncated = text[:length] if len(text) > length else text
        vec = self.model.encode(truncated, show_progress_bar=False).tolist()
        self._query_cache[text] = vec
        return vec

    def embed_many(self, texts: list, max_chars: int = None):
        """
        Batch embed multiple texts with truncation.
        If skip is True, returns empty vectors for all texts.
        """
        if self.skip or not texts:
            return [[] for _ in texts]

        length = max_chars if max_chars is not None else self.max_chars
        result = [None] * len(texts)
        to_encode = []
        to_encode_indices = []

        for i, text in enumerate(texts):
            if not text:
                result[i] = []
            elif text in self._query_cache:
                result[i] = self._query_cache[text]
            else:
                truncated = text[:length] if len(text) > length else text
                to_encode.append(truncated)
                to_encode_indices.append(i)

        if to_encode:
            embeddings = self.model.encode(to_encode, show_progress_bar=False).tolist()
            for idx, vec in zip(to_encode_indices, embeddings):
                result[idx] = vec
                self._query_cache[texts[idx]] = vec

        return result

    def clear_cache(self):
        self._query_cache.clear()

    def __repr__(self) -> str:
        return f"Embedder(model={settings.EMBEDDING_MODEL}, cache_size={len(self._query_cache)})"
