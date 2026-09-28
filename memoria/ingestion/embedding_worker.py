import torch
from sentence_transformers import SentenceTransformer
from cache.config import settings
from core.logger import debug
from collections import OrderedDict


class Embedder:
    def __init__(self, model_name: str = None, max_chars: int = 512):
        """
        Embedder with fast character-based truncation, configurable skip,
        and bounded LRU query embedding cache.
        """
        model_name = model_name or settings.EMBEDDING_MODEL
        self.max_chars = max_chars

        self.skip = getattr(settings, "SKIP_EMBEDDING", False)

        self._query_cache = OrderedDict()
        self._query_cache_max_size = getattr(
            settings,
            "EMBEDDER_QUERY_CACHE_MAX_SIZE",
            10000,
        )

        # Memoria targets low-resource CPU machines. Keep Torch's
        # intra-op parallelism bounded rather than allowing it to spawn
        # a large worker pool.
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Torch may already have initialized its thread pools.
            pass

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
            vec = self._query_cache.pop(text)
            self._query_cache[text] = vec
            return vec

        length = max_chars if max_chars is not None else self.max_chars
        truncated = text[:length] if len(text) > length else text

        vec = self.model.encode(
            truncated,
            show_progress_bar=False,
        ).tolist()

        self._query_cache[text] = vec
        self._trim_query_cache()

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
                vec = self._query_cache.pop(text)
                self._query_cache[text] = vec
                result[i] = vec

            else:
                truncated = text[:length] if len(text) > length else text
                to_encode.append(truncated)
                to_encode_indices.append(i)

        if to_encode:
            embeddings = self.model.encode(
                to_encode,
                show_progress_bar=False,
                batch_size=8,
            ).tolist()

            for idx, vec in zip(to_encode_indices, embeddings):
                text = texts[idx]
                result[idx] = vec
                self._query_cache[text] = vec

            self._trim_query_cache()

        return result

    def _trim_query_cache(self):
        """Evict least-recently-used query embeddings when the cache is full."""
        if len(self._query_cache) <= self._query_cache_max_size:
            return

        to_remove = len(self._query_cache) - self._query_cache_max_size

        for _ in range(to_remove):
            self._query_cache.popitem(last=False)

        debug(
            f"[Embedder] Query cache trimmed to "
            f"{len(self._query_cache)} entries (LRU)",
            category="cache",
        )

    def clear_cache(self):
        self._query_cache.clear()

    def __repr__(self) -> str:
        return (
            f"Embedder("
            f"model={settings.EMBEDDING_MODEL}, "
            f"cache_size={len(self._query_cache)}, "
            f"cache_max={self._query_cache_max_size}"
            f")"
        )
