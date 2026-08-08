from sentence_transformers import SentenceTransformer
from cache.config import settings
from core.logger import debug


class Embedder:
    def __init__(self, model_name: str = None):
        model_name = model_name or settings.EMBEDDING_MODEL
        try:
            self.model = SentenceTransformer(model_name)
            debug(f"[Embedder] Loaded model: {model_name}")
        except Exception as e:
            debug(f"[Embedder] Failed to load model: {e}")
            raise

    def embed(self, text: str):
        """Embed a single text string."""
        if not text:
            return []
        return self.model.encode(text).tolist()

    def embed_many(self, texts: list):
        """Embed multiple texts in batch for efficiency."""
        if not texts:
            return []
        # Filter out empty texts
        valid_texts = [t for t in texts if t]
        if not valid_texts:
            return [[] for _ in texts]

        # Batch encode
        embeddings = self.model.encode(valid_texts).tolist()

        # Map back to original indices (empty texts get empty embeddings)
        result = []
        idx = 0
        for text in texts:
            if text:
                result.append(embeddings[idx])
                idx += 1
            else:
                result.append([])
        return result

    def __repr__(self) -> str:
        return f"Embedder(model={settings.EMBEDDING_MODEL})"
