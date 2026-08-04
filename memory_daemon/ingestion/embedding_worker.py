from sentence_transformers import SentenceTransformer
from cache.config import settings

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def embed(self, text: str):
        return self.model.encode(text).tolist()
