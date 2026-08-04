from retrieval.db import MemoryDB
from retrieval.vector_store import VectorStore
from ingestion.embedding_worker import Embedder
from core.llm_adapter import LLMAdapter
from core.logger import info
from cache.config import settings
from core.logger import debug




info("Bootstrapping memory system...")

def bootstrap():
    debug("hit bootstrap")
    db = MemoryDB()
    vs = VectorStore(dim=settings.VECTOR_DIM)
    embedder = Embedder()
    llm = LLMAdapter()
    return db, vs, embedder, llm
