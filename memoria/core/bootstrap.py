from db.facade import MemoryDB
from retrieval.vector_store import VectorStore
from ingestion.embedding_worker import Embedder
from core.llm_adapter import LLMAdapter
from core.logger import info, debug
from cache.config import settings

# Module-level info (runs on import)
info("Bootstrapping memory system...")


def bootstrap():
    """
    Initialize and return core memory system components.

    Returns:
        tuple: (MemoryDB, VectorStore, Embedder, LLMAdapter)
    """
    debug("hit bootstrap", category="core")

    db = MemoryDB()
    vs = VectorStore(dim=settings.VECTOR_DIM)
    embedder = Embedder()
    llm = LLMAdapter()

    return db, vs, embedder, llm
