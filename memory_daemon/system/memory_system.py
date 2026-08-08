"""
MemorySystem — Main entry point for the memory system.
Facade that delegates to specialized handlers.
"""

from cache.config import settings
from core.logger import debug

# Import the new internal modules (prefixed with _ to indicate internal)
from system._initializer import initialize_components
from system._query_handler import handle_query
from system._store_handler import handle_store, handle_store_many


class MemorySystem:
    """
    Main entry point for the memory system.
    Thin facade that delegates to specialized handlers.
    All public methods remain unchanged.
    """

    def __init__(self, db, vector_store, embedder, entity_store, llm=None):
        self.db = db
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm = llm

        # Initialize all components
        # This sets self.attribute_map, self.extractor, self.pruner, etc.
        initialize_components(self, db, vector_store, embedder, entity_store, llm)

        # For backward compatibility with any external code checking these
        self.use_blackboard = getattr(settings, "USE_BLACKBOARD", False)

        debug("MemorySystem initialized")

    # -------------------------------------
    # Public API - All signatures unchanged
    # -------------------------------------

    def store(self, text):
        """Store a single memory."""
        return handle_store(self, text)

    def store_many(self, texts):
        """Store multiple memories."""
        return handle_store_many(self, texts)

    def query(self, text):
        """Query the memory system."""
        return handle_query(self, text)

    def ingest_pdf(self, filepath: str, max_pages: int = 100):
        """Ingest a PDF file."""
        return self.pdf_worker.ingest_pdf(filepath, max_pages)

    def ingest_code(self, directory: str, max_files: int = 1000):
        """Ingest a codebase."""
        self.code_worker.ingest_codebase(directory, max_files)

    def register_embedding(self, mem_id, vector):
        """Register an embedding for a memory."""
        debug("\nREGISTER EMBEDDING")
        debug(mem_id)
        debug(len(vector))
        debug(vector[:5])
        self.embedding_cache.add(mem_id, vector)
        self.vector_store.add(mem_id, vector, persist=True)

    def consolidate(self, threshold: float = None):
        """Run consolidation manually."""
        threshold = threshold or getattr(settings, "CONSOLIDATE_THRESHOLD", 0.5)
        self.consolidator.run(threshold)

    def prune(self, dry_run: bool = False) -> dict:
        """Manually run the pruner."""
        return self.pruner.prune_now(dry_run=dry_run)
