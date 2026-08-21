"""
shared/memory_interface.py

Unified interface used by:

- Benchmark runner
- Synthetic data generator
- CLI / TUI
- FastAPI (optional)

Keeps every subsystem talking to the same contract instead
of directly to MemoryController.
"""

from core.logger import debug, info
from memory.memory_controller import MemoryController


class MemoryInterface:

    def __init__(self):
        self.controller = MemoryController()
        info("[MemoryInterface] Initialized", category="interface")

    @property
    def plugin_manager(self):
        """Expose the plugin manager from the controller."""
        return getattr(self.controller, 'plugin_manager', None)

    # --------------------------------------------------
    # SINGLE MEMORY
    # --------------------------------------------------

    def remember(self, text: str) -> int:
        """
        Store one memory.

        Returns
        -------
        int
            Memory ID.
        """
        debug(f"[MemoryInterface] remember: {text[:50]}...", category="interface")
        # Pre-store hook could be added here, but MemorySystem already has it.
        return self.controller.remember(text)

    # --------------------------------------------------
    # BATCH STORE
    # --------------------------------------------------

    def remember_many(self, texts: list, metadatas: list = None, skip_embedding_build: bool = False) -> list:
        """
        Store many memories.

        Parameters
        ----------
        texts : iterable[str]
            List of memory texts.
        metadatas : list[dict], optional
            List of metadata dicts corresponding to each text.
        skip_embedding_build : bool, default False
            If True, skip embedding computation and vector store operations
            (assumes FAISS index is already loaded from cache).

        Returns
        -------
        list[int]
            IDs of stored memories.
        """
        debug(f"[MemoryInterface] remember_many: {len(texts)} texts", category="interface")
        if metadatas is not None:
            return self.controller.remember_many(texts, metadatas=metadatas, skip_embedding_build=skip_embedding_build)
        return self.controller.remember_many(texts, metadatas=None, skip_embedding_build=skip_embedding_build)

    def store_many(self, texts: list, metadatas: list = None, skip_embedding_build: bool = False) -> list:
        """Alias for remember_many() with same parameters."""
        return self.remember_many(texts, metadatas=metadatas, skip_embedding_build=skip_embedding_build)

    # --------------------------------------------------
    # SINGLE QUERY
    # --------------------------------------------------

    def recall(self, query: str) -> dict:
        """
        Query memory.

        Returns ranking output directly.
        """
        debug(f"[MemoryInterface] recall: {query[:50]}...", category="interface")
        return self.controller.recall(query)

    # --------------------------------------------------
    # BATCH QUERY
    # --------------------------------------------------

    def recall_many(self, queries: list) -> list:
        """
        Execute multiple queries.

        Returns
        -------
        list
            One recall result per query.
        """
        debug(f"[MemoryInterface] recall_many: {len(queries)} queries", category="interface")
        return [self.controller.recall(q) for q in queries]

    # --------------------------------------------------
    # GOALS
    # --------------------------------------------------

    def set_goal(self, goal: str, progress: str = "started") -> int:
        """
        Set a new goal.

        Parameters
        ----------
        goal : str
            Goal description.
        progress : str, optional
            Initial progress status.

        Returns
        -------
        int
            Goal ID.
        """
        debug(f"[MemoryInterface] set_goal: {goal[:50]}...", category="interface")
        return self.controller.set_goal(goal, progress)

    def update_goal(self, goal_id: int, progress: str = None, status: str = None):
        """
        Update an existing goal.

        Parameters
        ----------
        goal_id : int
            Goal ID.
        progress : str, optional
            New progress.
        status : str, optional
            New status.

        Returns
        -------
        None
        """
        debug(f"[MemoryInterface] update_goal: {goal_id}", category="interface")
        return self.controller.update_goal(goal_id, progress, status)

    def list_goals(self, status: str = None) -> list:
        """
        List all goals, optionally filtered by status.

        Parameters
        ----------
        status : str, optional
            Filter by status (e.g., "active", "completed").

        Returns
        -------
        list[dict]
            List of goal records.
        """
        return self.controller.list_goals(status)

    # --------------------------------------------------
    # CHAT (LLM)
    # --------------------------------------------------

    def raw_chat(self, prompt: str) -> str:
        """
        Send a raw prompt directly to the LLM (no retrieval).

        Parameters
        ----------
        prompt : str
            User prompt.

        Returns
        -------
        str
            LLM response.
        """
        debug(f"[MemoryInterface] raw_chat: {prompt[:50]}...", category="interface")
        return self.controller.raw_chat(prompt)

    def chat(self, prompt: str) -> str:
        """
        Chat with the memory system – recalls relevant memories
        and generates a response using the configured LLM.

        Parameters
        ----------
        prompt : str
            User prompt.

        Returns
        -------
        str
            LLM response.
        """
        debug(f"[MemoryInterface] chat: {prompt[:50]}...", category="interface")
        return self.controller.chat(prompt)

    # --------------------------------------------------
    # REFLECTION (V4 feature stub)
    # --------------------------------------------------

    def reflect(self) -> dict:
        """
        Reflection: analyze the current state of the memory system.
        V4 feature stub.
        """
        debug("[MemoryInterface] reflect called", category="interface")
        return self.controller.reflect()

    # --------------------------------------------------
    # DIAGNOSTICS
    # --------------------------------------------------

    def stats(self) -> dict:
        """Get system statistics."""
        return self.controller.stats()

    def __repr__(self) -> str:
        return f"MemoryInterface(controller={self.controller})"

    def __str__(self) -> str:
        return f"MemoryInterface (connected to {self.controller})"


    def ingest_code(self, directory: str, max_files: int = 1000):
        return self.controller.system.ingest_code(directory, max_files)

    def ingest_pdf(self, filepath: str, max_pages: int = 100):
        return self.controller.system.ingest_pdf(filepath, max_pages)
