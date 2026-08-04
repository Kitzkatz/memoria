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

from memory.memory_controller import MemoryController


class MemoryInterface:

    def __init__(self):
        self.controller = MemoryController()

    # --------------------------------------------------
    # SINGLE MEMORY
    # --------------------------------------------------

    def remember(self, text):
        """
        Store one memory.

        Returns
        -------
        int
            Memory ID.
        """
        return self.controller.remember(text)

    # --------------------------------------------------
    # BATCH STORE
    # --------------------------------------------------

    def remember_many(self, texts):
        """
        Store many memories.

        Parameters
        ----------
        texts : iterable[str]

        Returns
        -------
        list[int]
            IDs of stored memories.
        """
        return self.controller.remember_many(texts)

    def store_many(self, texts):
        """Alias for remember_many()."""
        return self.remember_many(texts)

    # --------------------------------------------------
    # SINGLE QUERY
    # --------------------------------------------------

    def recall(self, query):
        """
        Query memory.

        Returns ranking output directly.
        """
        return self.controller.recall(query)

    # --------------------------------------------------
    # BATCH QUERY
    # --------------------------------------------------

    def recall_many(self, queries):
        """
        Execute multiple queries.

        Returns
        -------
        list
            One recall result per query.
        """
        return [self.controller.recall(q) for q in queries]

    # --------------------------------------------------
    # GOALS
    # --------------------------------------------------

    def set_goal(self, goal, progress="started"):
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
        return self.controller.set_goal(goal, progress)

    def update_goal(self, goal_id, progress=None, status=None):
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
        return self.controller.update_goal(goal_id, progress, status)

    def list_goals(self, status=None):
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

    def raw_chat(self, prompt: str):
        return self.controller.llm.chat(prompt)


    
    def chat(self, prompt):
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
        return self.controller.chat(prompt)

    # --------------------------------------------------
    # REFLECTION (placeholder for V4)
    # --------------------------------------------------

    def reflect(self):
        """Placeholder for reflection (V4)."""
        return self.controller.reflect()
