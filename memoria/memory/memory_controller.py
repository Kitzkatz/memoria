from core.logger import debug
from core.bootstrap import bootstrap
from system.memory_system import MemorySystem
from memory.goal_tracker import GoalTracker
from graph.entity_store import EntityStore
from cache.config import settings


class MemoryController:

    def __init__(self):
        db, vs, embedder, llm = bootstrap()
        entity_store = EntityStore(db)
        self.system = MemorySystem(db, vs, embedder, entity_store, llm=llm)
        self.goals = GoalTracker(db)
        self.llm = llm

    def remember(self, text: str, metadata: dict = None):
        """
        Store a memory.

        Args:
            text: Memory text
            metadata: Optional metadata dict (session_id, etc.)
        """
        if metadata:
            return self.system.store(text, metadata=metadata)
        return self.system.store(text)

    def remember_many(self, texts, metadatas=None, skip_embedding_build=False):
        """
        Store multiple memories.

        Args:
            texts: List of memory texts
            metadatas: Optional list of metadata dicts (one per text)
            skip_embedding_build: If True, skip embedding computation and vector store ops
        """
        if metadatas:
            return self.system.store_many(texts, metadatas=metadatas, skip_embedding_build=skip_embedding_build)
        return self.system.store_many(texts, metadatas=None, skip_embedding_build=skip_embedding_build)

    def set_goal(self, goal, progress="started"):
        return self.goals.set_goal(goal, progress)

    def update_goal(self, goal_id, progress=None, status=None):
        return self.goals.update_goal(goal_id, progress=progress, status=status)

    def list_goals(self, status=None):
        return self.goals.list_goals(status=status)

    def recall(self, query: str):
        return self.system.query(query)

    def reflect(self):
        return self.system.reflect()

    def chat(self, prompt: str, top_n=None):
        response = self.recall(prompt)
        top_n = top_n or getattr(settings, "TOP_N", 5)
        memories = response["results"][:top_n]
        context = "\n".join(m["text"] for m in memories if m and m.get("text"))

        if not context:
            context = "No relevant memories found."

        # Llama 3 chat template
        full_prompt = (
            "<|start_header_id|>system<|end_header_id|>\n"
            "You are a helpful assistant. Answer concisely based on the context provided.\n"
            "<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"Context:\n{context}\n\n"
            f"User:\n{prompt}\n"
            "<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
        

        reply = self.llm.chat(full_prompt)

        print("CONTROLLER:", repr(reply))

        return reply
