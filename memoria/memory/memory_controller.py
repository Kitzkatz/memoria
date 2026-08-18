from core.logger import debug
from core.bootstrap import bootstrap
from system.memory_system import MemorySystem
from memory.goal_tracker import GoalTracker
from graph.entity_store import EntityStore
from cache.config import settings
from pathlib import Path


class MemoryController:

    def __init__(self):
        db, vs, embedder, llm = bootstrap()
        entity_store = EntityStore(db)
        self.system = MemorySystem(db, vs, embedder, entity_store, llm=llm)
        self.goals = GoalTracker(db)
        self.llm = llm

    def remember(self, text: str, metadata: dict = None):
        if metadata:
            return self.system.store(text, metadata=metadata)
        return self.system.store(text)

    def remember_many(self, texts, metadatas=None, skip_embedding_build=False):
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

    def chat(self, prompt: str, top_n=None, template=None, template_vars=None):
        """
        Chat with memory‑augmented LLM using a customizable template.

        Args:
            prompt: User's message.
            top_n: Number of memories to retrieve (default from settings.TOP_N).
            template: Optional template string or path to template file.
                      If None, uses default from settings.CHAT_TEMPLATE_FILE.
            template_vars: Optional dict of additional variables for the template.
                           Default includes: system, context, user, assistant.
        """
        # 1. Retrieve memories
        response = self.recall(prompt)
        top_n = top_n or getattr(settings, "TOP_N", 5)
        memories = response.get("results", [])[:top_n]
        context = "\n".join(m["text"] for m in memories if m and m.get("text"))
        if not context:
            context = "No relevant memories found."

        # 2. Load template
        if template is None:
            template = self._load_default_template()
        elif Path(template).exists():
            template = self._load_template_from_file(template)

        # 3. Prepare template variables
        default_vars = {
            "system": "You are a helpful assistant. Answer concisely based on the context provided.",
            "context": context,
            "user": prompt,
            "assistant": "",  # optional placeholder for assistant response
        }
        if template_vars:
            default_vars.update(template_vars)

        # 4. Format the template
        try:
            full_prompt = template.format(**default_vars)
        except KeyError as e:
            debug(f"[Controller] Missing template variable: {e}")
            # fallback to simple concatenation
            full_prompt = f"{default_vars['system']}\n\nContext:\n{context}\n\nUser:\n{prompt}\n\nAssistant:\n"

        # 5. Send to LLM
        reply = self.llm.chat(full_prompt)
        print("CONTROLLER:", repr(reply))
        return reply

    def _load_default_template(self):
        """Load the default template from config path."""
        template_dir = getattr(settings, "CHAT_TEMPLATE_DIR", "chat_templates")
        template_file = getattr(settings, "CHAT_TEMPLATE_FILE", "llama3.txt")
        path = Path(template_dir) / template_file
        if path.exists():
            return self._load_template_from_file(path)
        # Fallback to built‑in Llama 3 template
        return self._get_builtin_llama3_template()

    def _load_template_from_file(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            debug(f"[Controller] Failed to load template from {filepath}: {e}")
            return self._get_builtin_llama3_template()

    def _get_builtin_llama3_template(self):
        """Default Llama 3 chat template."""
        return (
            "<|start_header_id|>system<|end_header_id|>\n"
            "{system}\n"
            "<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            "Context:\n{context}\n\n"
            "User:\n{user}\n"
            "<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
            "{assistant}"
        )
