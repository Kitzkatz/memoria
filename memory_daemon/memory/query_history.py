# memory/query_history.py
"""
Query history tracking for personalization and context.
"""

import json
import time
from typing import List, Dict, Optional


class QueryHistory:
    def __init__(self, max_history: int = 1000, persist_path: str = "query_history.json"):
        self.max_history = max_history
        self.persist_path = persist_path
        self.history: List[Dict] = []
        self._load()

    def record(self, query: str, results: List[Dict], top_click: Optional[int] = None):
        """
        Record a query and its results.
        """
        entry = {
            "query": query,
            "timestamp": time.time(),
            "results": [
                {"id": r["id"], "rank": r.get("rank", 0), "text": r.get("text", "")[:200]}
                for r in results[:10]
            ],
            "clicked": top_click
        }
        self.history.append(entry)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        self._save()

    def get_recent(self, n: int = 10) -> List[Dict]:
        """Return the n most recent queries."""
        return self.history[-n:]

    def get_frequent(self, n: int = 10) -> List[str]:
        """Return the most frequent queries."""
        from collections import Counter
        queries = [h["query"] for h in self.history]
        return [q for q, _ in Counter(queries).most_common(n)]

    def get_context(self) -> List[str]:
        """Return the last few queries for context."""
        recent = self.get_recent(5)
        return [h["query"] for h in recent]

    def get_previous_query(self) -> Optional[str]:
        """Return the immediately preceding query."""
        if len(self.history) >= 2:
            return self.history[-2]["query"]
        return None

    def _save(self):
        try:
            with open(self.persist_path, "w") as f:
                json.dump(self.history, f, indent=2, default=str)
        except Exception as e:
            print(f"[QueryHistory] Save error: {e}")

    def _load(self):
        try:
            with open(self.persist_path, "r") as f:
                self.history = json.load(f)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[QueryHistory] Load error: {e}")
