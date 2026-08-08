"""
Query history tracking for personalization and context.
"""

import json
import time
import threading
from typing import List, Dict, Optional

from core.logger import debug


class QueryHistory:
    def __init__(self, max_history: int = 1000, persist_path: str = "query_history.json"):
        self.max_history = max_history
        self.persist_path = persist_path
        self.history: List[Dict] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Debounce saving
        self._dirty = False
        self._last_save = time.time()
        self._save_threshold = 5.0
        
        self._load()

    def record(self, query: str, results: List[Dict]):
        """
        Record a query and its results.
        """
        with self._lock:
            entry = {
                "query": query,
                "timestamp": time.time(),
                "results": [
                    {"id": r["id"], "rank": r.get("rank", 0), "text": r.get("text", "")[:200]}
                    for r in results[:10]
                ]
            }
            self.history.append(entry)
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]
            self._dirty = True
            self._maybe_save()

    def get_recent(self, n: int = 10) -> List[Dict]:
        """Return the n most recent queries."""
        with self._lock:
            return self.history[-n:]

    def get_frequent(self, n: int = 10) -> List[str]:
        """Return the most frequent queries."""
        from collections import Counter
        with self._lock:
            queries = [h["query"] for h in self.history]
            return [q for q, _ in Counter(queries).most_common(n)]

    def get_context(self) -> List[str]:
        """Return the last few queries for context."""
        recent = self.get_recent(5)
        return [h["query"] for h in recent]

    def get_previous_query(self) -> Optional[str]:
        """Return the immediately preceding query."""
        with self._lock:
            if len(self.history) >= 2:
                return self.history[-2]["query"]
            return None

    # -------------------------
    # Persistence
    # -------------------------

    def _maybe_save(self):
        """Debounced save - only writes every few seconds."""
        now = time.time()
        if now - self._last_save >= self._save_threshold and self._dirty:
            self._save()
            self._dirty = False
            self._last_save = now

    def _save(self):
        try:
            with self._lock:
                # Keep only max_history entries before saving
                if len(self.history) > self.max_history:
                    self.history = self.history[-self.max_history:]
                
                with open(self.persist_path, "w") as f:
                    json.dump(self.history, f, indent=2, default=str)
        except Exception as e:
            debug(f"[QueryHistory] Save error: {e}")

    def _load(self):
        try:
            with open(self.persist_path, "r") as f:
                self.history = json.load(f)
        except FileNotFoundError:
            pass
        except Exception as e:
            debug(f"[QueryHistory] Load error: {e}")
