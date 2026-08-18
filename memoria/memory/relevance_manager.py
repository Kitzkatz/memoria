"""
Relevance Manager — Tracks query frequency, recency, and user feedback
to boost relevant memories.
"""

import time
import json
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from core.logger import debug


class RelevanceManager:
    def __init__(self, db, persist_path: str = "relevance_data.json"):
        self.db = db
        self.persist_path = persist_path

        self.query_counts = defaultdict(int)
        self.last_accessed = {}
        self.user_feedback = {}

        # Thread safety
        self._lock = threading.RLock()

        # Debounced saving
        self._dirty = False
        self._last_save = time.time()
        self._save_threshold = 5.0

        self._load()

    # -------------------------
    # Record operations
    # -------------------------

    def record_query(self, mem_id: int):
        """Record that a memory was queried."""
        with self._lock:
            self.query_counts[mem_id] += 1
            self.last_accessed[mem_id] = time.time()
            self._dirty = True
            self._maybe_save()

    def record_feedback(self, mem_id: int, score: int):
        """
        Record user feedback for a memory.

        Args:
            mem_id: Memory ID
            score: Positive or negative integer (e.g., +1 for good, -1 for bad)
        """
        with self._lock:
            current = self.user_feedback.get(mem_id, 0)
            self.user_feedback[mem_id] = current + score
            self._dirty = True
            self._maybe_save()

    def record_click(self, mem_id: int):
        """Convenience method: record a positive click."""
        self.record_feedback(mem_id, 1)

    def record_skip(self, mem_id: int):
        """Convenience method: record a negative skip."""
        self.record_feedback(mem_id, -1)

    # -------------------------
    # Score calculation
    # -------------------------

    def get_relevance_score(self, mem_id: int) -> float:
        """
        Compute relevance score for a memory.

        Formula:
        - Query count (frequency): 0.5 weight
        - Recency: 0.3 weight (decays over 7 days)
        - User feedback: 0.2 weight

        Returns:
            Float score between 0 and ~10+
        """
        with self._lock:
            count = self.query_counts.get(mem_id, 0)
            recency = self.last_accessed.get(mem_id, 0)
            feedback = self.user_feedback.get(mem_id, 0)

            # Recency score: decays from 1.0 to near 0 over ~30 days
            age = time.time() - recency
            recency_score = 1.0 / (1.0 + age / (30 * 24 * 3600))

            return (count * 0.5) + (recency_score * 0.3) + (feedback * 0.2)

    def get_relevance_scores_batch(self, mem_ids: List[int]) -> Dict[int, float]:
        """Get relevance scores for multiple memories in batch."""
        with self._lock:
            return {mem_id: self.get_relevance_score(mem_id) for mem_id in mem_ids}

    # -------------------------
    # Retrieval
    # -------------------------

    def get_top_relevant(self, limit: int = 50) -> List[int]:
        """Return top N memory IDs by relevance score."""
        with self._lock:
            scores = [
                (mem_id, self.get_relevance_score(mem_id))
                for mem_id in self.query_counts.keys()
            ]
            scores.sort(key=lambda x: x[1], reverse=True)
            return [mem_id for mem_id, _ in scores[:limit]]

    def get_top_relevant_by_type(self, mem_type: str, limit: int = 50) -> List[int]:
        """
        Return top N memory IDs of a specific type by relevance score.
        """
        with self._lock:
            # Fetch more than needed to account for filtering
            rows = self.db.fetch_many_by_type(mem_type, limit=limit * 2)

            scores = []
            for row in rows:
                mem_id = row["id"]
                score = self.get_relevance_score(mem_id)
                scores.append((mem_id, score))

            scores.sort(key=lambda x: x[1], reverse=True)
            return [mem_id for mem_id, _ in scores[:limit]]

    def get_memories_with_scores(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Return memories with their relevance scores.
        Useful for debugging and UI display.
        """
        with self._lock:
            mem_ids = list(self.query_counts.keys())
            scores = self.get_relevance_scores_batch(mem_ids)
            sorted_mem_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:limit]

            result = []
            for mem_id in sorted_mem_ids:
                row = self.db.fetch(mem_id)
                if row:
                    result.append({
                        "id": mem_id,
                        "text": row.get("text", ""),
                        "relevance_score": scores[mem_id],
                        "query_count": self.query_counts.get(mem_id, 0),
                        "user_feedback": self.user_feedback.get(mem_id, 0),
                    })
            return result

    # -------------------------
    # Maintenance
    # -------------------------

    def prune_old_entries(self, max_age_days: int = 180):
        """
        Remove entries that haven't been accessed in a long time.
        Prevents unbounded growth.
        """
        with self._lock:
            cutoff = time.time() - (max_age_days * 24 * 3600)
            to_remove = [
                mem_id for mem_id, last_ts in self.last_accessed.items()
                if last_ts < cutoff
            ]

            for mem_id in to_remove:
                self.query_counts.pop(mem_id, None)
                self.last_accessed.pop(mem_id, None)
                self.user_feedback.pop(mem_id, None)

            if to_remove:
                self._dirty = True
                self._maybe_save()
                debug(f"[RelevanceManager] Pruned {len(to_remove)} old entries", category="memory")

    # -------------------------
    # Persistence
    # -------------------------

    def _maybe_save(self):
        """Debounced save."""
        now = time.time()
        if now - self._last_save >= self._save_threshold and self._dirty:
            self._save()
            self._dirty = False
            self._last_save = now

    def _save(self):
        """Save relevance data to disk."""
        try:
            with self._lock:
                data = {
                    "query_counts": dict(self.query_counts),
                    "last_accessed": self.last_accessed,
                    "user_feedback": self.user_feedback,
                }
                with open(self.persist_path, "w") as f:
                    json.dump(data, f, indent=2, default=str)

                debug(f"[RelevanceManager] Saved {len(self.query_counts)} entries", category="memory")

        except Exception as e:
            debug(f"[RelevanceManager] Save error: {e}", category="memory")

    def _load(self):
        """Load relevance data from disk."""
        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)

            with self._lock:
                self.query_counts = defaultdict(int, data.get("query_counts", {}))
                self.last_accessed = data.get("last_accessed", {})
                self.user_feedback = data.get("user_feedback", {})

            debug(f"[RelevanceManager] Loaded {len(self.query_counts)} entries", category="memory")

        except FileNotFoundError:
            pass
        except Exception as e:
            debug(f"[RelevanceManager] Load error: {e}", category="memory")

    def save_now(self):
        """Force an immediate save."""
        with self._lock:
            self._save()
            self._dirty = False
            self._last_save = time.time()

    # -------------------------
    # Statistics
    # -------------------------

    def stats(self) -> dict:
        """Return relevance manager statistics."""
        with self._lock:
            return {
                "total_memories": len(self.query_counts),
                "memories_with_feedback": len(self.user_feedback),
                "total_feedback": sum(self.user_feedback.values()),
                "dirty": self._dirty,
                "persist_path": self.persist_path,
            }

    # -------------------------
    # Magic Methods
    # -------------------------

    def __len__(self) -> int:
        return len(self.query_counts)

    def __contains__(self, mem_id: int) -> bool:
        return mem_id in self.query_counts
