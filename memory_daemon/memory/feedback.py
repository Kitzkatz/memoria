# memory/feedback.py
"""
Automatic feedback loop — tracks user behavior and adjusts memory relevance.
No manual input required.
"""

import json
import time
from collections import defaultdict
from typing import Dict, List, Optional


class FeedbackLoop:
    def __init__(self, db, persist_path: str = "feedback_data.json"):
        self.db = db
        self.persist_path = persist_path

        # Memory feedback scores: mem_id -> cumulative score
        self.memory_feedback: Dict[int, float] = defaultdict(float)

        # Query history: query -> list of (mem_id, action, timestamp)
        self.query_history: Dict[str, List[Dict]] = defaultdict(list)

        # Recent clicks for session context
        self.session_clicks: List[int] = []

        self._load()

    # -------------------------
    # Record user behavior
    # -------------------------

    def record_click(self, mem_id: int, query: str):
        """
        User clicked on a memory — positive signal.
        """
        self.memory_feedback[mem_id] += 0.5
        self.query_history[query].append({
            "mem_id": mem_id,
            "action": "click",
            "timestamp": time.time()
        })
        self.session_clicks.append(mem_id)
        self._save()

    def record_skip(self, mem_id: int, query: str):
        """
        User skipped a memory (scrolled past) — negative signal.
        """
        self.memory_feedback[mem_id] -= 0.2
        self.query_history[query].append({
            "mem_id": mem_id,
            "action": "skip",
            "timestamp": time.time()
        })
        self._save()

    def record_dwell(self, mem_id: int, query: str, duration_ms: float):
        """
        User spent time reading a memory — strong positive signal.
        Duration > 5 seconds is meaningful.
        """
        boost = min(1.0, duration_ms / 5000.0)  # 0-1 based on dwell time
        self.memory_feedback[mem_id] += boost
        self.query_history[query].append({
            "mem_id": mem_id,
            "action": "dwell",
            "duration_ms": duration_ms,
            "timestamp": time.time()
        })
        self._save()

    def record_follow_up(self, query: str, previous_query: str):
        """
        User asked a follow-up query — the previous result was relevant.
        """
        # Boost the memories from the previous query
        if previous_query in self.query_history:
            for entry in self.query_history[previous_query][-5:]:
                if entry["action"] in ("click", "dwell"):
                    self.memory_feedback[entry["mem_id"]] += 0.3
        self._save()

    # -------------------------
    # Query feedback signals
    # -------------------------

    def get_boost(self, mem_id: int) -> float:
        """Return feedback score for a memory (clamped between -1 and 1)."""
        score = self.memory_feedback.get(mem_id, 0.0)
        return max(-1.0, min(1.0, score))

    def get_session_boost(self, mem_id: int) -> float:
        """Boost memories that were clicked in the current session."""
        if mem_id in self.session_clicks:
            return 0.2
        return 0.0

    def get_top_boosted(self, limit: int = 50) -> List[int]:
        """Return memory IDs with highest feedback scores."""
        sorted_scores = sorted(
            self.memory_feedback.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [mem_id for mem_id, _ in sorted_scores[:limit]]

    def clear_session(self):
        """Clear session context (call between user sessions)."""
        self.session_clicks = []

    # -------------------------
    # Persistence
    # -------------------------

    def _save(self):
        try:
            data = {
                "memory_feedback": dict(self.memory_feedback),
                "query_history": {
                    q: entries for q, entries in self.query_history.items()
                }
            }
            with open(self.persist_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"[FeedbackLoop] Save error: {e}")

    def _load(self):
        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)
            self.memory_feedback = defaultdict(float, data.get("memory_feedback", {}))
            self.query_history = defaultdict(list, data.get("query_history", {}))
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[FeedbackLoop] Load error: {e}")
