# memory/feedback.py
"""
Automatic feedback loop — tracks user behavior and adjusts memory relevance.
No manual input required.
"""

import json
import time
import threading
from collections import defaultdict
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class FeedbackLoop:
    def __init__(self, db, persist_path: str = "feedback_data.json"):
        self.db = db
        self.persist_path = persist_path
        
        # Internal lock for thread safety
        self._lock = threading.RLock()
        
        # Memory feedback scores: mem_id -> cumulative score
        self.memory_feedback: Dict[int, float] = defaultdict(float)
        
        # Query history: query -> list of (mem_id, action, timestamp)
        self.query_history: Dict[str, List[Dict]] = defaultdict(list)
        
        # Recent clicks for session context
        self.session_clicks: List[int] = []
        
        # Internal: track when scores were last updated for decay
        self._last_update: Dict[int, float] = {}
        
        # Internal: debounce saving
        self._dirty = False
        self._last_save = time.time()
        self._save_threshold = 5.0  # seconds between saves
        
        self._load()

    # -------------------------
    # Record user behavior
    # -------------------------

    def record_click(self, mem_id: int, query: str):
        """
        User clicked on a memory — positive signal.
        """
        with self._lock:
            self.memory_feedback[mem_id] += 0.5
            self._last_update[mem_id] = time.time()
            self.query_history[query].append({
                "mem_id": mem_id,
                "action": "click",
                "timestamp": time.time()
            })
            self.session_clicks.append(mem_id)
            # Prune session clicks to prevent bloat
            if len(self.session_clicks) > 1000:
                self.session_clicks = self.session_clicks[-500:]
            self._dirty = True
            self._maybe_save()

    def record_skip(self, mem_id: int, query: str):
        """
        User skipped a memory (scrolled past) — negative signal.
        """
        with self._lock:
            self.memory_feedback[mem_id] -= 0.2
            self._last_update[mem_id] = time.time()
            self.query_history[query].append({
                "mem_id": mem_id,
                "action": "skip",
                "timestamp": time.time()
            })
            # Prune query history to prevent memory bloat
            if len(self.query_history[query]) > 50:
                self.query_history[query] = self.query_history[query][-30:]
            self._dirty = True
            self._maybe_save()

    def record_dwell(self, mem_id: int, query: str, duration_ms: float):
        """
        User spent time reading a memory — strong positive signal.
        Duration > 5 seconds is meaningful.
        """
        with self._lock:
            # Use log scaling instead of linear cap
            import math
            if duration_ms > 100:  # Ignore accidental dwells
                boost = min(2.0, math.log(duration_ms / 1000 + 1) * 0.5)
            else:
                boost = 0.0
            self.memory_feedback[mem_id] += boost
            self._last_update[mem_id] = time.time()
            self.query_history[query].append({
                "mem_id": mem_id,
                "action": "dwell",
                "duration_ms": duration_ms,
                "timestamp": time.time()
            })
            self._dirty = True
            self._maybe_save()

    def record_follow_up(self, query: str, previous_query: str):
        """
        User asked a follow-up query — the previous result was relevant.
        """
        with self._lock:
            if previous_query in self.query_history:
                # Get recent entries, weight by recency
                recent = self.query_history[previous_query][-10:]
                for idx, entry in enumerate(recent):
                    if entry["action"] in ("click", "dwell"):
                        # More recent = more weight
                        recency_weight = 1.0 - (idx / len(recent)) * 0.5
                        self.memory_feedback[entry["mem_id"]] += 0.3 * recency_weight
                        self._last_update[entry["mem_id"]] = time.time()
            self._dirty = True
            self._maybe_save()

    # -------------------------
    # Query feedback signals
    # -------------------------

    def get_boost(self, mem_id: int) -> float:
        """Return feedback score for a memory (clamped between -1 and 1)."""
        with self._lock:
            # Apply time decay if we have last_update info
            base_score = self.memory_feedback.get(mem_id, 0.0)
            if mem_id in self._last_update:
                days = (time.time() - self._last_update[mem_id]) / 86400.0
                decay = 0.95 ** days  # 5% per day decay
                base_score *= decay
            return max(-1.0, min(1.0, base_score))

    def get_session_boost(self, mem_id: int) -> float:
        """Boost memories that were clicked in the current session."""
        with self._lock:
            if mem_id in self.session_clicks:
                # Recency matters - last 5 clicks get more boost
                if len(self.session_clicks) >= 5 and mem_id in self.session_clicks[-5:]:
                    return 0.3
                return 0.2
            return 0.0

    def get_top_boosted(self, limit: int = 50) -> List[int]:
        """Return memory IDs with highest feedback scores."""
        with self._lock:
            # Apply decay to all scores before sorting
            now = time.time()
            scored = []
            for mem_id, score in self.memory_feedback.items():
                if mem_id in self._last_update:
                    days = (now - self._last_update[mem_id]) / 86400.0
                    decay = 0.95 ** days
                    score *= decay
                scored.append((mem_id, score))
            sorted_scores = sorted(scored, key=lambda x: x[1], reverse=True)
            return [mem_id for mem_id, _ in sorted_scores[:limit]]

    def clear_session(self):
        """Clear session context (call between user sessions)."""
        with self._lock:
            self.session_clicks = []
            # Don't save immediately - let it batch

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
                # Clean up old history before saving
                for query in list(self.query_history.keys()):
                    if len(self.query_history[query]) > 100:
                        self.query_history[query] = self.query_history[query][-50:]
                
                data = {
                    "memory_feedback": dict(self.memory_feedback),
                    "query_history": {
                        q: entries for q, entries in self.query_history.items()
                    },
                    "last_update": self._last_update
                }
                # Write atomically
                import tempfile
                import os
                fd, temp_path = tempfile.mkstemp(
                    dir=os.path.dirname(self.persist_path) or '.',
                    prefix='feedback_tmp_'
                )
                try:
                    with os.fdopen(fd, 'w') as f:
                        json.dump(data, f, indent=2, default=str)
                    os.replace(temp_path, self.persist_path)
                except Exception:
                    os.unlink(temp_path)
                    raise
        except Exception as e:
            # At least log to stderr with context
            import sys
            print(f"[FeedbackLoop] Save error: {e}", file=sys.stderr)

    def _load(self):
        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)
            with self._lock:
                self.memory_feedback = defaultdict(float, data.get("memory_feedback", {}))
                self.query_history = defaultdict(list, data.get("query_history", {}))
                self._last_update = data.get("last_update", {})
        except FileNotFoundError:
            pass
        except Exception as e:
            import sys
            print(f"[FeedbackLoop] Load error: {e}", file=sys.stderr)
