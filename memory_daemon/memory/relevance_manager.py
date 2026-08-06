# memory/relevance_manager.py
import time
import json
from collections import defaultdict
from datetime import datetime, timedelta

class RelevanceManager:
    def __init__(self, db, persist_path="relevance_data.json"):
        self.db = db
        self.persist_path = persist_path
        self.query_counts = defaultdict(int)
        self.last_accessed = {}
        self.user_feedback = {}
        self._load()

    def record_query(self, mem_id):
        self.query_counts[mem_id] += 1
        self.last_accessed[mem_id] = time.time()
        self._save()

    def record_feedback(self, mem_id, score: int):
        self.user_feedback[mem_id] = self.user_feedback.get(mem_id, 0) + score
        self._save()

    def get_relevance_score(self, mem_id) -> float:
        count = self.query_counts.get(mem_id, 0)
        recency = self.last_accessed.get(mem_id, 0)
        feedback = self.user_feedback.get(mem_id, 0)

        age = time.time() - recency
        recency_score = 1.0 / (1.0 + age / (7 * 24 * 3600))

        return (count * 0.5) + (recency_score * 0.3) + (feedback * 0.2)

    def get_top_relevant(self, limit=50) -> list:
        scores = [(mem_id, self.get_relevance_score(mem_id)) for mem_id in self.query_counts.keys()]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [mem_id for mem_id, _ in scores[:limit]]

    def get_top_relevant_by_type(self, mem_type: str, limit=50) -> list:
        # Fetch top N memories of that type from the DB
        rows = self.db.fetch_many_by_type(mem_type, limit=limit * 2)  # fetch more to score
        scores = [(row["id"], self.get_relevance_score(row["id"])) for row in rows]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [mem_id for mem_id, _ in scores[:limit]]

    def promote_to_relevance(self, mem_id):
        # Placeholder: insert into relevance table
        pass

    def _save(self):
        data = {
            "query_counts": dict(self.query_counts),
            "last_accessed": self.last_accessed,
            "user_feedback": self.user_feedback,
        }
        with open(self.persist_path, "w") as f:
            json.dump(data, f)

    def _load(self):
        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)
            self.query_counts = defaultdict(int, data.get("query_counts", {}))
            self.last_accessed = data.get("last_accessed", {})
            self.user_feedback = data.get("user_feedback", {})
        except FileNotFoundError:
            pass
