import uuid
import threading
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class BlackboardEntry:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str  # "query", "candidates", "ranked", "result", "error"
    content: Dict[str, Any]
    source: str  # "user", "faiss", "bm25", "graph", "ranker"
    priority: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)


@dataclass
class Task:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str  # "retrieve", "rank", "consolidate", "prune"
    payload: Dict[str, Any]
    priority: int = 0
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class Blackboard:
    def __init__(self):
        self._lock = threading.RLock()
        self._entries: Dict[str, BlackboardEntry] = {}
        self._tasks: Dict[str, Task] = {}
        self._subscriptions: Dict[str, List[Callable]] = {}
        self._history: List[BlackboardEntry] = []

    def post(self, entry: BlackboardEntry) -> str:
        """Post an entry to the blackboard."""
        with self._lock:
            self._entries[entry.id] = entry
            self._history.append(entry)
            self._trigger_subscriptions(entry.type, entry)
            return entry.id

    def get(self, entry_id: str) -> Optional[BlackboardEntry]:
        with self._lock:
            return self._entries.get(entry_id)

    def get_by_type(self, entry_type: str) -> List[BlackboardEntry]:
        with self._lock:
            return [e for e in self._entries.values() if e.type == entry_type]

    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to entries of a specific type."""
        with self._lock:
            if event_type not in self._subscriptions:
                self._subscriptions[event_type] = []
            self._subscriptions[event_type].append(callback)

    def _trigger_subscriptions(self, event_type: str, entry: BlackboardEntry):
        for callback in self._subscriptions.get(event_type, []):
            callback(entry)

    def submit_task(self, task: Task) -> str:
        with self._lock:
            self._tasks[task.id] = task
            return task.id

    def get_task(self, task_id: str) -> Optional[Task]:
        with self._lock:
            return self._tasks.get(task_id)

    def update_task(self, task_id: str, **kwargs):
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                for key, value in kwargs.items():
                    setattr(task, key, value)
            return task

    def get_pending_tasks(self) -> List[Task]:
        with self._lock:
            return [t for t in self._tasks.values() if t.status == "pending"]

    def get_tasks_by_type(self, task_type: str) -> List[Task]:
        with self._lock:
            return [t for t in self._tasks.values() if t.type == task_type]
