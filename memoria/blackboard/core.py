import uuid
import threading
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class BlackboardEntry:
    type: str                     # required
    content: Dict[str, Any]       # required
    source: str                   # required
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: List[str] = field(default_factory=list)


@dataclass
class Task:
    type: str                     # required
    payload: Dict[str, Any]       # required
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = 0
    status: str = "pending"       # pending, running, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class Blackboard:
    def __init__(self):
        self._lock = threading.RLock()
        self._entries: Dict[str, BlackboardEntry] = {}
        self._tasks: Dict[str, Task] = {}
        self._subscriptions: Dict[str, List[Callable]] = {}
        self._history: List[BlackboardEntry] = []
        self._max_history = 10000  # Prevent unbounded growth

    # -------------------------
    # Entries
    # -------------------------

    def post(self, entry: BlackboardEntry) -> str:
        """Post an entry to the blackboard."""
        with self._lock:
            self._entries[entry.id] = entry
            self._history.append(entry)

            # Prune history if too large
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

            self._trigger_subscriptions(entry.type, entry)
            return entry.id

    def get(self, entry_id: str) -> Optional[BlackboardEntry]:
        with self._lock:
            return self._entries.get(entry_id)

    def get_by_type(self, entry_type: str) -> List[BlackboardEntry]:
        with self._lock:
            return [e for e in self._entries.values() if e.type == entry_type]

    def get_entries(self, entry_type: Optional[str] = None) -> List[BlackboardEntry]:
        """Get entries, optionally filtered by type."""
        if entry_type is not None:
            return self.get_by_type(entry_type)
        with self._lock:
            return list(self._entries.values())

    def remove_entry(self, entry_id: str) -> bool:
        """Remove an entry from the blackboard."""
        with self._lock:
            if entry_id in self._entries:
                del self._entries[entry_id]
                return True
            return False

    def clear_entries(self):
        """Clear all entries from the blackboard."""
        with self._lock:
            self._entries.clear()

    # -------------------------
    # Subscriptions
    # -------------------------

    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to entries of a specific type."""
        with self._lock:
            if event_type not in self._subscriptions:
                self._subscriptions[event_type] = []
            self._subscriptions[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe a callback from an event type."""
        with self._lock:
            if event_type in self._subscriptions:
                self._subscriptions[event_type] = [cb for cb in self._subscriptions[event_type] if cb != callback]

    def _trigger_subscriptions(self, event_type: str, entry: BlackboardEntry):
        for callback in self._subscriptions.get(event_type, []):
            try:
                callback(entry)
            except Exception as e:
                # Don't let subscription failures break the blackboard
                from core.logger import debug
                debug(f"[Blackboard] Subscription error: {e}")

    # -------------------------
    # Tasks
    # -------------------------

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
                    if key == "status" and value == "running":
                        task.started_at = datetime.now(timezone.utc)
                    elif key == "status" and value in ("completed", "failed"):
                        task.completed_at = datetime.now(timezone.utc)
                    setattr(task, key, value)
            return task

    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the blackboard."""
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                return True
            return False

    def get_pending_tasks(self) -> List[Task]:
        with self._lock:
            return [t for t in self._tasks.values() if t.status == "pending"]

    def get_tasks_by_type(self, task_type: str) -> List[Task]:
        with self._lock:
            return [t for t in self._tasks.values() if t.type == task_type]

    def clear_completed_tasks(self) -> int:
        """Remove all completed or failed tasks. Returns count removed."""
        with self._lock:
            to_remove = [tid for tid, t in self._tasks.items() if t.status in ("completed", "failed")]
            for tid in to_remove:
                del self._tasks[tid]
            return len(to_remove)

    def clear_all_tasks(self):
        """Remove all tasks from the blackboard."""
        with self._lock:
            self._tasks.clear()

    # -------------------------
    # History
    # -------------------------

    def get_history(self, n: int = 100) -> List[BlackboardEntry]:
        """Get the last n entries from history."""
        with self._lock:
            return self._history[-n:] if self._history else []

    def clear_history(self):
        """Clear the history."""
        with self._lock:
            self._history.clear()

    # -------------------------
    # Stats
    # -------------------------

    def stats(self) -> Dict[str, int]:
        """Return blackboard statistics."""
        with self._lock:
            return {
                "entries": len(self._entries),
                "tasks": len(self._tasks),
                "history": len(self._history),
                "subscriptions": sum(len(v) for v in self._subscriptions.values()),
            }
