import threading
import time
from typing import Dict, Any, Optional, Callable
from .core import Blackboard, Task

class Scheduler:
    def __init__(self, blackboard: Blackboard, poll_interval: float = 0.1):
        self.blackboard = blackboard
        self.poll_interval = poll_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._workers: Dict[str, Callable] = {}
        self._task_handlers: Dict[str, Callable] = {}

    def register_worker(self, task_type: str, handler: Callable):
        """Register a worker function for a specific task type."""
        self._task_handlers[task_type] = handler

    def start(self):
        """Start the scheduler in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self):
        while self._running:
            # Process pending tasks in priority order
            tasks = self.blackboard.get_pending_tasks()
            tasks.sort(key=lambda t: t.priority, reverse=True)

            for task in tasks:
                self._dispatch_task(task)

            time.sleep(self.poll_interval)

    def _dispatch_task(self, task: Task):
        self.blackboard.update_task(task.id, status="running", started_at=time.time())

        handler = self._task_handlers.get(task.type)
        if handler:
            try:
                result = handler(task.payload)
                self.blackboard.update_task(task.id, status="completed", result=result)
            except Exception as e:
                self.blackboard.update_task(task.id, status="failed", error=str(e))
        else:
            self.blackboard.update_task(task.id, status="failed", error=f"No handler for task type: {task.type}")

    def submit(self, task_type: str, payload: Dict[str, Any], priority: int = 0) -> str:
        """Submit a new task to the scheduler."""
        task = Task(type=task_type, payload=payload, priority=priority)
        return self.blackboard.submit_task(task)
