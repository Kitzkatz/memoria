import threading
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, Any, Optional, Callable, List
from .core import Blackboard, Task

class Scheduler:
    def __init__(self, blackboard: Blackboard, max_workers: int = 4):
        self.blackboard = blackboard
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._task_handlers: Dict[str, Callable] = {}
        self._futures: List[Future] = []
        self._lock = threading.Lock()

    def register_worker(self, task_type: str, handler: Callable):
        """Register a worker function for a specific task type."""
        self._task_handlers[task_type] = handler

    def submit(self, task_type: str, payload: Dict[str, Any], priority: int = 0) -> str:
        """Submit a task and return its ID."""
        task = Task(type=task_type, payload=payload, priority=priority)
        task_id = self.blackboard.submit_task(task)

        # Submit to thread pool
        future = self._executor.submit(self._run_task, task_id)
        with self._lock:
            self._futures.append(future)
        return task_id

    def _run_task(self, task_id: str):
        """Execute a single task."""
        task = self.blackboard.get_task(task_id)
        if not task:
            return
        self.blackboard.update_task(task_id, status="running", started_at=time.time())

        handler = self._task_handlers.get(task.type)
        if handler:
            try:
                result = handler(task.payload)
                self.blackboard.update_task(task_id, status="completed", result=result)
            except Exception as e:
                self.blackboard.update_task(task_id, status="failed", error=str(e))
        else:
            self.blackboard.update_task(task_id, status="failed", error=f"No handler for task type: {task.type}")

    def wait_for_tasks(self, timeout: float = 5.0) -> bool:
        """
        Wait for all submitted tasks to complete.
        Returns True if all tasks finished within timeout, False if timeout reached.
        """
        with self._lock:
            futures = list(self._futures)
            self._futures.clear()

        if not futures:
            return True

        done, not_done = concurrent.futures.wait(futures, timeout=timeout)
        return len(not_done) == 0

    def shutdown(self):
        """Shut down the thread pool."""
        self._executor.shutdown(wait=False)
