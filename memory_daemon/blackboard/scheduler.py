import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait

from core.logger import debug
from blackboard.core import Task


class Scheduler:

    def __init__(self, blackboard, max_workers=None):
        if max_workers is None:
            import os
            max_workers = min(8, (os.cpu_count() or 4))

        self.blackboard = blackboard
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.handlers = {}
        self.futures = {}
        self.lock = threading.Lock()

    def register_worker(self, task_type, handler):
        self.handlers[task_type] = handler

    def submit(self, task_type, payload, priority=0):
        task = Task(type=task_type, payload=payload, priority=priority)
        task_id = self.blackboard.submit_task(task)
        future = self.executor.submit(self._run_task, task_id)
        with self.lock:
            self.futures[task_id] = future
        return task_id

    def _run_task(self, task_id):
        task = self.blackboard.get_task(task_id)
        if task is None:
            return

        handler = self.handlers.get(task.type)
        if handler is None:
            self.blackboard.update_task(
                task_id,
                status="failed",
                error=f"No handler registered for '{task.type}'"
            )
            return

        self.blackboard.update_task(task_id, status="running")
        start = time.perf_counter()

        try:
            result = handler(task.payload)
            elapsed = (time.perf_counter() - start) * 1000
            debug(f"{task.type}: {elapsed:.2f} ms")
            self.blackboard.update_task(
                task_id,
                status="completed",
                result=result
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            debug(f"{task.type} FAILED ({elapsed:.2f} ms): {e}")
            self.blackboard.update_task(
                task_id,
                status="failed",
                error=str(e)
            )

    def wait_for_tasks(self, task_ids, timeout=1.0):
        with self.lock:
            futures = [self.futures[tid] for tid in task_ids if tid in self.futures]

        if not futures:
            return True

        done, not_done = wait(futures, timeout=timeout)
        return len(not_done) == 0

    def results(self, task_ids, clear_after=True):
        """
        Retrieve results for completed tasks.
        If clear_after=True, removes tasks from blackboard after retrieval.
        """
        results = []
        with self.lock:
            for tid in task_ids:
                task = self.blackboard.get_task(tid)
                if task is None:
                    continue
                if task.status == "completed" and task.result is not None:
                    results.append(task.result)
                # Remove the future from tracking
                self.futures.pop(tid, None)

        # Optionally clear completed tasks from blackboard to free memory
        if clear_after and results:
            for tid in task_ids:
                task = self.blackboard.get_task(tid)
                if task and task.status in ("completed", "failed"):
                    self.blackboard.remove_task(tid)

        return results

    def get_task_status(self, task_id):
        """Return the status of a task for debugging."""
        task = self.blackboard.get_task(task_id)
        return task.status if task else None

    def clear_completed(self):
        """Remove all completed/failed tasks from blackboard to free memory."""
        self.blackboard.clear_completed()

    def shutdown(self, wait=True):
        """Shut down the executor. If wait=True, finish all pending tasks."""
        if wait:
            self.executor.shutdown(wait=True)
        else:
            self.executor.shutdown(wait=False)
