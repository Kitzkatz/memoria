import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, Set

from core.logger import debug
from blackboard.core import Task


# ---------------------------------------------------------------------------
# Execution metadata
# ---------------------------------------------------------------------------

@dataclass
class TaskExecution:
    task_id: Any
    task_type: str
    future: Any

    submitted_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    status: str = "submitted"
    result: Any = None
    error: Optional[str] = None

    @property
    def queue_ms(self):
        if self.started_at is None:
            return None
        return (self.started_at - self.submitted_at) * 1000

    @property
    def execution_ms(self):
        if self.started_at is None or self.completed_at is None:
            return None
        return (self.completed_at - self.started_at) * 1000

    @property
    def total_ms(self):
        if self.completed_at is None:
            return None
        return (self.completed_at - self.submitted_at) * 1000


@dataclass
class ExecutionResult:
    task_ids: list
    completed_ids: list = field(default_factory=list)
    failed_ids: list = field(default_factory=list)
    pending_ids: list = field(default_factory=list)
    cancelled_ids: list = field(default_factory=list)

    results: list = field(default_factory=list)

    elapsed_ms: float = 0.0

    policy_name: str = "all"
    finish_reason: str = "unknown"

    task_stats: Dict[Any, Dict[str, Any]] = field(default_factory=dict)

    @property
    def complete(self):
        return not self.pending_ids


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------

class CancellationToken:
    """
    Cooperative cancellation token.

    ThreadPoolExecutor cannot kill a function that is already running.
    Workers can periodically check this token and return early.
    """

    def __init__(self):
        self._event = threading.Event()

    def cancel(self):
        self._event.set()

    @property
    def cancelled(self):
        return self._event.is_set()

    def is_cancelled(self):
        return self._event.is_set()


# ---------------------------------------------------------------------------
# Completion policies
# ---------------------------------------------------------------------------

class CompletionPolicy:
    """
    Base policy.

    Policies receive the current execution state and decide whether the
    scheduler is allowed to finish.
    """

    name = "base"

    def should_finish(self, state):
        raise NotImplementedError


class AllCompletePolicy(CompletionPolicy):
    """
    Do not finish until every task reaches a terminal state.
    """

    name = "all"

    def should_finish(self, state):
        return state["pending_count"] == 0


class QuorumPolicy(CompletionPolicy):
    """
    Finish after N tasks reach a terminal state.

    This is intentionally generic. It does not know whether FAISS,
    BM25, graph, etc. are better or worse sources.
    """

    name = "quorum"

    def __init__(self, count):
        if count < 1:
            raise ValueError("Quorum count must be >= 1")
        self.count = count

    def should_finish(self, state):
        return state["completed_count"] >= self.count


class SourceCoveragePolicy(CompletionPolicy):
    """
    Finish when required source types have completed and the minimum
    number of distinct sources is available.

    The scheduler expects the task payload to optionally contain:
        {"source": "faiss"}

    or the task type itself can be used as the source.
    """

    name = "source_coverage"

    def __init__(
        self,
        required_sources=None,
        min_sources=1,
    ):
        self.required_sources = set(required_sources or [])
        self.min_sources = max(1, min_sources)

    def should_finish(self, state):
        sources = state["completed_sources"]

        if not self.required_sources.issubset(sources):
            return False

        return len(sources) >= self.min_sources


class SufficientPolicy(CompletionPolicy):
    """
    Generic sufficiency policy.

    This is deliberately NOT coupled to MemorySystem yet.

    A callable evaluator receives the current scheduler state and returns:
        True  -> sufficient
        False -> keep waiting

    Later this can evaluate RetrievalState:
        candidate count
        source diversity
        required retrieval modalities
        evidence quality
        etc.
    """

    name = "sufficient"

    def __init__(self, evaluator: Callable[[dict], bool]):
        if not callable(evaluator):
            raise TypeError("SufficientPolicy evaluator must be callable")

        self.evaluator = evaluator

    def should_finish(self, state):
        return bool(self.evaluator(state))


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class Scheduler:

    def __init__(self, blackboard, max_workers=None, plugin_manager=None):
        if max_workers is None:
            max_workers = min(8, (os.cpu_count() or 4))

        self.blackboard = blackboard
        self.plugin_manager = plugin_manager  # <-- NEW
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        self.handlers = {}

        # task_id -> TaskExecution
        self.executions = {}

        # Kept as an alias so existing debugging / external code that
        # references scheduler.futures doesn't immediately break.
        self.futures = {}

        self.lock = threading.Lock()

    # ------------------------------------------------------------------
    # Worker registration
    # ------------------------------------------------------------------

    def register_worker(self, task_type, handler):
        self.handlers[task_type] = handler

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit(self, task_type, payload, priority=0):
        task = Task(
            type=task_type,
            payload=payload,
            priority=priority
        )

        task_id = self.blackboard.submit_task(task)

        submitted_at = time.perf_counter()

        with self.lock:
            execution = TaskExecution(
                task_id=task_id,
                task_type=task_type,
                future=None,
                submitted_at=submitted_at,
            )

            self.executions[task_id] = execution

            future = self.executor.submit(
                self._run_task,
                task_id,
            )

            execution.future = future
            self.futures[task_id] = future

        return task_id

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------

    def _run_task(self, task_id):
        task = self.blackboard.get_task(task_id)

        if task is None:
            with self.lock:
                execution = self.executions.get(task_id)
                if execution:
                    execution.status = "failed"
                    execution.error = "Task disappeared from blackboard"
                    execution.completed_at = time.perf_counter()
            return

        started_at = time.perf_counter()

        with self.lock:
            execution = self.executions.get(task_id)

            if execution:
                execution.started_at = started_at
                execution.status = "running"

        handler = self.handlers.get(task.type)

        if handler is None:
            error = f"No handler registered for '{task.type}'"

            self.blackboard.update_task(
                task_id,
                status="failed",
                error=error,
            )

            self._finish_execution(
                task_id,
                status="failed",
                error=error,
            )
            return

        self.blackboard.update_task(
            task_id,
            status="running",
        )

        try:
            result = handler(task.payload)

            completed_at = time.perf_counter()
            elapsed = (completed_at - started_at) * 1000

            debug(
                f"{task.type}: "
                f"{elapsed:.2f} ms"
            )

            self.blackboard.update_task(
                task_id,
                status="completed",
                result=result,
            )

            self._finish_execution(
                task_id,
                status="completed",
                result=result,
            )

        except Exception as e:
            completed_at = time.perf_counter()
            elapsed = (completed_at - started_at) * 1000

            debug(
                f"{task.type} FAILED "
                f"({elapsed:.2f} ms): {e}"
            )

            self.blackboard.update_task(
                task_id,
                status="failed",
                error=str(e),
            )

            self._finish_execution(
                task_id,
                status="failed",
                error=str(e),
            )

    def _finish_execution(
        self,
        task_id,
        status,
        result=None,
        error=None,
    ):
        completed_at = time.perf_counter()

        with self.lock:
            execution = self.executions.get(task_id)

            if execution is None:
                return

            execution.status = status
            execution.result = result
            execution.error = error
            execution.completed_at = completed_at

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _build_execution_state(self, task_ids):
        """
        Build the state exposed to completion policies.

        This is intentionally scheduler-level state.

        Retrieval-specific candidate state will be layered on top later.
        """

        completed = []
        failed = []
        pending = []
        cancelled = []

        completed_sources = set()

        with self.lock:
            executions = {
                tid: self.executions.get(tid)
                for tid in task_ids
            }

        for tid, execution in executions.items():

            if execution is None:
                continue

            status = execution.status

            if status == "completed":
                completed.append(tid)

                source = self._source_for_execution(execution)

                if source:
                    completed_sources.add(source)

            elif status == "failed":
                failed.append(tid)

            elif status == "cancelled":
                cancelled.append(tid)

            else:
                pending.append(tid)

        completed_results = {}

        with self.lock:
            for tid in completed:
                execution = self.executions.get(tid)
                if execution is not None:
                    completed_results[tid] = execution.result

        return {
            "task_ids": list(task_ids),

            "completed_ids": completed,
            "failed_ids": failed,
            "pending_ids": pending,
            "cancelled_ids": cancelled,

            "completed_count": len(completed),
            "failed_count": len(failed),
            "pending_count": len(pending),
            "cancelled_count": len(cancelled),

            "terminal_count": (
                len(completed)
                + len(failed)
                + len(cancelled)
            ),

            "completed_sources": completed_sources,

            "completed_results": completed_results,
        }

    def _source_for_execution(self, execution):
        """
        Determine the logical source represented by a task.

        For now task_type is the source.

        Later this can inspect Task metadata if necessary.
        """
        return execution.task_type

    # ------------------------------------------------------------------
    # Policy execution
    # ------------------------------------------------------------------

    def execute(
        self,
        task_ids,
        policy=None,
        deadline=None,
        cancel_pending=False,
    ):
        """
        Execute a group of submitted tasks according to a completion policy.

        Parameters
        ----------
        task_ids:
            Tasks belonging to this execution/query.

        policy:
            CompletionPolicy instance.
            Defaults to AllCompletePolicy.

        deadline:
            Maximum number of seconds to wait.
            None means no deadline.

        cancel_pending:
            If True, attempt to cancel pending futures when the policy
            finishes or the deadline expires.

        Returns
        -------
        ExecutionResult
        """

        task_ids = list(task_ids)

        # ---- Plugin hook: pre-scheduler ----
        if self.plugin_manager:
            try:
                self.plugin_manager.memoria_scheduler_pre(task_ids)
            except Exception as e:
                debug(f"[Plugin] scheduler_pre error: {e}")

        if not task_ids:
            return ExecutionResult(
                task_ids=[],
                policy_name=(
                    policy.name
                    if policy
                    else "all"
                ),
                finish_reason="no_tasks",
            )

        if policy is None:
            policy = AllCompletePolicy()

        start = time.perf_counter()

        finish_reason = "unknown"

        # Snapshot futures.
        with self.lock:
            futures = {
                tid: self.futures[tid]
                for tid in task_ids
                if tid in self.futures
            }

        if not futures:
            return ExecutionResult(
                task_ids=task_ids,
                policy_name=policy.name,
                finish_reason="no_tracked_tasks",
            )

        future_to_task = {
            future: tid
            for tid, future in futures.items()
        }

        pending_futures = set(futures.values())

        while pending_futures:

            # Calculate remaining deadline.
            timeout = None

            if deadline is not None:
                elapsed = time.perf_counter() - start
                remaining = deadline - elapsed

                if remaining <= 0:
                    finish_reason = "deadline"
                    break

                timeout = remaining

            done, pending_futures = wait(
                pending_futures,
                timeout=timeout,
                return_when=FIRST_COMPLETED,
            )

            if not done:
                # Deadline expired.
                finish_reason = "deadline"
                break

            # Futures have completed. Their execution records were
            # updated by _run_task.
            state = self._build_execution_state(task_ids)

            if policy.should_finish(state):
                finish_reason = policy.name
                break

        # If we got here because everything finished naturally.
        state = self._build_execution_state(task_ids)

        if state["pending_count"] == 0:
            if finish_reason == "unknown":
                finish_reason = "all_complete"

        # Optional cancellation.
        cancelled_ids = []

        if cancel_pending:
            cancelled_ids = self._cancel_pending(
                task_ids,
            )

        # Final state after cancellation attempts.
        state = self._build_execution_state(task_ids)

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        result_objects = []

        with self.lock:
            for tid in state["completed_ids"]:
                execution = self.executions.get(tid)

                if execution is not None:
                    if execution.result is not None:
                        result_objects.append(
                            execution.result
                        )

        task_stats = {}

        with self.lock:
            for tid in task_ids:
                execution = self.executions.get(tid)

                if execution is None:
                    continue

                task_stats[tid] = {
                    "task_type": execution.task_type,
                    "status": execution.status,
                    "queue_ms": execution.queue_ms,
                    "execution_ms": execution.execution_ms,
                    "total_ms": execution.total_ms,
                    "error": execution.error,
                }

        execution_result = ExecutionResult(
            task_ids=task_ids,
            completed_ids=state["completed_ids"],
            failed_ids=state["failed_ids"],
            pending_ids=state["pending_ids"],
            cancelled_ids=cancelled_ids,
            results=result_objects,
            elapsed_ms=elapsed_ms,
            policy_name=policy.name,
            finish_reason=finish_reason,
            task_stats=task_stats,
        )

        # ---- Plugin hook: post-scheduler ----
        if self.plugin_manager:
            try:
                self.plugin_manager.memoria_scheduler_post(execution_result)
            except Exception as e:
                debug(f"[Plugin] scheduler_post error: {e}")

        return execution_result

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def _cancel_pending(self, task_ids):
        """
        Attempt to cancel tasks that have not started.

        Python cannot forcibly terminate an already-running thread.

        Those require cooperative cancellation support at the worker level.
        """

        cancelled = []

        with self.lock:
            executions = {
                tid: self.executions.get(tid)
                for tid in task_ids
            }

        for tid, execution in executions.items():

            if execution is None:
                continue

            if execution.status in (
                "completed",
                "failed",
                "cancelled",
            ):
                continue

            if execution.future.cancel():

                with self.lock:
                    execution.status = "cancelled"
                    execution.completed_at = time.perf_counter()

                cancelled.append(tid)

                try:
                    self.blackboard.update_task(
                        tid,
                        status="cancelled",
                    )
                except Exception:
                    pass

        return cancelled

    # ------------------------------------------------------------------
    # Backward-compatible waiting
    # ------------------------------------------------------------------

    def wait_for_tasks(
        self,
        task_ids,
        min_wait=None,
        max_wait=None,
        quiet_period=None,
        poll_interval=None,
    ):
        """
        Backward-compatible wrapper.

        IMPORTANT:
        The old quiet-period behavior is intentionally gone.

        If max_wait is supplied, it becomes an explicit deadline.

        Otherwise this waits for ALL tasks.

        The old parameters remain accepted so existing V4 code does not
        immediately explode while we migrate the query path.
        """

        deadline = max_wait

        result = self.execute(
            task_ids,
            policy=AllCompletePolicy(),
            deadline=deadline,
            cancel_pending=False,
        )

        return result.complete

    # ------------------------------------------------------------------
    # Result retrieval
    # ------------------------------------------------------------------

    def results(
        self,
        task_ids,
        clear_after=True,
    ):
        """
        Retrieve completed results.

        Only terminal tasks are cleared.
        """

        results = []
        to_clear = []

        with self.lock:
            for tid in task_ids:

                execution = self.executions.get(tid)

                if execution is None:
                    continue

                if (
                    execution.status == "completed"
                    and execution.result is not None
                ):
                    results.append(
                        execution.result
                    )

                if execution.status in (
                    "completed",
                    "failed",
                    "cancelled",
                ):
                    self.executions.pop(
                        tid,
                        None,
                    )

                    self.futures.pop(
                        tid,
                        None,
                    )

                    to_clear.append(tid)

        if (
            clear_after
            and to_clear
            and hasattr(
                self.blackboard,
                "remove_task",
            )
        ):
            for tid in to_clear:
                self.blackboard.remove_task(tid)

        return results


    def cleanup_tasks(self, task_ids):
        """
        Remove terminal execution records for the supplied tasks.

        Pending/running tasks are intentionally preserved.
        """

        to_clear = []

        with self.lock:
            for tid in task_ids:

                execution = self.executions.get(tid)

                if execution is None:
                    continue

                if execution.status in (
                    "completed",
                    "failed",
                    "cancelled",
                ):
                    self.executions.pop(
                        tid,
                        None,
                    )

                    self.futures.pop(
                        tid,
                        None,
                    )

                    to_clear.append(tid)

        if (
            to_clear
            and hasattr(
                self.blackboard,
                "remove_task",
            )
        ):
            for tid in to_clear:
                self.blackboard.remove_task(tid)

        return to_clear

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def get_task_status(self, task_id):
        """Return scheduler execution status."""

        with self.lock:
            execution = self.executions.get(task_id)

        if execution is not None:
            return execution.status

        task = self.blackboard.get_task(task_id)

        return task.status if task else None

    def get_task_stats(self, task_id):
        """Return timing/debug information for one task."""

        with self.lock:
            execution = self.executions.get(task_id)

            if execution is None:
                return None

            return {
                "task_id": task_id,
                "task_type": execution.task_type,
                "status": execution.status,
                "queue_ms": execution.queue_ms,
                "execution_ms": execution.execution_ms,
                "total_ms": execution.total_ms,
                "error": execution.error,
            }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clear_completed(self):
        """Remove completed/failed tasks from blackboard."""

        if hasattr(
            self.blackboard,
            "clear_completed",
        ):
            self.blackboard.clear_completed()
        else:
            debug(
                "Scheduler.clear_completed: "
                "blackboard has no clear_completed()"
            )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self, wait=True):
        """
        Shut down the executor.

        If wait=True, finish all pending executor work.
        """

        self.executor.shutdown(
            wait=wait,
        )
