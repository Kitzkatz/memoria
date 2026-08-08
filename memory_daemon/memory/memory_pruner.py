"""
Memory Pruner — Background task that archives or deletes old, low-importance memories.
"""

import time
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

from core.logger import debug


class MemoryPruner:
    def __init__(
        self,
        db,
        vector_store,
        embedding_cache,
        threshold: float = 0.1,
        max_age_days: int = 365,
        batch_size: int = 100,
        interval_seconds: int = 3600,
        auto_start: bool = False,
    ):
        self.db = db
        self.vector_store = vector_store
        self.embedding_cache = embedding_cache
        self.threshold = threshold
        self.max_age_days = max_age_days
        self.batch_size = batch_size
        self.interval_seconds = interval_seconds

        # Thread safety
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stopping = False

        if auto_start:
            self.start()

    # ---------------------------------
    # Prune logic
    # ---------------------------------

    def prune(self, dry_run: bool = False) -> dict:
        """
        Prune memories that are below the importance threshold and older than max_age_days.
        Returns a summary of what was pruned.
        """
        with self._lock:
            if self._stopping:
                return {"pruned": 0, "total": 0, "dry_run": dry_run, "interrupted": True}

            debug("[MemoryPruner] Starting prune scan...")
            start_time = time.perf_counter()

            # Get total count
            total = self.db.count()
            if total == 0:
                return {"pruned": 0, "total": 0, "dry_run": dry_run}

            # Calculate cutoff date (timezone-aware)
            cutoff = datetime.now(timezone.utc) - timedelta(days=self.max_age_days)
            cutoff_str = cutoff.isoformat()

            pruned_count = 0
            last_id = 0

            # Use a direct cursor for batched fetching
            conn = self.db.conn
            cur = conn.cursor()

            while True:
                # Check if we should stop
                if self._stopping:
                    debug("[MemoryPruner] Prune interrupted by stop signal")
                    break

                # Fetch a batch using id-based pagination
                cur.execute("""
                    SELECT *
                    FROM memories
                    WHERE id > ?
                      AND tombstone = 0
                      AND importance < ?
                      AND created_at < ?
                    ORDER BY id
                    LIMIT ?
                """, (last_id, self.threshold, cutoff_str, self.batch_size))

                batch = cur.fetchall()

                if not batch:
                    break

                pruned_in_batch = 0
                for row in batch:
                    mem_id = row["id"]

                    if not dry_run:
                        # Soft delete using existing delete() method
                        self.db.delete(mem_id)

                        # Remove from embedding cache
                        self.embedding_cache.remove(mem_id)

                    pruned_in_batch += 1
                    last_id = mem_id

                pruned_count += pruned_in_batch
                debug(f"[MemoryPruner] Pruned {pruned_count} memories so far...")

                # If we got fewer than batch_size, we've hit the end
                if len(batch) < self.batch_size:
                    break

            # If we pruned something and not dry run, rebuild the FAISS index
            if pruned_count > 0 and not dry_run:
                debug("[MemoryPruner] Rebuilding FAISS index after pruning...")
                self.vector_store.rebuild_from_db(self.db)

            elapsed = time.perf_counter() - start_time
            debug(f"[MemoryPruner] Prune scan complete: {pruned_count} pruned, {total} total, {elapsed:.2f}s")

            return {
                "pruned": pruned_count,
                "total": total,
                "dry_run": dry_run,
                "elapsed_seconds": round(elapsed, 2),
                "interrupted": False,
            }

    # ---------------------------------
    # Background runner
    # ---------------------------------

    def start(self):
        """Start the background pruner thread."""
        with self._lock:
            if self._running:
                return

            self._running = True
            self._stopping = False
            self._thread = threading.Thread(target=self._loop, name="MemoryPruner")
            self._thread.daemon = False  # Not a daemon — let it finish gracefully
            self._thread.start()
            debug(f"[MemoryPruner] Started with interval {self.interval_seconds}s")

    def stop(self, wait: bool = True):
        """Stop the background pruner thread gracefully."""
        with self._lock:
            if not self._running:
                return

            self._stopping = True
            self._running = False

        if wait and self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                debug("[MemoryPruner] Thread did not stop gracefully")
            else:
                debug("[MemoryPruner] Thread stopped")

        debug("[MemoryPruner] Stopped")

    def _loop(self):
        """Background loop."""
        while self._running and not self._stopping:
            try:
                self.prune(dry_run=False)
            except Exception as e:
                debug(f"[MemoryPruner] Error in background loop: {e}")

            # Sleep in small increments to allow stop signal
            for _ in range(self.interval_seconds):
                if self._stopping or not self._running:
                    break
                time.sleep(1)

    # ---------------------------------
    # Manual control
    # ---------------------------------

    def prune_now(self, dry_run: bool = False) -> dict:
        """Run a prune immediately."""
        return self.prune(dry_run=dry_run)
