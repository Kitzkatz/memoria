# memory/pruner.py
"""
Memory Pruner — Background task that archives or deletes old, low-importance memories.
"""

import time
import threading
from datetime import datetime, timedelta
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
        self._running = False
        self._thread: Optional[threading.Thread] = None

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
        debug("[MemoryPruner] Starting prune scan...")
        start_time = time.perf_counter()

        # Get all memories
        memories = self.db.fetch_all()
        total = len(memories)

        if total == 0:
            return {"pruned": 0, "total": 0, "dry_run": dry_run}

        # Calculate cutoff date
        cutoff = datetime.utcnow() - timedelta(days=self.max_age_days)
        cutoff_str = cutoff.isoformat()

        pruned_ids = []
        pruned_count = 0

        for mem in memories:
            mem_id = mem["id"]
            importance = mem.get("importance", 0.0)
            created_at = mem.get("created_at", "")

            # Check if memory should be pruned
            if importance < self.threshold and created_at < cutoff_str:
                pruned_ids.append(mem_id)
                pruned_count += 1

                if not dry_run:
                    # Soft delete the memory
                    self.db.delete(mem_id)
                    # Remove from embedding cache
                    self.embedding_cache.remove(mem_id)
                    # Note: FAISS index doesn't support removal easily.
                    # The vector_store will be rebuilt periodically or on demand.

                # Batch processing to avoid memory blowup
                if len(pruned_ids) >= self.batch_size:
                    debug(f"[MemoryPruner] Pruned {len(pruned_ids)} memories so far...")
                    pruned_ids = []

        # Final batch
        if pruned_ids and not dry_run:
            debug(f"[MemoryPruner] Final batch: {len(pruned_ids)} memories pruned")

        elapsed = time.perf_counter() - start_time
        debug(f"[MemoryPruner] Prune scan complete: {pruned_count} pruned, {total} total, {elapsed:.2f}s")

        return {
            "pruned": pruned_count,
            "total": total,
            "dry_run": dry_run,
            "elapsed_seconds": round(elapsed, 2),
        }

    # ---------------------------------
    # Background runner
    # ---------------------------------

    def start(self):
        """Start the background pruner thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        debug(f"[MemoryPruner] Started with interval {self.interval_seconds}s")

    def stop(self):
        """Stop the background pruner thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        debug("[MemoryPruner] Stopped")

    def _loop(self):
        """Background loop."""
        while self._running:
            try:
                self.prune(dry_run=False)
            except Exception as e:
                debug(f"[MemoryPruner] Error in background loop: {e}")
            time.sleep(self.interval_seconds)

    # ---------------------------------
    # Manual control
    # ---------------------------------

    def prune_now(self, dry_run: bool = False) -> dict:
        """Run a prune immediately."""
        return self.prune(dry_run=dry_run)
