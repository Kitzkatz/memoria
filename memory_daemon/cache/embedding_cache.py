from core.logger import debug
from threading import RLock
import pickle
from pathlib import Path
from cache.config import settings


class EmbeddingCache:

    def __init__(self, cache_path=None):
        self._cache = {}
        self._lock = RLock()
        self.cache_path = Path(cache_path or settings.CACHE_PATH)
        self._load()

    # ------------------------
    # Add / Update
    # ------------------------
    def add(self, mem_id, vector):
        with self._lock:
            self._cache[mem_id] = vector

    # ------------------------
    # Batch Add
    # ------------------------
    def add_many(self, ids, vectors):
        with self._lock:
            for mem_id, vector in zip(ids, vectors):
                self._cache[mem_id] = vector

    # ------------------------
    # Retrieve
    # ------------------------
    def get(self, mem_id):
        with self._lock:
            return self._cache.get(mem_id)

    # ------------------------
    # Exists
    # ------------------------
    def contains(self, mem_id):
        with self._lock:
            return mem_id in self._cache

    # ------------------------
    # Delete
    # ------------------------
    def remove(self, mem_id):
        with self._lock:
            self._cache.pop(mem_id, None)

    # ------------------------
    # Diagnostics
    # ------------------------
    def count(self):
        with self._lock:
            return len(self._cache)

    # ------------------------
    # Clear Runtime
    # ------------------------
    def clear(self):
        with self._lock:
            self._cache.clear()

    # ------------------------
    # Iterate
    # ------------------------
    def items(self):
        with self._lock:
            return list(self._cache.items())

    # ------------------------
    # Persistence
    # ------------------------
    def save(self):
        with self._lock:
            with open(self.cache_path, "wb") as f:
                pickle.dump(self._cache, f)

            debug(f"[EMBEDDING CACHE] Saved {len(self._cache)} vectors")

    def _load(self):
        if not self.cache_path.exists():
            debug("[EMBEDDING CACHE] No cache file found, starting empty")
            return

        try:
            with open(self.cache_path, "rb") as f:
                self._cache = pickle.load(f)

            debug(f"[EMBEDDING CACHE] Loaded {len(self._cache)} vectors")

        except Exception as e:
            debug(f"[EMBEDDING CACHE] Failed to load ({e}), starting empty")
            self._cache = {}

    # ------------------------
    # Verification
    # ------------------------
    def verify(self, db):

        db_ids = {row["id"] for row in db.fetch_all()}
        cache_ids = set(self._cache.keys())

        missing = db_ids - cache_ids
        orphaned = cache_ids - db_ids

        debug(
            f"[EMBEDDING CACHE] DB={len(db_ids)} "
            f"Cache={len(cache_ids)} "
            f"Missing={len(missing)} "
            f"Orphaned={len(orphaned)}"
        )

        return {
            "db_count": len(db_ids),
            "cache_count": len(cache_ids),
            "missing": missing,
            "orphaned": orphaned,
            "in_sync": not missing and not orphaned
        }

    # ------------------------
    # Rebuild
    # ------------------------
    def rebuild(self, db, vector_store):

        with self._lock:
            self._cache.clear()

            rows = db.fetch_all()

            for row in rows:
                mem_id = row["id"]
                vector = vector_store.get(mem_id)

                if vector is not None:
                    self._cache[mem_id] = vector

            debug(
                f"[EMBEDDING CACHE] Rebuilt {len(self._cache)} vectors "
                f"from {len(rows)} db rows"
            )

        self.save()
