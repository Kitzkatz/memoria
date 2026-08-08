"""
Embedding cache — in-memory cache for embeddings with pickle persistence.
Pickle is used for performance (fast binary serialization of large numpy arrays).
"""

from core.logger import debug
from threading import RLock
import pickle
import json
import time
import hashlib
import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from cache.config import settings


# Version stamp for cache files
CACHE_VERSION = 2


class EmbeddingCache:

    def __init__(self, cache_path=None):
        self._cache: Dict[int, List[float]] = {}
        self._lock = RLock()
        self.cache_path = Path(cache_path or settings.CACHE_PATH)
        self._dirty = False
        self._last_save = time.time()
        self._save_threshold = 5.0
        self._max_size = getattr(settings, "EMBEDDING_CACHE_MAX_SIZE", 100000)
        self._version = CACHE_VERSION
        self._load()

    # ------------------------
    # Add / Update
    # ------------------------

    def add(self, mem_id: int, vector: List[float]):
        with self._lock:
            self._cache[mem_id] = vector
            self._dirty = True
            self._trim_if_needed()
            self._maybe_save()

    def add_many(self, ids: List[int], vectors: List[List[float]]):
        with self._lock:
            for mem_id, vector in zip(ids, vectors):
                if vector is not None:
                    self._cache[mem_id] = vector
            self._dirty = True
            self._trim_if_needed()
            self._maybe_save()

    # ------------------------
    # Retrieve
    # ------------------------

    def get(self, mem_id: int) -> Optional[List[float]]:
        with self._lock:
            return self._cache.get(mem_id)

    def get_many(self, ids: List[int]) -> Dict[int, Optional[List[float]]]:
        with self._lock:
            return {mem_id: self._cache.get(mem_id) for mem_id in ids}

    # ------------------------
    # Exists
    # ------------------------

    def contains(self, mem_id: int) -> bool:
        with self._lock:
            return mem_id in self._cache

    # ------------------------
    # Delete
    # ------------------------

    def remove(self, mem_id: int):
        with self._lock:
            if mem_id in self._cache:
                del self._cache[mem_id]
                self._dirty = True
                self._maybe_save()

    def remove_many(self, ids: List[int]):
        with self._lock:
            for mem_id in ids:
                if mem_id in self._cache:
                    del self._cache[mem_id]
            self._dirty = True
            self._maybe_save()

    # ------------------------
    # Diagnostics
    # ------------------------

    def count(self) -> int:
        with self._lock:
            return len(self._cache)

    def size_bytes(self) -> int:
        with self._lock:
            import sys
            total = sys.getsizeof(self._cache)
            for key, value in self._cache.items():
                total += sys.getsizeof(key)
                total += sys.getsizeof(value)
            return total

    # ------------------------
    # Clear
    # ------------------------

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._dirty = True
            self._maybe_save()

    # ------------------------
    # Iterate
    # ------------------------

    def items(self) -> List[Tuple[int, List[float]]]:
        with self._lock:
            return list(self._cache.items())

    def keys(self) -> List[int]:
        with self._lock:
            return list(self._cache.keys())

    def values(self) -> List[List[float]]:
        with self._lock:
            return list(self._cache.values())

    # ------------------------
    # Size Management
    # ------------------------

    def _trim_if_needed(self):
        if len(self._cache) <= self._max_size:
            return

        debug(f"[EMBEDDING CACHE] Trimming from {len(self._cache)} to {self._max_size}", category="cache")

        # Remove oldest entries (simplified LRU)
        keys = list(self._cache.keys())
        to_remove = keys[:len(keys) - self._max_size]
        for key in to_remove:
            del self._cache[key]

        debug(f"[EMBEDDING CACHE] Trimmed to {len(self._cache)} entries", category="cache")

    # ------------------------
    # Persistence (Pickle with safeguards)
    # ------------------------

    def _maybe_save(self):
        now = time.time()
        if now - self._last_save >= self._save_threshold and self._dirty:
            self.save()
            self._dirty = False
            self._last_save = now

    def save(self):
        """Save cache to disk using pickle."""
        with self._lock:
            try:
                # Ensure directory exists
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)

                # Wrap with metadata (version, checksum)
                data = {
                    "version": self._version,
                    "timestamp": time.time(),
                    "count": len(self._cache),
                    "cache": self._cache,
                }

                # Write atomically using temp file
                temp_path = self.cache_path.with_suffix(".tmp")
                with open(temp_path, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

                # Atomic rename
                temp_path.replace(self.cache_path)

                debug(f"[EMBEDDING CACHE] Saved {len(self._cache)} vectors to {self.cache_path} ({self.cache_path.stat().st_size} bytes)", category="cache")

            except Exception as e:
                debug(f"[EMBEDDING CACHE] Save error: {e}", category="cache")
                # Clean up temp file if it exists
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except:
                        pass

    def save_json(self, path: Optional[str] = None):
        """Save cache as JSON for debugging (slow, large)."""
        with self._lock:
            path = path or str(self.cache_path) + ".json"
            try:
                data = {str(k): v for k, v in self._cache.items()}
                with open(path, "w") as f:
                    json.dump(data, f, indent=2)
                debug(f"[EMBEDDING CACHE] Saved JSON to {path}", category="cache")
            except Exception as e:
                debug(f"[EMBEDDING CACHE] JSON save error: {e}", category="cache")

    def _load(self):
        """Load cache from disk with version checking."""
        if not self.cache_path.exists():
            debug("[EMBEDDING CACHE] No cache file found, starting empty", category="cache")
            return

        try:
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)

            # New format (with metadata)
            if isinstance(data, dict) and "cache" in data:
                version = data.get("version", 1)
                if version != self._version:
                    debug(f"[EMBEDDING CACHE] Version mismatch (file={version}, expected={self._version}), rebuilding", category="cache")
                    self._cache = {}
                    return

                self._cache = data["cache"]
                debug(f"[EMBEDDING CACHE] Loaded {len(self._cache)} vectors (v{version}) from {self.cache_path}", category="cache")

            # Legacy format (raw dict)
            elif isinstance(data, dict):
                self._cache = data
                debug(f"[EMBEDDING CACHE] Loaded {len(self._cache)} vectors (legacy) from {self.cache_path}", category="cache")

            else:
                debug("[EMBEDDING CACHE] Invalid cache format, starting empty", category="cache")
                self._cache = {}

        except (pickle.UnpicklingError, EOFError, ValueError, AttributeError) as e:
            debug(f"[EMBEDDING CACHE] Failed to load: {e}, starting empty", category="cache")
            self._cache = {}
        except Exception as e:
            debug(f"[EMBEDDING CACHE] Load error: {e}, starting empty", category="cache")
            self._cache = {}

    # ------------------------
    # Verification
    # ------------------------

    def verify(self, db):
        with self._lock:
            db_rows = db.fetch_all()
            db_ids = {row["id"] for row in db_rows}
            cache_ids = set(self._cache.keys())

            missing = db_ids - cache_ids
            orphaned = cache_ids - db_ids

            debug(
                f"[EMBEDDING CACHE] DB={len(db_ids)} "
                f"Cache={len(cache_ids)} "
                f"Missing={len(missing)} "
                f"Orphaned={len(orphaned)}",
                category="cache"
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

    def rebuild(self, db, vector_store, force: bool = False):
        """
        Rebuild cache from the database and vector store.

        Args:
            db: Database connection
            vector_store: Vector store with embeddings
            force: If True, rebuild even if cache exists
        """
        with self._lock:
            # Check if we should rebuild
            if not force and self.cache_path.exists() and self.count() > 0:
                debug("[EMBEDDING CACHE] Cache already exists, skipping rebuild (use force=True to override)", category="cache")
                return

            self._cache.clear()
            rows = db.fetch_all()

            if not rows:
                debug("[EMBEDDING CACHE] No rows in database, cache remains empty", category="cache")
                return

            debug(f"[EMBEDDING CACHE] Rebuilding {len(rows)} vectors from database...", category="cache")

            for row in rows:
                mem_id = row["id"]
                vector = vector_store.get(mem_id)
                if vector is not None:
                    self._cache[mem_id] = vector

            debug(
                f"[EMBEDDING CACHE] Rebuilt {len(self._cache)} vectors "
                f"from {len(rows)} db rows",
                category="cache"
            )

            self._dirty = True
            self.save()

    # ------------------------
    # Stats
    # ------------------------

    def stats(self) -> dict:
        with self._lock:
            return {
                "count": len(self._cache),
                "max_size": self._max_size,
                "dirty": self._dirty,
                "cache_path": str(self.cache_path),
                "file_exists": self.cache_path.exists(),
                "file_size_bytes": self.cache_path.stat().st_size if self.cache_path.exists() else 0,
                "size_bytes": self.size_bytes(),
                "version": self._version,
            }

    # ------------------------
    # Magic Methods
    # ------------------------

    def __len__(self) -> int:
        return self.count()

    def __contains__(self, mem_id: int) -> bool:
        return self.contains(mem_id)

    def __repr__(self) -> str:
        return f"EmbeddingCache(count={self.count()}, max_size={self._max_size}, path={self.cache_path})"
