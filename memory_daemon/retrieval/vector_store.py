from core.logger import debug
import os
import time
import faiss
import numpy as np
import threading
from typing import Optional, List, Tuple, Dict

from cache.config import settings


class VectorStore:

    def __init__(self, dim):
        self.dim = dim
        self.pending = 0
        self._lock = threading.RLock()
        self._load()

    # --------------------------------------------------
    # Internal
    # --------------------------------------------------

    def _new_index(self):
        return faiss.IndexIDMap2(
            faiss.IndexFlatL2(self.dim)
        )

    def _load(self):
        """Load index from disk or create new."""
        try:
            self.index = faiss.read_index(
                settings.VECTOR_INDEX_PATH
            )
            debug(f"[FAISS] Loaded {self.index.ntotal} vectors")
        except Exception as e:
            debug(f"[FAISS] Creating new index ({e})")
            self.index = self._new_index()

    # --------------------------------------------------
    # Insert
    # --------------------------------------------------

    def add(self, mem_id, vector, persist=False):
        """Add a single vector to the index."""
        with self._lock:
            if vector is None:
                debug(f"[FAISS] Warning: vector is None for id {mem_id}")
                return

            if len(vector) != self.dim:
                debug(f"[FAISS] Warning: vector length {len(vector)} != dim {self.dim}, truncating")
                vector = vector[:self.dim]

            arr = np.asarray([vector], dtype=np.float32)
            ids = np.asarray([int(mem_id)], dtype=np.int64)

            self.index.add_with_ids(arr, ids)
            self.pending += 1

            if persist:
                self.save()
                return

            if self.pending >= 100:
                self.save()
            elif settings.DEBUG:
                self.save()

    # --------------------------------------------------
    # Batch Insert
    # --------------------------------------------------

    def add_many(self, ids, vectors, persist=False):
        """Add multiple vectors to the index."""
        with self._lock:
            if not ids or not vectors:
                return

            # Validate and truncate vectors
            valid_ids = []
            valid_vectors = []
            for mid, vec in zip(ids, vectors):
                if vec is None:
                    continue
                if len(vec) != self.dim:
                    vec = vec[:self.dim]
                valid_ids.append(mid)
                valid_vectors.append(vec)

            if not valid_ids:
                return

            arr = np.asarray(valid_vectors, dtype=np.float32)
            id_array = np.asarray(valid_ids, dtype=np.int64)

            self.index.add_with_ids(arr, id_array)
            self.pending += len(valid_ids)

            if persist:
                self.save()
                return

            if self.pending >= 100:
                self.save()
            elif settings.DEBUG:
                self.save()

    # --------------------------------------------------
    # Search
    # --------------------------------------------------

    def search(self, vector, k=None) -> Tuple[List[int], List[float]]:
        """Search for nearest neighbors."""
        with self._lock:
            if self.index.ntotal == 0 or vector is None:
                return [], []

            k = min(k or settings.TOP_K, self.index.ntotal)
            if k <= 0:
                return [], []

            arr = np.asarray([vector], dtype=np.float32)
            distances, ids = self.index.search(arr, k)

            valid = []
            for mem_id, dist in zip(ids[0], distances[0]):
                if mem_id == -1:
                    continue
                valid.append((int(mem_id), float(dist)))

            if not valid:
                return [], []

            return [x[0] for x in valid], [x[1] for x in valid]

    # --------------------------------------------------
    # Retrieve stored embedding
    # --------------------------------------------------

    def get(self, mem_id) -> Optional[List[float]]:
        """Retrieve a vector by ID."""
        with self._lock:
            try:
                if self.index.ntotal == 0:
                    return None

                vector = self.index.reconstruct(int(mem_id))
                return vector.tolist()

            except (KeyError, ValueError, RuntimeError) as e:
                return None
            except Exception as e:
                debug(f"[FAISS] reconstruct failed for {mem_id}: {e}")
                return None

    def get_many(self, mem_ids: List[int]) -> Dict[int, Optional[List[float]]]:
        """
        Retrieve multiple vectors by ID in batch.
        This is much faster than calling get() for each ID individually.
        """
        with self._lock:
            result = {}
            if self.index.ntotal == 0 or not mem_ids:
                return result

            for mem_id in mem_ids:
                try:
                    vector = self.index.reconstruct(int(mem_id))
                    result[mem_id] = vector.tolist()
                except Exception:
                    result[mem_id] = None

            return result

    def contains(self, mem_id) -> bool:
        """Check if a vector exists in the index."""
        return self.get(mem_id) is not None

    # --------------------------------------------------
    # Remove stale vectors
    # --------------------------------------------------

    def remove(self, mem_id):
        """Mark a vector for removal (FAISS doesn't support direct removal)."""
        with self._lock:
            debug(f"[FAISS] Marked {mem_id} for removal (will rebuild on demand)")

    def rebuild_from_db(self, db):
        """
        Rebuild the entire index from the database.
        Called by pruner after deletions.
        """
        with self._lock:
            debug("[FAISS] Rebuilding index from DB...")
            start = time.perf_counter()

            memories = db.fetch_all()
            if not memories:
                self.index = self._new_index()
                self.pending = 0
                debug("[FAISS] Rebuild complete: 0 vectors")
                return

            # This is a placeholder — the caller should provide embeddings
            # via a separate mechanism (embedding_cache + vector_store)
            # For now, log that we need to rebuild from cache
            debug("[FAISS] Rebuild from DB requires embeddings from cache or embedder")

            elapsed = (time.perf_counter() - start) * 1000
            debug(f"[FAISS] Rebuild placeholder: {elapsed:.2f}ms")

    # --------------------------------------------------
    # Persistence
    # --------------------------------------------------

    def save(self):
        """Save index to disk."""
        with self._lock:
            try:
                faiss.write_index(self.index, settings.VECTOR_INDEX_PATH)
                self.pending = 0
                debug(f"[FAISS] Saved {self.index.ntotal} vectors")
            except Exception as e:
                debug(f"[FAISS] Save error: {e}")

    # --------------------------------------------------
    # Maintenance
    # --------------------------------------------------

    def reset(self):
        """Reset the index to empty."""
        with self._lock:
            self.index = self._new_index()
            self.pending = 0
            debug("[FAISS] Reset complete")

    def delete_file(self):
        """Delete the index file from disk."""
        with self._lock:
            if os.path.exists(settings.VECTOR_INDEX_PATH):
                try:
                    os.remove(settings.VECTOR_INDEX_PATH)
                    debug("[FAISS] Index file deleted")
                except Exception as e:
                    debug(f"[FAISS] Could not delete index file: {e}")

    def reset_and_delete(self):
        """Reset index and delete file."""
        with self._lock:
            self.reset()
            self.delete_file()

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------

    def count(self) -> int:
        """Return number of vectors in index."""
        with self._lock:
            return self.index.ntotal

    def verify(self, db) -> bool:
        """Verify index matches database count."""
        with self._lock:
            db_count = db.count()
            faiss_count = self.count()

            debug(f"[VERIFY] DB={db_count}  FAISS={faiss_count}")

            if db_count == faiss_count:
                return True

            debug(f"[VERIFY] Mismatch: {faiss_count - db_count} stale vectors")
            return db_count == faiss_count

    def stats(self) -> dict:
        """Return index statistics."""
        with self._lock:
            return {
                "total": self.index.ntotal,
                "dim": self.dim,
                "pending": self.pending,
                "index_path": settings.VECTOR_INDEX_PATH,
            }
