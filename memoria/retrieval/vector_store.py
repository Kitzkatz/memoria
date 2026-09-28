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
            loaded = faiss.read_index(settings.VECTOR_INDEX_PATH)

            if loaded.d != self.dim:
                raise ValueError(
                    f"FAISS index dimension {loaded.d} does not match expected {self.dim}"
                )

            self.index = loaded
            debug(f"[FAISS] Loaded {self.index.ntotal} vectors")
        except Exception as e:
            debug(f"[FAISS] Creating new index ({e})")
            self.index = self._new_index()

    # --------------------------------------------------
    # Insert
    # --------------------------------------------------

    def store(self, mem_id, vector, persist=False):
        """Add a single vector to the index."""
        with self._lock:
            if vector is None:
                debug(f"[FAISS] Warning: vector is None for id {mem_id}")
                return

            if len(vector) != self.dim:
                raise ValueError(
                    f"Vector dimension {len(vector)} does not match index dimension {self.dim}"
                )

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

    def store_many(self, ids, vectors, persist=False):
        """Add multiple vectors to the index."""
        with self._lock:
            if not ids or not vectors:
                return

            # Validate vectors
            valid_ids = []
            valid_vectors = []
            for mid, vec in zip(ids, vectors):
                if vec is None:
                    continue
                if len(vec) != self.dim:
                    raise ValueError(
                        f"Vector dimension {len(vec)} does not match index dimension {self.dim}"
                    )
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

    def fetch(self, mem_id) -> Optional[List[float]]:
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

    def fetch_many(self, mem_ids: List[int]) -> Dict[int, Optional[List[float]]]:
        """
        Retrieve multiple vectors by ID in batch.
        This is much faster than calling fetch() for each ID individually.
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
        return self.fetch(mem_id) is not None

    # --------------------------------------------------
    # Remove stale vectors
    # --------------------------------------------------

    def remove(self, mem_id):
        """Remove a vector from the index by memory ID."""
        with self._lock:
            ids = np.asarray([int(mem_id)], dtype=np.int64)
            removed = self.index.remove_ids(ids)

            if removed:
                self.pending += int(removed)
                debug(f"[FAISS] Removed vector {mem_id}")
            else:
                debug(f"[FAISS] Vector {mem_id} not found")
                
    def rebuild_from_db(self, db, embedding_cache):
        """
        Rebuild the entire FAISS index from active database memories
        using embeddings already stored in the embedding cache.
        """
        with self._lock:
            debug("[FAISS] Rebuilding index from DB...")
            start = time.perf_counter()

            memories = db.fetch_all()

            new_index = self._new_index()

            if not memories:
                self.index = new_index
                self.pending = 0
                debug("[FAISS] Rebuild complete: 0 vectors")
                return

            ids = []
            vectors = []
            missing = []

            for memory in memories:
                mem_id = int(memory["id"])
                vector = embedding_cache.get(mem_id)

                if vector is None:
                    missing.append(mem_id)
                    continue

                if len(vector) != self.dim:
                    raise ValueError(
                        f"Embedding dimension {len(vector)} for memory {mem_id} "
                        f"does not match index dimension {self.dim}"
                    )

                ids.append(mem_id)
                vectors.append(vector)

            if ids:
                arr = np.asarray(vectors, dtype=np.float32)
                id_array = np.asarray(ids, dtype=np.int64)
                new_index.add_with_ids(arr, id_array)

            self.index = new_index
            self.pending = 0
            self.save()

            elapsed = (time.perf_counter() - start) * 1000

            debug(
                f"[FAISS] Rebuild complete: "
                f"{len(ids)} vectors, "
                f"{len(missing)} missing embeddings, "
                f"{elapsed:.2f}ms"
            )

            if missing:
                debug(
                    f"[FAISS] Missing embeddings for IDs: "
                    f"{sorted(missing)[:20]}"
                )

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
    # Cache-specific methods
    # --------------------------------------------------

    def save_to_file(self, filepath: str) -> None:
        """Save the current index to a specific file."""
        with self._lock:
            try:
                faiss.write_index(self.index, filepath)
                debug(f"[FAISS] Saved index to {filepath} (ntotal={self.index.ntotal})")
            except Exception as e:
                debug(f"[FAISS] Save to file error: {e}")

    def load_from_file(self, filepath: str) -> None:
        """Load an index from a specific file, replacing the current index."""
        with self._lock:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Index file not found: {filepath}")
            try:
                loaded = faiss.read_index(filepath)

                if loaded.d != self.dim:
                    raise ValueError(
                        f"FAISS index dimension {loaded.d} does not match expected {self.dim}"
                    )

                self.index = loaded
                self.pending = 0
                debug(f"[FAISS] Loaded index from {filepath} (ntotal={self.index.ntotal})")
            except Exception as e:
                debug(f"[FAISS] Load from file error: {e}")
                raise

    def cache_exists(self, filepath: str) -> bool:
        """Check if a cache file exists."""
        return os.path.exists(filepath)

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
        """Verify FAISS vector IDs match non-tombstone database memory IDs."""
        with self._lock:
            db_memories = db.fetch_all()
            db_ids = {int(memory["id"]) for memory in db_memories}

            if not isinstance(self.index, faiss.IndexIDMap2):
                debug("[VERIFY] FAISS index is not an IndexIDMap2")
                return False

            faiss_ids = set(
                int(mem_id)
                for mem_id in faiss.vector_to_array(self.index.id_map)
            )

            if db_ids == faiss_ids:
                debug(
                    f"[VERIFY] DB={len(db_ids)}  FAISS={len(faiss_ids)}  IDs match"
                )
                return True

            missing_from_faiss = db_ids - faiss_ids
            stale_in_faiss = faiss_ids - db_ids

            debug(
                f"[VERIFY] Mismatch: "
                f"DB={len(db_ids)} FAISS={len(faiss_ids)} "
                f"missing_from_faiss={len(missing_from_faiss)} "
                f"stale_in_faiss={len(stale_in_faiss)}"
            )

            if missing_from_faiss:
                debug(
                    f"[VERIFY] Missing FAISS IDs: "
                    f"{sorted(missing_from_faiss)[:20]}"
                )

            if stale_in_faiss:
                debug(
                    f"[VERIFY] Stale FAISS IDs: "
                    f"{sorted(stale_in_faiss)[:20]}"
                )

            return False

    def stats(self) -> dict:
        """Return index statistics."""
        with self._lock:
            return {
                "total": self.index.ntotal,
                "dim": self.dim,
                "pending": self.pending,
                "index_path": settings.VECTOR_INDEX_PATH,
            }
