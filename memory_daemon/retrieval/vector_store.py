from core.logger import debug
import os
import faiss
import numpy as np

from cache.config import settings


class VectorStore:

    def __init__(self, dim):

        self.dim = dim
        self.pending = 0

        self._load()

    # --------------------------------------------------
    # Internal
    # --------------------------------------------------

    def _new_index(self):

        return faiss.IndexIDMap2(
            faiss.IndexFlatL2(self.dim)
        )

    def _load(self):

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

        arr = np.asarray(
            [vector],
            dtype=np.float32
        )

        ids = np.asarray(
            [int(mem_id)],
            dtype=np.int64
        )

        debug("\n========== VECTORSTORE ADD ==========")
        debug("ID:", mem_id)
        debug("Shape:", arr.shape)
        debug("dtype:", arr.dtype)
        debug("Index count BEFORE:", self.index.ntotal)

        self.index.add_with_ids(arr, ids)

        debug("Index count AFTER:", self.index.ntotal)

        self.pending += 1

        #
        # Development mode:
        # force persistence immediately
        #

        if persist:
            self.save()
            return

        #
        # Production mode:
        # batch disk writes
        #

        if self.pending >= 100:
            self.save()
        elif settings.DEBUG:
            self.save()

    # --------------------------------------------------
    # Batch Insert
    # --------------------------------------------------

    def add_many(self, ids, vectors, persist=False):

        arr = np.asarray(
            vectors,
            dtype=np.float32
        )

        id_array = np.asarray(
            ids,
            dtype=np.int64
        )

        debug("\n========== VECTORSTORE BATCH ADD ==========")
        debug("Vectors:", len(vectors))
        debug("Shape:", arr.shape)
        debug("Index BEFORE:", self.index.ntotal)

        self.index.add_with_ids(
            arr,
            id_array
        )

        debug("Index AFTER:", self.index.ntotal)

        self.pending += len(ids)

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

    def search(self, vector, k=None):

        if self.index.ntotal == 0:
            return [], []

        k = min(
            k or settings.TOP_K,
            self.index.ntotal
        )
        debug("TOP_k: ", settings.TOP_K)
        debug("Index total: ", self.index.ntotal)
        debug("Using k: ", k)
        arr = np.asarray(
            [vector],
            dtype=np.float32
        )

        distances, ids = self.index.search(arr, k)
        debug("Raw IDs: ", ids[0])
        debug("Raw distances: ", distances[0])
        
        valid = []
        debug("Valid count: ", len(valid))
        debug("Valid ids: ", ids)
        for mem_id, dist in zip(ids[0], distances[0]):

            if mem_id == -1:
                continue

            valid.append((int(mem_id), float(dist)))

        if not valid:
            return [], []

        ids = [x[0] for x in valid]
        dists = [x[1] for x in valid]

        return ids, dists
        # --------------------------------------------------
    # Retrieve stored embedding (in-memory only)
    # --------------------------------------------------

    def get(self, mem_id):

        try:
            vector = self.index.reconstruct(int(mem_id))
            return vector.tolist()

        except Exception as e:
            debug(f"[FAISS] reconstruct failed for {mem_id}: {e}")
            return None
    # --------------------------------------------------
    # Persistence
    # --------------------------------------------------

    def save(self):

        faiss.write_index(
            self.index,
            settings.VECTOR_INDEX_PATH
        )

        self.pending = 0

        debug(f"[FAISS] Saved {self.index.ntotal} vectors")

    # --------------------------------------------------
    # Maintenance
    # --------------------------------------------------

    def reset(self):

        self.index = self._new_index()

        self.pending = 0

        debug("[FAISS] Reset complete")

    def delete_file(self):

        if os.path.exists(settings.VECTOR_INDEX_PATH):

            os.remove(settings.VECTOR_INDEX_PATH)

            debug("[FAISS] Index file deleted")

    def reset_and_delete(self):

        self.reset()

        self.delete_file()

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------

    def count(self):

        return self.index.ntotal

    def verify(self, db):

        db_count = db.count()

        faiss_count = self.count()

        debug(
            f"[VERIFY] DB={db_count}  FAISS={faiss_count}"
        )

        return db_count == faiss_count


