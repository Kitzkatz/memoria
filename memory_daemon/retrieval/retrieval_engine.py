from core.logger import debug
from memory.models import MemoryRecord
from ranking.models import CandidateRecord
from cache.config import settings
import time
from typing import List, Dict, Optional, Set


class RetrievalEngine:

    def __init__(self, db, vector_store, embedding_cache, graph_search=None):
        self.db = db
        self.vector_store = vector_store
        self.embedding_cache = embedding_cache
        self.graph_search = graph_search

    # -------------------------------------
    # Build one MemoryRecord from a db row
    # -------------------------------------

    def _build_memory_record(self, row):
        return MemoryRecord(
            id=row["id"],
            text=row["text"],
            normalized_text=row["normalized_text"],
            tokens=row["tokens"],
            token_count=row["token_count"],
            memory_type=row["memory_type"],
            metadata=row["metadata"],
            entities=row["entities"],
            relationships=row["relationships"],
            importance=row["importance"],
            created_at=row["created_at"],
            last_accessed=row["last_accessed"]
        )

    # -------------------------------------
    # Helper: get embedding with cache fallback (single)
    # -------------------------------------

    def _get_embedding(self, mem_id):
        """Get embedding from cache, falling back to vector store."""
        embedding = self.embedding_cache.get(mem_id)
        if embedding is None:
            embedding = self.vector_store.get(mem_id)
            if embedding is not None:
                self.embedding_cache.add(mem_id, embedding)
        return embedding

    # -------------------------------------
    # Helper: get embeddings in batch (MASSIVE SPEEDUP)
    # -------------------------------------

    def _get_embeddings_batch(self, mem_ids: List[int]) -> Dict[int, List[float]]:
        """
        Get embeddings for multiple memory IDs in batch.
        This is much faster than calling _get_embedding() for each ID.
        """
        result = {}
        missing_ids = []

        # Check cache first
        for mem_id in mem_ids:
            emb = self.embedding_cache.get(mem_id)
            if emb is not None:
                result[mem_id] = emb
            else:
                missing_ids.append(mem_id)

        # Batch fetch missing from vector store
        if missing_ids:
            vectors = self.vector_store.get_many(missing_ids)
            for mem_id, vector in vectors.items():
                if vector is not None:
                    self.embedding_cache.add(mem_id, vector)
                    result[mem_id] = vector

        return result

    # -------------------------------------
    # FAISS candidates
    # -------------------------------------

    def vector_candidates(self, ids, distances):
        rows = self.db.fetch_many(ids)
        candidates = []

        # Collect all mem_ids for batch embedding
        mem_ids = [mem_id for mem_id in ids if rows.get(mem_id) is not None]
        embeddings = self._get_embeddings_batch(mem_ids)

        for mem_id, dist in zip(ids, distances):
            row = rows.get(mem_id)
            if row is None:
                continue

            memory = self._build_memory_record(row)
            embedding = embeddings.get(mem_id)

            candidates.append(
                CandidateRecord(
                    memory=memory,
                    distance=float(dist),
                    embedding=embedding,
                    graph_hit=False
                )
            )

        return candidates

    # -------------------------------------
    # Graph candidates
    # -------------------------------------

    def graph_candidates(self, entities, existing_ids, depth=1):
        if not self.graph_search or not entities:
            return []

        graph_limit = getattr(settings, "GRAPH_TOP_K", 50)
        graph_memory_ids = self.graph_search.search(
            entities,
            depth=depth,
            limit=graph_limit * 2
        )

        new_ids = [mem_id for mem_id in graph_memory_ids if mem_id not in existing_ids]

        if not new_ids:
            return []

        new_ids = new_ids[:graph_limit]

        rows = self.db.fetch_many(new_ids)

        # Batch get embeddings
        valid_mem_ids = [mem_id for mem_id in new_ids if rows.get(mem_id) is not None]
        embeddings = self._get_embeddings_batch(valid_mem_ids)

        candidates = []
        for mem_id in new_ids:
            row = rows.get(mem_id)
            if row is None:
                continue

            memory = self._build_memory_record(row)
            embedding = embeddings.get(mem_id)

            candidates.append(
                CandidateRecord(
                    memory=memory,
                    distance=0.0,
                    embedding=embedding,
                    graph_hit=True
                )
            )

            existing_ids.add(mem_id)

        return candidates

    # -------------------------------------
    # Attribute candidates
    # -------------------------------------

    def attribute_candidates(self, subject, attribute, existing_ids):
        if not subject or not attribute:
            return []

        rows = self.db.search_attribute(subject, attribute)
        if not rows:
            return []

        # Collect mem_ids for batch embedding
        mem_ids = []
        candidates = []

        for row in rows:
            mem_id = row["id"]
            if mem_id in existing_ids:
                continue
            mem_ids.append(mem_id)

        if not mem_ids:
            return []

        embeddings = self._get_embeddings_batch(mem_ids)

        for row in rows:
            mem_id = row["id"]
            if mem_id in existing_ids:
                continue

            memory = self._build_memory_record(row)
            embedding = embeddings.get(mem_id)

            candidates.append(
                CandidateRecord(
                    memory=memory,
                    distance=0.0,
                    embedding=embedding,
                    graph_hit=False
                )
            )
            existing_ids.add(mem_id)

        return candidates

    # -------------------------------------
    # Combined entry point
    # -------------------------------------

    def retrieve(self, query, ids, distances):
        t0_global = time.perf_counter()

        # -----------------------------
        # Vector candidates (primary)
        # -----------------------------

        t0 = time.perf_counter()
        candidates = self.vector_candidates(ids, distances)
        vector_ms = (time.perf_counter() - t0) * 1000
        debug(f"[TIMING] vector_candidates: {vector_ms:.2f}ms - {len(candidates)} candidates")

        existing_ids = {c.memory.id for c in candidates}

        # -----------------------------
        # Attribute candidates
        # -----------------------------

        t0 = time.perf_counter()
        subject = query.metadata.get("subject")
        attribute = query.metadata.get("attribute")

        attr_candidates = self.attribute_candidates(subject, attribute, existing_ids)
        attr_ms = (time.perf_counter() - t0) * 1000

        if attr_candidates:
            candidates.extend(attr_candidates)
            debug(f"[TIMING] attribute_candidates: {attr_ms:.2f}ms - {len(attr_candidates)} candidates")

        # -----------------------------
        # Graph candidates
        # -----------------------------

        t0 = time.perf_counter()
        graph_depth = getattr(settings, "GRAPH_SEARCH_DEPTH", 1)
        graph_candidates = self.graph_candidates(
            query.entities,
            existing_ids,
            depth=graph_depth
        )
        graph_ms = (time.perf_counter() - t0) * 1000

        graph_limit = getattr(settings, "GRAPH_TOP_K", 50)
        if graph_candidates:
            candidates.extend(graph_candidates[:graph_limit])
            debug(f"[TIMING] graph_candidates: {graph_ms:.2f}ms - {len(graph_candidates)} candidates")

        total_ms = (time.perf_counter() - t0_global) * 1000
        debug(f"[TIMING] TOTAL retrieve: {total_ms:.2f}ms - {len(candidates)} total candidates")

        return candidates
