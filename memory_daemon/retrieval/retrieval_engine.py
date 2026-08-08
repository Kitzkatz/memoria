from core.logger import debug
from memory.models import MemoryRecord
from ranking.models import CandidateRecord
from cache.config import settings


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
    # Helper: get embedding with cache fallback
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
    # FAISS candidates
    # -------------------------------------

    def vector_candidates(self, ids, distances):
        rows = self.db.fetch_many(ids)
        candidates = []

        for mem_id, dist in zip(ids, distances):
            row = rows.get(mem_id)
            if row is None:
                continue

            memory = self._build_memory_record(row)
            embedding = self._get_embedding(mem_id)

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
            limit=graph_limit * 2  # Fetch extra to account for filtering
        )

        # Filter out existing IDs and duplicates
        new_ids = [mem_id for mem_id in graph_memory_ids if mem_id not in existing_ids]

        if not new_ids:
            return []

        # Limit after filtering
        new_ids = new_ids[:graph_limit]

        # Batch fetch
        rows = self.db.fetch_many(new_ids)

        candidates = []
        for mem_id in new_ids:
            row = rows.get(mem_id)
            if row is None:
                continue

            memory = self._build_memory_record(row)
            embedding = self._get_embedding(mem_id)

            candidates.append(
                CandidateRecord(
                    memory=memory,
                    distance=0.0,
                    embedding=embedding,
                    graph_hit=True
                )
            )

        return candidates

    # -------------------------------------
    # Attribute candidates
    # -------------------------------------

    def attribute_candidates(self, subject, attribute, existing_ids):
        """Get candidates via attribute search."""
        if not subject or not attribute:
            return []

        rows = self.db.search_attribute(subject, attribute)
        if not rows:
            return []

        candidates = []
        for row in rows:
            mem_id = row["id"]
            if mem_id in existing_ids:
                continue

            memory = self._build_memory_record(row)
            embedding = self._get_embedding(mem_id)

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
        import time

        t0_global = time.perf_counter()

        # -----------------------------
        # Vector candidates (primary)
        # -----------------------------

        t0 = time.perf_counter()
        candidates = self.vector_candidates(ids, distances)
        vector_ms = (time.perf_counter() - t0) * 1000
        debug(f"[TIMING] vector_candidates: {vector_ms:.2f}ms - {len(candidates)} candidates")

        existing_ids = {c.memory.id for c in candidates}

        # Early exit if we have enough candidates
        top_k = getattr(settings, "TOP_K", 10)
        if len(candidates) >= top_k * 3:  # 3x top_k before spending more time
            total_ms = (time.perf_counter() - t0_global) * 1000
            debug(f"[TIMING] Early exit: {total_ms:.2f}ms - {len(candidates)} candidates")
            return candidates

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
