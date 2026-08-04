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

            embedding = (
                self.embedding_cache.get(mem_id)
                or self.vector_store.get(mem_id)
            )

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

    def graph_candidates(self, entities, existing_ids):
        if not self.graph_search or not entities:
            return []
        
        # Pass limit to graph search
        graph_memory_ids = self.graph_search.search(entities, depth=1, limit=getattr(settings, "GRAPH_TOP_K", 50))
        
        # Filter out existing IDs
        new_ids = [mem_id for mem_id in graph_memory_ids if mem_id not in existing_ids]
        
        if not new_ids:
            return []
        
        # BATCH FETCH
        rows = self.db.fetch_many(new_ids)
        
        candidates = []
        for mem_id in new_ids:
            row = rows.get(mem_id)
            if row is None:
                continue
            
            memory = self._build_memory_record(row)
            
            embedding = self.embedding_cache.get(mem_id)
            if embedding is None:
                embedding = self.vector_store.get(mem_id)
                if embedding is not None:
                    self.embedding_cache.add(mem_id, embedding)
            
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
    # Combined entry point
    # -------------------------------------

    def retrieve(self, query, ids, distances):
        import time
        
        # Global timer
        t0_global = time.perf_counter()

        # -----------------------------
        # Vector candidates
        # -----------------------------

        t0 = time.perf_counter()
        candidates = self.vector_candidates(ids, distances)
        debug(f"[TIMING] vector_candidates: {(time.perf_counter() - t0)*1000:.2f}ms - {len(candidates)} candidates")
        
        existing_ids = {c.memory.id for c in candidates}

        # -----------------------------
        # Attribute candidates
        # -----------------------------

        t0 = time.perf_counter()
        attribute_rows = self.db.search_attribute(
            query.metadata.get("subject"),
            query.metadata.get("attribute")
        )
        debug(f"[TIMING] search_attribute: {(time.perf_counter() - t0)*1000:.2f}ms - {len(attribute_rows)} rows")

        for row in attribute_rows:
            if row["id"] in existing_ids:
                continue

            memory = self._build_memory_record(row)
            embedding = (
                self.embedding_cache.get(row["id"])
                or self.vector_store.get(row["id"])
            )

            candidates.append(
                CandidateRecord(
                    memory=memory,
                    distance=0.0,
                    embedding=embedding,
                    graph_hit=False
                )
            )
            existing_ids.add(row["id"])

        # -----------------------------
        # Graph candidates
        # -----------------------------

        t0 = time.perf_counter()
        graph_candidates = self.graph_candidates(
            query.entities,
            existing_ids
        )
        debug(f"[TIMING] graph_candidates: {(time.perf_counter() - t0)*1000:.2f}ms - {len(graph_candidates)} candidates")

        graph_limit = getattr(settings, "GRAPH_TOP_K", 50)
        candidates.extend(graph_candidates[:graph_limit])

        total_ms = (time.perf_counter() - t0_global) * 1000
        debug(f"[TIMING] TOTAL retrieve: {total_ms:.2f}ms - {len(candidates)} total candidates")

        return candidates
 

        
