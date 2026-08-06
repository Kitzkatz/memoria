# blackboard/consolidator.py
import time
from typing import List, Dict, Any
from core.logger import debug
import numpy as np

class Consolidator:
    def __init__(self, db, vector_store, embedding_cache):
        self.db = db
        self.vector_store = vector_store
        self.embedding_cache = embedding_cache

    def run(self, threshold: float = 0.5, batch_size: int = 500):
        """Run consolidation on all memories."""
        debug("[Consolidator] Starting consolidation...")
        start = time.perf_counter()

        # Fetch all memories
        memories = self.db.fetch_all()
        if len(memories) < 2:
            return

        # Build embeddings for each memory
        embeddings = []
        memory_ids = []
        for mem in memories:
            mem_id = mem["id"]
            embedding = self.embedding_cache.get(mem_id)
            if embedding is None:
                embedding = self.vector_store.get(mem_id)
                if embedding is not None:
                    self.embedding_cache.add(mem_id, embedding)
            if embedding is not None:
                embeddings.append(embedding)
                memory_ids.append(mem_id)

        if len(embeddings) < 2:
            return

        # Cluster
        from memory.clustering import Clusterer
        clusterer = Clusterer(distance_threshold=threshold)
        clusters = clusterer.cluster(np.array(embeddings), memory_ids)

        # Consolidate clusters with >1 member
        consolidated = 0
        for cluster_id, cluster_mem_ids in clusters.items():
            if len(cluster_mem_ids) > 1:
                self._merge_cluster(cluster_mem_ids, memories)
                consolidated += 1

        elapsed = time.perf_counter() - start
        debug(f"[Consolidator] Consolidated {consolidated} clusters in {elapsed:.2f}s")

    def _merge_cluster(self, cluster_mem_ids: List[int], all_memories: List[Dict]):
        """Merge a cluster of memories into one."""
        # Find the best memory in the cluster
        cluster_mems = [m for m in all_memories if m["id"] in cluster_mem_ids]
        if not cluster_mems:
            return

        # Use the highest importance memory as base
        best = max(cluster_mems, key=lambda m: m.get("importance", 0.5))

        # Merge entities and relationships
        entities = set(best.get("entities", []))
        relationships = list(best.get("relationships", []))

        for mem in cluster_mems:
            if mem["id"] == best["id"]:
                continue
            entities.update(mem.get("entities", []))
            for rel in mem.get("relationships", []):
                if rel not in relationships:
                    relationships.append(rel)

        # Update the best memory with merged data
        self.db.update(
            best["id"],
            entities=list(entities),
            relationships=relationships,
            importance=best.get("importance", 0.5) + 0.1
        )

        # Soft delete the other memories
        for mem_id in cluster_mem_ids:
            if mem_id != best["id"]:
                self.db.delete(mem_id)

        debug(f"[Consolidator] Merged {len(cluster_mem_ids)} memories into ID {best['id']}")
