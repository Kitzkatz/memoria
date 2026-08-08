import time
import threading
from typing import List, Dict, Any, Optional
from core.logger import debug
import numpy as np


class Consolidator:
    def __init__(self, db, vector_store, embedding_cache):
        self.db = db
        self.vector_store = vector_store
        self.embedding_cache = embedding_cache
        self._lock = threading.RLock()
        self._running = False
        self._stopping = False

    def run(self, threshold: float = 0.5, batch_size: int = 500, dry_run: bool = False):
        """
        Run consolidation on all memories.
        Process in batches to avoid OOM.
        """
        with self._lock:
            if self._stopping:
                debug("[Consolidator] Interrupted by stop signal")
                return

            debug(f"[Consolidator] Starting consolidation (threshold={threshold}, batch={batch_size})...")
            start = time.perf_counter()

            # Get total count
            total = self.db.count()
            if total < 2:
                debug("[Consolidator] Less than 2 memories, skipping")
                return

            consolidated = 0
            last_id = 0
            clusterer = None

            # Import clusterer lazily
            try:
                from memory.clustering import Clusterer
                clusterer = Clusterer(distance_threshold=threshold)
            except ImportError:
                debug("[Consolidator] Clusterer not available, skipping")
                return

            while True:
                if self._stopping:
                    debug("[Consolidator] Interrupted by stop signal")
                    break

                # Fetch a batch using direct cursor
                conn = self.db.conn
                cur = conn.cursor()
                cur.execute("""
                    SELECT *
                    FROM memories
                    WHERE id > ?
                      AND tombstone = 0
                    ORDER BY id
                    LIMIT ?
                """, (last_id, batch_size))

                batch = cur.fetchall()
                if not batch:
                    break

                # Build embeddings for this batch
                embeddings = []
                memory_ids = []
                batch_memories = []

                for row in batch:
                    mem_id = row["id"]
                    embedding = self.embedding_cache.get(mem_id)
                    if embedding is None:
                        embedding = self.vector_store.get(mem_id)
                        if embedding is not None:
                            self.embedding_cache.add(mem_id, embedding)

                    if embedding is not None:
                        embeddings.append(embedding)
                        memory_ids.append(mem_id)

                        # Convert row to dict for later use
                        import json
                        tokens = json.loads(row["tokens"]) if row["tokens"] else []
                        normalized_text = row["normalized_text"] if row["normalized_text"] is not None else row["text"]
                        batch_memories.append({
                            "id": row["id"],
                            "text": row["text"],
                            "normalized_text": normalized_text,
                            "tokens": tokens,
                            "token_count": row["token_count"] or len(tokens),
                            "memory_type": row["memory_type"],
                            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                            "entities": json.loads(row["entities"]) if row["entities"] else [],
                            "relationships": json.loads(row["relationships"]) if row["relationships"] else [],
                            "importance": float(row["importance"] or 0.5),
                            "created_at": row["created_at"],
                            "last_accessed": row["last_accessed"],
                            "tombstone": row["tombstone"]
                        })

                    last_id = mem_id

                if len(embeddings) < 2:
                    continue

                # Cluster
                clusters = clusterer.cluster(np.array(embeddings), memory_ids)

                # Process clusters
                for cluster_id, cluster_mem_ids in clusters.items():
                    if len(cluster_mem_ids) > 1:
                        if not dry_run:
                            self._merge_cluster(cluster_mem_ids, batch_memories)
                        consolidated += 1

                # If we got fewer than batch_size, we've hit the end
                if len(batch) < batch_size:
                    break

            elapsed = time.perf_counter() - start
            debug(f"[Consolidator] Consolidated {consolidated} clusters in {elapsed:.2f}s{' (dry run)' if dry_run else ''}")

    def _merge_cluster(self, cluster_mem_ids: List[int], all_memories: List[Dict]):
        """Merge a cluster of memories into one."""
        cluster_mems = [m for m in all_memories if m["id"] in cluster_mem_ids]
        if len(cluster_mems) < 2:
            return

        # Use the highest importance memory as base
        best = max(cluster_mems, key=lambda m: m.get("importance", 0.5))

        # Merge entities using a set (deduplicate)
        entities = set(best.get("entities", []))

        # Merge relationships using a set of tuples for deduplication
        relationships_set = set()
        for rel in best.get("relationships", []):
            # Convert dict to a tuple of sorted items for deduplication
            if isinstance(rel, dict):
                rel_tuple = tuple(sorted(rel.items()))
                relationships_set.add(rel_tuple)
            else:
                relationships_set.add((rel,))

        for mem in cluster_mems:
            if mem["id"] == best["id"]:
                continue

            # Merge entities
            for ent in mem.get("entities", []):
                entities.add(ent)

            # Merge relationships
            for rel in mem.get("relationships", []):
                if isinstance(rel, dict):
                    rel_tuple = tuple(sorted(rel.items()))
                    relationships_set.add(rel_tuple)
                else:
                    relationships_set.add((rel,))

        # Convert back to original format
        merged_relationships = []
        for rel_tuple in relationships_set:
            if len(rel_tuple) == 1:
                merged_relationships.append(rel_tuple[0])
            else:
                merged_relationships.append(dict(rel_tuple))

        # Update the best memory with merged data
        self.db.update(
            best["id"],
            entities=list(entities),
            relationships=merged_relationships,
            importance=min(1.0, best.get("importance", 0.5) + 0.1)
        )

        # Soft delete the other memories
        for mem_id in cluster_mem_ids:
            if mem_id != best["id"]:
                self.db.delete(mem_id)

        debug(f"[Consolidator] Merged {len(cluster_mem_ids)} memories into ID {best['id']}")

    def stop(self):
        """Stop any running consolidation."""
        with self._lock:
            self._stopping = True

    def reset_stop(self):
        """Reset the stop flag."""
        with self._lock:
            self._stopping = False
