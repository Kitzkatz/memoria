import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from typing import List, Dict, Any, Optional

from core.logger import debug


class Clusterer:
    def __init__(self, distance_threshold: float = 0.5, linkage_method: str = "ward"):
        self.distance_threshold = distance_threshold
        self.linkage_method = linkage_method
        self.clusters = {}

    def cluster(self, embeddings: np.ndarray, memory_ids: List[int]) -> Dict[int, List[int]]:
        """
        Cluster memories based on embeddings.
        Expects embeddings to already be built/retrieved.
        """
        if len(embeddings) < 2:
            # Return each memory as its own cluster
            return {i: [mid] for i, mid in enumerate(memory_ids)}

        # Check if all embeddings are valid
        if embeddings.shape[0] != len(memory_ids):
            debug(f"[Clusterer] Mismatch: {embeddings.shape[0]} embeddings, {len(memory_ids)} memory IDs")
            return {i: [mid] for i, mid in enumerate(memory_ids)}

        # If batch size is large, warn about O(n²)
        if len(embeddings) > 2000:
            debug(f"[Clusterer] Large batch ({len(embeddings)} embeddings) — O(n²) clustering may be slow")

        try:
            # Compute distance matrix
            dist_matrix = pdist(embeddings, metric="cosine")
            linkage_matrix = linkage(dist_matrix, method=self.linkage_method)

            # Form flat clusters
            labels = fcluster(linkage_matrix, t=self.distance_threshold, criterion="distance")

            # Group by cluster label
            clusters = {}
            for mid, label in zip(memory_ids, labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(mid)

            return clusters

        except Exception as e:
            debug(f"[Clusterer] Clustering failed: {e} — returning singleton clusters")
            return {i: [mid] for i, mid in enumerate(memory_ids)}

    # ---------------------------------
    # Deprecated/Unused
    # ---------------------------------

    def build_embeddings(self, memories: List[Dict[str, Any]]) -> np.ndarray:
        """
        DEPRECATED: Use embedding_cache.get() or vector_store.get() directly.
        This method is kept for backward compatibility but does nothing useful.
        """
        debug("[Clusterer] build_embeddings() is deprecated — embeddings should be retrieved via cache/vector store")
        return np.array([])

    def consolidate_cluster(self, cluster_mem_ids: List[int], memories: List[Dict]) -> Dict[str, Any]:
        """
        Merge a cluster into a single memory.
        This is a convenience wrapper around the consolidator's logic.
        """
        if not cluster_mem_ids:
            return {}

        # Find the highest importance memory as base
        best_mem = max(
            [m for m in memories if m.get("id") in cluster_mem_ids],
            key=lambda m: m.get("importance", 0.5),
            default=None
        )

        if best_mem is None:
            return {}

        # Merge entities using set
        entities = set(best_mem.get("entities", []))

        # Merge relationships using tuple deduplication
        relationships_set = set()
        for rel in best_mem.get("relationships", []):
            if isinstance(rel, dict):
                relationships_set.add(tuple(sorted(rel.items())))
            else:
                relationships_set.add((rel,))

        texts = [best_mem.get("text", "")]

        for mem_id in cluster_mem_ids:
            if mem_id == best_mem.get("id"):
                continue
            mem = next((m for m in memories if m.get("id") == mem_id), None)
            if mem:
                for ent in mem.get("entities", []):
                    entities.add(ent)

                for rel in mem.get("relationships", []):
                    if isinstance(rel, dict):
                        relationships_set.add(tuple(sorted(rel.items())))
                    else:
                        relationships_set.add((rel,))

                texts.append(mem.get("text", ""))

        # Convert relationships back to original format
        merged_relationships = []
        for rel_tuple in relationships_set:
            if len(rel_tuple) == 1:
                merged_relationships.append(rel_tuple[0])
            else:
                merged_relationships.append(dict(rel_tuple))

        return {
            "id": best_mem.get("id"),
            "text": " | ".join(texts),
            "entities": list(entities),
            "relationships": merged_relationships,
            "importance": min(1.0, best_mem.get("importance", 0.5) + 0.1),
            "metadata": best_mem.get("metadata", {}),
            "cluster_size": len(cluster_mem_ids),
            "cluster_members": cluster_mem_ids,
        }
