# memory/clustering.py
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from typing import List, Dict, Any, Optional

class Clusterer:
    def __init__(self, distance_threshold: float = 0.5, linkage_method: str = "ward"):
        self.distance_threshold = distance_threshold
        self.linkage_method = linkage_method
        self.clusters = {}

    def build_embeddings(self, memories: List[Dict[str, Any]]) -> np.ndarray:
        """Build embedding matrix from memory records."""
        embeddings = []
        for mem in memories:
            # Use the embedding stored in FAISS or compute from text
            # For now, we'll assume embeddings are available
            # This is a placeholder — you'll need to fetch actual embeddings
            embedding = mem.get("embedding")
            if embedding is not None:
                embeddings.append(embedding)
        return np.array(embeddings)

    def cluster(self, embeddings: np.ndarray, memory_ids: List[int]) -> Dict[int, List[int]]:
        """Cluster memories based on embeddings."""
        if len(embeddings) < 2:
            return {mid: [mid] for mid in memory_ids}

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

    def consolidate_cluster(self, cluster_mem_ids: List[int], memories: List[Dict]) -> Dict[str, Any]:
        """Merge a cluster into a single memory."""
        if not cluster_mem_ids:
            return {}

        # Find the highest importance memory as base
        best_mem = max(memories, key=lambda m: m.get("importance", 0.5))

        # Merge metadata, entities, relationships
        entities = set(best_mem.get("entities", []))
        relationships = list(best_mem.get("relationships", []))
        texts = [best_mem.get("text", "")]

        for mem_id in cluster_mem_ids:
            if mem_id == best_mem.get("id"):
                continue
            mem = next((m for m in memories if m.get("id") == mem_id), None)
            if mem:
                entities.update(mem.get("entities", []))
                for rel in mem.get("relationships", []):
                    if rel not in relationships:
                        relationships.append(rel)
                texts.append(mem.get("text", ""))

        # Create consolidated record
        return {
            "id": best_mem.get("id"),
            "text": " | ".join(texts),
            "entities": list(entities),
            "relationships": relationships,
            "importance": best_mem.get("importance", 0.5),
            "metadata": best_mem.get("metadata", {}),
            "cluster_size": len(cluster_mem_ids),
            "cluster_members": cluster_mem_ids,
        }
