# graph/numpy_graph.py
import json
import numpy as np
from collections import deque
from typing import List, Dict, Set, Optional, Tuple, Any
from core.logger import debug


class NumpyGraph:
    def __init__(self, db):
        self.db = db
        self.entities: List[str] = []
        self.entity_id: Dict[str, int] = {}
        self.adj_matrix: Optional[np.ndarray] = None
        self.edge_types: Dict[Tuple[int, int], str] = {}
        self.memory_ids: Dict[int, List[int]] = {}
        self._weights: Dict[Tuple[int, int], float] = {}
        self._built = False
        self.build()

    def build(self):
        """Build the numpy graph from the graph table."""
        debug("[NumpyGraph] Building graph...")
        cur = self.db.conn.cursor()
        cur.execute("""
            SELECT source, target, relation, weight, memory_id
            FROM graph
        """)
        rows = cur.fetchall()

        if not rows:
            debug("[NumpyGraph] No graph data found")
            self._built = True
            return

        entity_names = set()
        edge_data = []
        for row in rows:
            source = row["source"]
            target = row["target"]
            relation = row["relation"]
            weight = float(row["weight"] or 1.0)
            memory_id = row["memory_id"]

            entity_names.add(source)
            entity_names.add(target)
            edge_data.append((source, target, relation, weight, memory_id))

        # Assign IDs (sorted for reproducibility)
        for name in sorted(entity_names):
            self.entity_id[name] = len(self.entities)
            self.entities.append(name)

        n = len(self.entities)
        self.adj_matrix = np.zeros((n, n), dtype=np.float32)
        memory_ids: Dict[int, Set[int]] = {}

        for source, target, relation, weight, memory_id in edge_data:
            src_idx = self.entity_id.get(source)
            dst_idx = self.entity_id.get(target)
            if src_idx is None or dst_idx is None:
                continue

            self.adj_matrix[src_idx, dst_idx] = weight
            self.edge_types[(src_idx, dst_idx)] = relation
            self._weights[(src_idx, dst_idx)] = weight

            if src_idx not in memory_ids:
                memory_ids[src_idx] = set()
            memory_ids[src_idx].add(memory_id)

            if dst_idx not in memory_ids:
                memory_ids[dst_idx] = set()
            memory_ids[dst_idx].add(memory_id)

        self.memory_ids = {k: list(v) for k, v in memory_ids.items()}
        self._built = True
        debug(f"[NumpyGraph] Built: {len(self.entities)} entities, {np.count_nonzero(self.adj_matrix)} edges")

    def rebuild(self):
        """Rebuild the graph from the database."""
        self.entities = []
        self.entity_id = {}
        self.adj_matrix = None
        self.edge_types = {}
        self.memory_ids = {}
        self._weights = {}
        self._built = False
        self.build()

    # -------------------------
    # Safe Serialization (JSON)
    # -------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert the graph to a JSON-serializable dictionary."""
        return {
            "version": 2,
            "entities": self.entities,
            "entity_id": self.entity_id,
            "adj_matrix": self.adj_matrix.tolist() if self.adj_matrix is not None else None,
            "edge_types": {f"{k[0]},{k[1]}": v for k, v in self.edge_types.items()},
            "memory_ids": {str(k): v for k, v in self.memory_ids.items()},
            "_weights": {f"{k[0]},{k[1]}": v for k, v in self._weights.items()},
            "_built": self._built,
        }

    def from_dict(self, data: Dict[str, Any]):
        """Restore the graph from a dictionary."""
        self.entities = data["entities"]
        self.entity_id = data["entity_id"]
        self.adj_matrix = np.array(data["adj_matrix"], dtype=np.float32) if data["adj_matrix"] is not None else None
        self.edge_types = {tuple(map(int, k.split(","))): v for k, v in data["edge_types"].items()}
        self.memory_ids = {int(k): v for k, v in data["memory_ids"].items()}
        self._weights = {tuple(map(int, k.split(","))): v for k, v in data.get("_weights", {}).items()}
        self._built = data.get("_built", True)
        debug(f"[NumpyGraph] Restored from dict: {len(self.entities)} entities")

    def save(self, filepath: str):
        """Save the graph as JSON (safe, portable)."""
        data = self.to_dict()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        debug(f"[NumpyGraph] Saved to {filepath} (JSON)")

    def load(self, filepath: str):
        """Load the graph from a JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.from_dict(data)
            debug(f"[NumpyGraph] Loaded from {filepath} (JSON)")
            return True
        except FileNotFoundError:
            debug(f"[NumpyGraph] File not found: {filepath}")
            return False
        except json.JSONDecodeError:
            # Fallback: try to load as pickle (legacy)
            debug(f"[NumpyGraph] JSON decode failed, trying pickle fallback...")
            return self._load_pickle(filepath)

    def _load_pickle(self, filepath: str) -> bool:
        """Legacy pickle loader (for backward compatibility)."""
        import pickle
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.entities = data['entities']
            self.entity_id = data['entity_id']
            self.adj_matrix = data['adj_matrix']
            self.edge_types = data['edge_types']
            self.memory_ids = data['memory_ids']
            self._weights = data.get('_weights', {})
            self._built = data.get('_built', True)
            debug(f"[NumpyGraph] Loaded from {filepath} (pickle legacy)")
            return True
        except Exception as e:
            debug(f"[NumpyGraph] Legacy pickle load failed: {e}")
            return False

    # -------------------------
    # Graph Operations
    # -------------------------

    def neighbors(self, entity_name: str, depth: int = 1, limit: int = 100) -> List[str]:
        if not self._built or not self.entities:
            return []
        idx = self.entity_id.get(entity_name)
        if idx is None:
            return []

        visited = {idx}
        frontier = {idx}
        for _ in range(depth):
            if not frontier:
                break
            frontier_list = list(frontier)
            mask = self.adj_matrix[frontier_list, :] > 0
            new_nodes = set(np.where(mask.any(axis=0))[0])
            frontier = new_nodes - visited
            visited.update(frontier)
            if len(visited) > limit:
                break

        result = [self.entities[i] for i in visited if i != idx]
        return result[:limit]

    def get_memory_ids_for_entity(self, entity_name: str) -> List[int]:
        if not self._built:
            return []
        idx = self.entity_id.get(entity_name)
        if idx is None:
            return []
        return self.memory_ids.get(idx, [])

    def multi_hop_search(self, entity_names: List[str], depth: int = 2, limit: int = 200) -> List[int]:
        if not self._built or not entity_names:
            return []

        memory_ids = set()
        for name in entity_names:
            idx = self.entity_id.get(name)
            if idx is None:
                continue
            memory_ids.update(self.memory_ids.get(idx, []))
            if depth > 0:
                neighbors = self.neighbors(name, depth=depth, limit=limit)
                for neighbor in neighbors:
                    n_idx = self.entity_id.get(neighbor)
                    if n_idx is not None:
                        memory_ids.update(self.memory_ids.get(n_idx, []))
            if len(memory_ids) >= limit:
                break

        return list(memory_ids)[:limit]

    def shortest_path(self, src: str, dst: str) -> float:
        if not self._built:
            return float('inf')
        src_idx = self.entity_id.get(src)
        dst_idx = self.entity_id.get(dst)
        if src_idx is None or dst_idx is None:
            return float('inf')
        if src_idx == dst_idx:
            return 0.0

        visited = {src_idx}
        queue = deque([(src_idx, 0)])
        while queue:
            current, dist = queue.popleft()
            neighbors = np.where(self.adj_matrix[current, :] > 0)[0]
            for nb in neighbors:
                if nb == dst_idx:
                    return dist + 1
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))
        return float('inf')

    def shortest_path_weighted(self, src: str, dst: str) -> float:
        """Dijkstra using edge weights."""
        if not self._built:
            return float('inf')
        src_idx = self.entity_id.get(src)
        dst_idx = self.entity_id.get(dst)
        if src_idx is None or dst_idx is None:
            return float('inf')
        if src_idx == dst_idx:
            return 0.0

        import heapq
        n = len(self.entities)
        dist = {i: float('inf') for i in range(n)}
        dist[src_idx] = 0
        pq = [(0, src_idx)]
        visited = set()

        while pq:
            d, u = heapq.heappop(pq)
            if u == dst_idx:
                return d
            if u in visited:
                continue
            visited.add(u)

            neighbors = np.where(self.adj_matrix[u, :] > 0)[0]
            for v in neighbors:
                weight = self._weights.get((u, v), 1.0)
                nd = d + weight
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))

        return float('inf')

    # -------------------------
    # Export / Diagnostics
    # -------------------------

    def explain(self, entity_name: str, depth: int = 2, limit: int = 10):
        if not self._built:
            debug("[NumpyGraph] Graph not built")
            return

        debug(f"Graph explanation for '{entity_name}' (depth={depth}):")
        idx = self.entity_id.get(entity_name)
        if idx is None:
            debug("  Entity not found")
            return

        debug(f"  Total entities: {len(self.entities)}")
        debug(f"  Total edges: {np.count_nonzero(self.adj_matrix)}")

        visited = {idx}
        frontier = {idx}
        for level in range(depth):
            if not frontier:
                break
            mask = self.adj_matrix[list(frontier), :] > 0
            new_nodes = set(np.where(mask.any(axis=0))[0])
            frontier = new_nodes - visited
            for node in frontier:
                for src in visited:
                    if self.adj_matrix[src, node] > 0:
                        relation = self.edge_types.get((src, node), "related")
                        weight = self.adj_matrix[src, node]
                        debug(f"  {self.entities[src]} --{relation} (w={weight:.2f})--> {self.entities[node]}")
            visited.update(frontier)

    def to_cytoscape(self) -> dict:
        if not self._built:
            return {"nodes": [], "edges": []}

        nodes = [{"id": i, "label": name} for i, name in enumerate(self.entities)]
        edges = []
        for (src, dst), rel in self.edge_types.items():
            edges.append({
                "source": src,
                "target": dst,
                "label": rel,
                "weight": float(self.adj_matrix[src, dst])
            })
        return {"nodes": nodes, "edges": edges}

    def stats(self) -> dict:
        if not self._built:
            return {"built": False}
        return {
            "built": True,
            "entities": len(self.entities),
            "edges": int(np.count_nonzero(self.adj_matrix)),
            "edge_types": len(set(self.edge_types.values())),
            "entities_with_memories": len(self.memory_ids),
            "total_memory_mappings": sum(len(v) for v in self.memory_ids.values()),
        }

    @property
    def built(self) -> bool:
        return self._built

    @property
    def num_entities(self) -> int:
        return len(self.entities)

    @property
    def num_edges(self) -> int:
        if not self._built:
            return 0
        return int(np.count_nonzero(self.adj_matrix))
