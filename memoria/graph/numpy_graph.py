# graph/numpy_graph.py
import json
import numpy as np
from collections import deque
from typing import List, Dict, Set, Optional, Tuple, Any
from core.logger import debug


class NumpyGraph:
    def __init__(self, db, entity_store=None):
        self.db = db
        self.entity_store = entity_store

        # Canonical entity IDs from the entities table.
        self.entities: List[int] = []
        self.entity_id: Dict[int, int] = {}

        self.adj_matrix: Optional[np.ndarray] = None
        self.edge_types: Dict[Tuple[int, int], str] = {}
        self.memory_ids: Dict[int, List[int]] = {}
        self._weights: Dict[Tuple[int, int], float] = {}
        self._built = False

        self.build()

    def _resolve_entity_id(self, entity):
        """
        Resolve either a canonical entity ID or a public entity name
        to the canonical entity ID.
        """
        if isinstance(entity, int):
            return entity

        if self.entity_store is None:
            return None

        record = self.entity_store.find(entity)
        return record.id if record else None

    
    def build(self):
        """Build the numpy graph from the graph table."""
        debug("[NumpyGraph] Building graph...")

        with self.db.lock:
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

        entity_ids = set()
        edge_data = []

        for row in rows:
            source = row["source"]
            target = row["target"]
            relation = row["relation"]
            weight = float(row["weight"] or 1.0)
            memory_id = row["memory_id"]

            entity_ids.add(source)
            entity_ids.add(target)
            edge_data.append(
                (source, target, relation, weight, memory_id)
            )

        # Canonical entity IDs are used internally.
        # Sorting gives deterministic node indexes.
        for entity_id in sorted(entity_ids):
            self.entity_id[entity_id] = len(self.entities)
            self.entities.append(entity_id)

        n = len(self.entities)

        self.adj_matrix = np.zeros(
            (n, n),
            dtype=np.float32,
        )

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

        self.memory_ids = {
            key: list(value)
            for key, value in memory_ids.items()
        }

        self._built = True

        debug(
            f"[NumpyGraph] Built: {len(self.entities)} entities, "
            f"{np.count_nonzero(self.adj_matrix)} edges"
        )


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
            "version": 3,
            "entities": self.entities,
            "entity_id": {
                str(k): v
                for k, v in self.entity_id.items()
            },
            "adj_matrix": (
                self.adj_matrix.tolist()
                if self.adj_matrix is not None
                else None
            ),
            "edge_types": {
                f"{k[0]},{k[1]}": v
                for k, v in self.edge_types.items()
            },
            "memory_ids": {
                str(k): v
                for k, v in self.memory_ids.items()
            },
            "_weights": {
                f"{k[0]},{k[1]}": v
                for k, v in self._weights.items()
            },
            "_built": self._built,
        }

    def from_dict(self, data: Dict[str, Any]):
        """Restore the graph from a dictionary."""
        self.entities = [
            int(entity_id)
            for entity_id in data.get("entities", [])
        ]

        # JSON object keys are strings, so restore canonical integer IDs.
        self.entity_id = {
            int(k): int(v)
            for k, v in data.get("entity_id", {}).items()
        }

        self.adj_matrix = (
            np.array(data["adj_matrix"], dtype=np.float32)
            if data.get("adj_matrix") is not None
            else None
        )

        self.edge_types = {
            tuple(map(int, k.split(","))): v
            for k, v in data.get("edge_types", {}).items()
        }

        self.memory_ids = {
            int(k): v
            for k, v in data.get("memory_ids", {}).items()
        }

        self._weights = {
            tuple(map(int, k.split(","))): v
            for k, v in data.get("_weights", {}).items()
        }

        self._built = data.get("_built", True)

        debug(
            f"[NumpyGraph] Restored from dict: "
            f"{len(self.entities)} entities"
        )

    def save(self, filepath: str):
        """Save the graph as JSON (safe, portable)."""
        data = self.to_dict()

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False
            )

        debug(f"[NumpyGraph] Saved to {filepath} (JSON)")

    def load(self, filepath: str):
        """Load the graph from a JSON file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.from_dict(data)

            debug(
                f"[NumpyGraph] Loaded from {filepath} (JSON)"
            )
            return True

        except FileNotFoundError:
            debug(
                f"[NumpyGraph] File not found: {filepath}"
            )
            return False

        except json.JSONDecodeError:
            debug(
                "[NumpyGraph] JSON decode failed, "
                "trying pickle fallback..."
            )
            return self._load_pickle(filepath)

    def _load_pickle(self, filepath: str) -> bool:
        """Legacy pickle loader (for backward compatibility)."""
        import pickle

        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)

            self.entities = [
                int(entity_id)
                for entity_id in data["entities"]
            ]

            self.entity_id = {
                int(k): int(v)
                for k, v in data["entity_id"].items()
            }

            self.adj_matrix = data["adj_matrix"]
            self.edge_types = data["edge_types"]
            self.memory_ids = {
                int(k): v
                for k, v in data["memory_ids"].items()
            }
            self._weights = data.get("_weights", {})
            self._built = data.get("_built", True)

            debug(
                f"[NumpyGraph] Loaded from {filepath} "
                "(pickle legacy)"
            )
            return True

        except Exception as e:
            debug(
                f"[NumpyGraph] Legacy pickle load failed: {e}"
            )
            return False

    # -------------------------
    # Graph Operations
    # -------------------------

    def neighbors(
        self,
        entity,
        depth: int = 1,
        limit: int = 100
    ) -> List[int]:
        if not self._built or not self.entities:
            return []

        canonical_id = self._resolve_entity_id(entity)
        if canonical_id is None:
            return []

        idx = self.entity_id.get(canonical_id)
        if idx is None:
            return []

        visited = {idx}
        frontier = {idx}

        for _ in range(depth):
            if not frontier:
                break

            frontier_list = list(frontier)

            mask = (
                self.adj_matrix[frontier_list, :] > 0
            )

            new_nodes = set(
                np.where(mask.any(axis=0))[0]
            )

            frontier = new_nodes - visited
            visited.update(frontier)

            if len(visited) > limit:
                break

        result = [
            self.entities[i]
            for i in visited
            if i != idx
        ]

        return result[:limit]

    def get_memory_ids_for_entity(
        self,
        entity
    ) -> List[int]:
        if not self._built:
            return []

        canonical_id = self._resolve_entity_id(entity)
        if canonical_id is None:
            return []

        idx = self.entity_id.get(canonical_id)
        if idx is None:
            return []

        return self.memory_ids.get(idx, [])

    def multi_hop_search(
        self,
        entity_names: List[Any],
        depth: int = 2,
        limit: int = 200
    ) -> List[int]:
        if not self._built or not entity_names:
            return []

        memory_ids = set()

        for entity in entity_names:
            canonical_id = self._resolve_entity_id(entity)

            if canonical_id is None:
                continue

            idx = self.entity_id.get(canonical_id)

            if idx is None:
                continue

            memory_ids.update(
                self.memory_ids.get(idx, [])
            )

            if depth > 0:
                neighbors = self.neighbors(
                    canonical_id,
                    depth=depth,
                    limit=limit
                )

                for neighbor_id in neighbors:
                    n_idx = self.entity_id.get(
                        neighbor_id
                    )

                    if n_idx is not None:
                        memory_ids.update(
                            self.memory_ids.get(
                                n_idx,
                                []
                            )
                        )

            if len(memory_ids) >= limit:
                break

        return list(memory_ids)[:limit]

    def shortest_path(
        self,
        src,
        dst
    ) -> float:
        if not self._built:
            return float("inf")

        src_id = self._resolve_entity_id(src)
        dst_id = self._resolve_entity_id(dst)

        if src_id is None or dst_id is None:
            return float("inf")

        src_idx = self.entity_id.get(src_id)
        dst_idx = self.entity_id.get(dst_id)

        if src_idx is None or dst_idx is None:
            return float("inf")

        if src_idx == dst_idx:
            return 0.0

        visited = {src_idx}
        queue = deque([(src_idx, 0)])

        while queue:
            current, dist = queue.popleft()

            neighbors = np.where(
                self.adj_matrix[current, :] > 0
            )[0]

            for nb in neighbors:
                if nb == dst_idx:
                    return dist + 1

                if nb not in visited:
                    visited.add(nb)
                    queue.append(
                        (nb, dist + 1)
                    )

        return float("inf")

    def shortest_path_weighted(
        self,
        src,
        dst
    ) -> float:
        """Dijkstra using edge weights."""
        if not self._built:
            return float("inf")

        src_id = self._resolve_entity_id(src)
        dst_id = self._resolve_entity_id(dst)

        if src_id is None or dst_id is None:
            return float("inf")

        src_idx = self.entity_id.get(src_id)
        dst_idx = self.entity_id.get(dst_id)

        if src_idx is None or dst_idx is None:
            return float("inf")

        if src_idx == dst_idx:
            return 0.0

        import heapq

        n = len(self.entities)
        dist = {
            i: float("inf")
            for i in range(n)
        }

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

            neighbors = np.where(
                self.adj_matrix[u, :] > 0
            )[0]

            for v in neighbors:
                weight = self._weights.get(
                    (u, v),
                    1.0
                )

                nd = d + weight

                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(
                        pq,
                        (nd, v)
                    )

        return float("inf")

    # -------------------------
    # Export / Diagnostics
    # -------------------------

    def explain(
        self,
        entity,
        depth: int = 2,
        limit: int = 10
    ):
        if not self._built:
            debug("[NumpyGraph] Graph not built")
            return

        canonical_id = self._resolve_entity_id(entity)

        debug(
            f"Graph explanation for '{entity}' "
            f"(depth={depth}):"
        )

        if canonical_id is None:
            debug("  Entity not found")
            return

        idx = self.entity_id.get(canonical_id)

        if idx is None:
            debug("  Entity not found")
            return

        debug(
            f"  Total entities: {len(self.entities)}"
        )
        debug(
            f"  Total edges: "
            f"{np.count_nonzero(self.adj_matrix)}"
        )

        visited = {idx}
        frontier = {idx}

        for level in range(depth):
            if not frontier:
                break

            mask = (
                self.adj_matrix[
                    list(frontier),
                    :
                ] > 0
            )

            new_nodes = set(
                np.where(mask.any(axis=0))[0]
            )

            frontier = new_nodes - visited

            for node in frontier:
                for src in visited:
                    if self.adj_matrix[src, node] > 0:
                        relation = self.edge_types.get(
                            (src, node),
                            "related"
                        )

                        weight = self.adj_matrix[
                            src,
                            node
                        ]

                        debug(
                            f"  {self.entities[src]} "
                            f"--{relation} "
                            f"(w={weight:.2f})--> "
                            f"{self.entities[node]}"
                        )

            visited.update(frontier)

            if len(visited) >= limit:
                break

    def to_cytoscape(self) -> dict:
        if not self._built:
            return {
                "nodes": [],
                "edges": []
            }

        nodes = [
            {
                "id": i,
                "label": str(entity_id)
            }
            for i, entity_id in enumerate(self.entities)
        ]

        edges = []

        for (src, dst), rel in self.edge_types.items():
            edges.append({
                "source": src,
                "target": dst,
                "label": rel,
                "weight": float(
                    self.adj_matrix[src, dst]
                )
            })

        return {
            "nodes": nodes,
            "edges": edges
        }

    def stats(self) -> dict:
        if not self._built:
            return {"built": False}

        return {
            "built": True,
            "entities": len(self.entities),
            "edges": int(
                np.count_nonzero(self.adj_matrix)
            ),
            "edge_types": len(
                set(self.edge_types.values())
            ),
            "entities_with_memories": len(
                self.memory_ids
            ),
            "total_memory_mappings": sum(
                len(v)
                for v in self.memory_ids.values()
            ),
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

        return int(
            np.count_nonzero(self.adj_matrix)
        )
