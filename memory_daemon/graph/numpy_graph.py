# graph/numpy_graph.py
import numpy as np
from collections import deque
from typing import List, Dict, Set, Optional, Tuple

class NumpyGraph:
    def __init__(self, db):
        self.db = db
        self.entities: List[str] = []
        self.entity_id: Dict[str, int] = {}
        self.adj_matrix: Optional[np.ndarray] = None
        self.edge_types: Dict[Tuple[int, int], str] = {}
        self.memory_ids: Dict[int, List[int]] = {}  # entity_id -> list of memory_ids
        self.build()

    def build(self):
        """Build the numpy graph from the graph table."""
        cur = self.db.conn.cursor()
        cur.execute("""
            SELECT source, target, relation, weight, memory_id
            FROM graph
        """)
        rows = cur.fetchall()

        # First pass: collect all unique entity names
        entity_names = set()
        edge_data = []
        for row in rows:
            source = row["source"]
            target = row["target"]
            relation = row["relation"]
            weight = row["weight"] or 1.0
            memory_id = row["memory_id"]

            entity_names.add(source)
            entity_names.add(target)
            edge_data.append((source, target, relation, weight, memory_id))

        # Assign IDs to entities
        for name in sorted(entity_names):
            self.entity_id[name] = len(self.entities)
            self.entities.append(name)

        n = len(self.entities)
        self.adj_matrix = np.zeros((n, n), dtype=np.float32)

        # Second pass: build adjacency matrix and memory mappings
        for source, target, relation, weight, memory_id in edge_data:
            src_idx = self.entity_id.get(source)
            dst_idx = self.entity_id.get(target)
            if src_idx is not None and dst_idx is not None:
                self.adj_matrix[src_idx, dst_idx] = weight
                self.edge_types[(src_idx, dst_idx)] = relation
                # Store memory_id mapping
                if src_idx not in self.memory_ids:
                    self.memory_ids[src_idx] = []
                if memory_id not in self.memory_ids[src_idx]:
                    self.memory_ids[src_idx].append(memory_id)
                if dst_idx not in self.memory_ids:
                    self.memory_ids[dst_idx] = []
                if memory_id not in self.memory_ids[dst_idx]:
                    self.memory_ids[dst_idx].append(memory_id)

    def neighbors(self, entity_name: str, depth: int = 1, limit: int = 100) -> List[str]:
        """Return entity names reachable within depth steps."""
        idx = self.entity_id.get(entity_name)
        if idx is None:
            return []
        visited = {idx}
        frontier = {idx}
        for _ in range(depth):
            if not frontier:
                break
            mask = self.adj_matrix[list(frontier), :] > 0
            new_nodes = set(np.where(mask.any(axis=0))[0])
            frontier = new_nodes - visited
            visited.update(frontier)
            if len(visited) > limit:
                break
        return [self.entities[i] for i in visited if i != idx][:limit]

    def get_memory_ids_for_entity(self, entity_name: str) -> List[int]:
        """Return memory IDs associated with an entity."""
        idx = self.entity_id.get(entity_name)
        if idx is None:
            return []
        return self.memory_ids.get(idx, [])

    def multi_hop_search(self, entity_names: List[str], depth: int = 2, limit: int = 200) -> List[int]:
        """Return memory IDs reachable within depth from any entity."""
        memory_ids = set()
        for name in entity_names:
            idx = self.entity_id.get(name)
            if idx is None:
                continue
            # Direct memories
            memory_ids.update(self.memory_ids.get(idx, []))
            # Neighbor memories
            neighbors = self.neighbors(name, depth=depth, limit=limit)
            for neighbor in neighbors:
                n_idx = self.entity_id.get(neighbor)
                if n_idx is not None:
                    memory_ids.update(self.memory_ids.get(n_idx, []))
        return list(memory_ids)[:limit]

    def shortest_path(self, src: str, dst: str) -> float:
        """Return shortest path distance (in edges) between two entities."""
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

    def explain(self, entity_name: str, depth: int = 2, limit: int = 10):
        """Print a readable explanation of the graph traversal."""
        print(f"Graph explanation for '{entity_name}' (depth={depth}):")
        idx = self.entity_id.get(entity_name)
        if idx is None:
            print("  Entity not found")
            return
        print(f"  Entity index: {idx}")
        print(f"  Total entities: {len(self.entities)}")
        print(f"  Total edges: {np.count_nonzero(self.adj_matrix)}")
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
                        print(f"  {self.entities[src]} --{relation}--> {self.entities[node]}")
            visited.update(frontier)

    def to_cytoscape(self) -> dict:
        """Export graph to Cytoscape JSON format."""
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

    def save(self, filepath: str):
        """Save the graph to a file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'entities': self.entities,
                'entity_id': self.entity_id,
                'adj_matrix': self.adj_matrix,
                'edge_types': self.edge_types,
                'memory_ids': self.memory_ids
            }, f)

    def load(self, filepath: str):
        """Load the graph from a file."""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.entities = data['entities']
        self.entity_id = data['entity_id']
        self.adj_matrix = data['adj_matrix']
        self.edge_types = data['edge_types']
        self.memory_ids = data['memory_ids']
