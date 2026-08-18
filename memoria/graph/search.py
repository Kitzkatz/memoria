from collections import deque
from typing import List, Set, Optional, Any
from core.logger import debug


class GraphSearch:

    def __init__(self, edge_store, entity_store, numpy_graph=None):
        self.edge_store = edge_store
        self.entity_store = entity_store
        self.numpy_graph = numpy_graph  # optional fast graph

    def find_entity(self, name):
        return self.entity_store.find(name)

    def neighbors(self, entity_name: str, depth: int = 1):
        """
        Return related entities and relations up to depth.
        Uses edge store; for fast lookups consider using numpy_graph.
        """
        if not entity_name:
            return []

        visited = set()
        results = []
        queue = deque()
        queue.append((entity_name, 0))

        while queue:
            current, level = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            if level >= depth:
                continue

            # Fetch edges where current appears as source or target
            edges = self.edge_store.fetch_edges_by_entity(current)
            for edge in edges:
                # Determine the other entity
                other = edge.target if edge.source == current else edge.source
                if other in visited:
                    continue
                results.append({
                    "entity_name": other,
                    "relation": edge.relation,
                    "source": edge.source,
                    "target": edge.target
                })
                queue.append((other, level + 1))

        return results

    def entity_memories(self, entity_name: str) -> List[int]:
        """Get all memory IDs associated with an entity."""
        return self.edge_store.get_memory_ids_for_entity(entity_name)

    def search(self, entities: List[str], depth: int = 1, limit: int = 200) -> List[int]:
        """
        Search for memory IDs related to a list of entities up to given depth.
        Uses numpy_graph if available for speed, otherwise falls back to edge store.
        """
        if not entities:
            return []

        debug(f"GraphSearch: entities={entities}, depth={depth}, limit={limit}", category="graph")

        # If numpy_graph is available and depth <= 2, use it for speed
        if self.numpy_graph and self.numpy_graph.built and depth <= 2:
            memory_ids = self.numpy_graph.multi_hop_search(entities, depth=depth, limit=limit)
            debug(f"GraphSearch: numpy_graph returned {len(memory_ids)} memories", category="graph")
            return memory_ids

        # Fallback: use edge store traversal
        memory_ids = set()
        processed = set()
        queue = deque(entities)

        # Pre-load entities to avoid repeated lookups
        entity_cache = {}

        while queue and len(memory_ids) < limit:
            name = queue.popleft()
            if name in processed:
                continue
            processed.add(name)

            # Get entity from cache or lookup
            if name not in entity_cache:
                entity = self.find_entity(name)
                if not entity:
                    continue
                entity_cache[name] = entity
            else:
                entity = entity_cache[name]

            # Direct memories
            for mem_id in self.entity_memories(name):
                memory_ids.add(mem_id)
                if len(memory_ids) >= limit:
                    return list(memory_ids)[:limit]

            # If depth > 0, traverse neighbors
            if depth > 0:
                neighbor_entities = self._get_neighbors_at_depth_with_cache(name, depth, processed, entity_cache)
                for neighbor_name in neighbor_entities:
                    if neighbor_name not in processed:
                        queue.append(neighbor_name)

        return list(memory_ids)[:limit]

    def _get_neighbors_at_depth(self, entity_name: str, depth: int) -> List[str]:
        """Get all entity names reachable within depth (excluding self)."""
        if not entity_name or depth <= 0:
            return []

        visited = {entity_name}
        frontier = {entity_name}

        for _ in range(depth):
            next_frontier = set()
            for current in frontier:
                edges = self.edge_store.fetch_edges_by_entity(current)
                for edge in edges:
                    other = edge.target if edge.source == current else edge.source
                    if other not in visited:
                        visited.add(other)
                        next_frontier.add(other)
            frontier = next_frontier
            if not frontier:
                break

        return list(visited - {entity_name})

    def _get_neighbors_at_depth_with_cache(self, entity_name: str, depth: int, processed: set, entity_cache: dict) -> List[str]:
        """
        Get neighbors with caching to avoid repeated DB lookups.
        """
        if not entity_name or depth <= 0:
            return []

        visited = {entity_name}
        frontier = {entity_name}
        result = []

        for _ in range(depth):
            next_frontier = set()
            for current in frontier:
                # Skip if already processed in the main search
                if current in processed:
                    continue

                edges = self.edge_store.fetch_edges_by_entity(current)
                for edge in edges:
                    other = edge.target if edge.source == current else edge.source
                    if other not in visited:
                        visited.add(other)
                        next_frontier.add(other)
                        result.append(other)

                        # Cache entity if we find it
                        if other not in entity_cache:
                            entity = self.find_entity(other)
                            if entity:
                                entity_cache[other] = entity

            frontier = next_frontier
            if not frontier:
                break

        return result

    def get_entity_relations(self, entity_name: str) -> List[dict]:
        """Get all relations for a specific entity."""
        if not entity_name:
            return []
        edges = self.edge_store.fetch_edges_by_entity(entity_name)
        return [
            {
                "source": edge.source,
                "relation": edge.relation,
                "target": edge.target,
                "memory_id": edge.memory_id
            }
            for edge in edges
        ]

    def get_entity_neighbors(self, entity_name: str, depth: int = 1) -> List[str]:
        """Get all entity names within depth without relation details."""
        neighbors = self.neighbors(entity_name, depth=depth)
        return [item["entity_name"] for item in neighbors]

    def get_entity_connections(self, entity_name: str) -> dict:
        """Get a summary of entity connections."""
        edges = self.edge_store.fetch_edges_by_entity(entity_name)
        outgoing = []
        incoming = []
        for edge in edges:
            if edge.source == entity_name:
                outgoing.append({"target": edge.target, "relation": edge.relation})
            if edge.target == entity_name:
                incoming.append({"source": edge.source, "relation": edge.relation})
        return {
            "entity": entity_name,
            "outgoing": outgoing,
            "incoming": incoming,
            "total_edges": len(edges)
        }

    def get_stats(self) -> dict:
        """Get search statistics."""
        return {
            "numpy_graph_available": self.numpy_graph is not None and self.numpy_graph.built,
            "entity_store_available": self.entity_store is not None,
        }
