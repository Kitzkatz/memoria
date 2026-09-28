from collections import deque
from typing import List
from core.logger import debug


class GraphSearch:

    def __init__(self, edge_store, entity_store, numpy_graph=None):
        self.edge_store = edge_store
        self.entity_store = entity_store
        self.numpy_graph = numpy_graph

    def find_entity(self, name):
        return self.entity_store.find(name)

    def _entity_id(self, name):
        """Resolve a public entity name to its canonical entity ID."""
        entity = self.find_entity(name)
        return entity.id if entity else None

    def _entity_name(self, entity_id):
        """Resolve a canonical entity ID to its public entity name."""
        entity = self.entity_store.find_by_id(entity_id)
        return entity.name if entity else None

    def neighbors(self, entity_name: str, depth: int = 1):
        """
        Return related entities and relations up to depth.

        Public API uses entity names; graph storage and traversal use
        canonical entity IDs.
        """
        if not entity_name or depth <= 0:
            return []

        entity_id = self._entity_id(entity_name)
        if entity_id is None:
            return []

        visited = set()
        results = []
        queue = deque()
        queue.append((entity_id, 0))

        while queue:
            current_id, level = queue.popleft()

            if current_id in visited:
                continue

            visited.add(current_id)

            if level >= depth:
                continue

            edges = self.edge_store.fetch_edges_by_entity(current_id)

            for edge in edges:
                other_id = (
                    edge.target
                    if edge.source == current_id
                    else edge.source
                )

                if other_id in visited:
                    continue

                other_name = self._entity_name(other_id)
                if other_name is None:
                    continue

                source_name = self._entity_name(edge.source)
                target_name = self._entity_name(edge.target)

                results.append({
                    "entity_name": other_name,
                    "relation": edge.relation,
                    "source": source_name,
                    "target": target_name,
                })

                queue.append((other_id, level + 1))

        return results

    def entity_memories(self, entity_name: str) -> List[int]:
        """Get all memory IDs associated with an entity."""
        entity_id = self._entity_id(entity_name)
        if entity_id is None:
            return []

        return self.edge_store.get_memory_ids_for_entity(entity_id)

    def search(
        self,
        entities: List[str],
        depth: int = 1,
        limit: int = 200
    ) -> List[int]:
        """
        Search for memory IDs related to a list of entities up to depth.

        Public API accepts entity names. Graph internals use canonical
        entity IDs. Uses numpy_graph when available.
        """
        if not entities:
            return []

        debug(
            f"GraphSearch: entities={entities}, depth={depth}, limit={limit}",
            category="graph"
        )

        # Resolve public names to canonical IDs once.
        entity_ids = []
        for name in entities:
            entity_id = self._entity_id(name)
            if entity_id is not None:
                entity_ids.append(entity_id)

        if not entity_ids:
            return []

        # Fast NumPy path.
        if self.numpy_graph and self.numpy_graph.built and depth <= 2:
            memory_ids = self.numpy_graph.multi_hop_search(
                entity_ids,
                depth=depth,
                limit=limit
            )
            debug(
                f"GraphSearch: numpy_graph returned {len(memory_ids)} memories",
                category="graph"
            )
            return memory_ids

        # Edge-store fallback.
        memory_ids = set()
        processed = set()
        queue = deque(entity_ids)

        while queue and len(memory_ids) < limit:
            current_id = queue.popleft()

            if current_id in processed:
                continue

            processed.add(current_id)

            # Direct memories.
            for mem_id in self.edge_store.get_memory_ids_for_entity(current_id):
                memory_ids.add(mem_id)

                if len(memory_ids) >= limit:
                    return list(memory_ids)[:limit]

            # Traverse neighbors.
            if depth > 0:
                neighbor_ids = self._get_neighbor_ids(
                    current_id,
                    depth,
                    processed
                )

                for neighbor_id in neighbor_ids:
                    if neighbor_id not in processed:
                        queue.append(neighbor_id)

        return list(memory_ids)[:limit]

    def _get_neighbor_ids(
        self,
        entity_id: int,
        depth: int,
        processed: set
    ) -> List[int]:
        """Get canonical entity IDs reachable within depth."""
        if entity_id is None or depth <= 0:
            return []

        visited = {entity_id}
        frontier = {entity_id}
        result = []

        for _ in range(depth):
            next_frontier = set()

            for current_id in frontier:
                if current_id in processed:
                    continue

                edges = self.edge_store.fetch_edges_by_entity(current_id)

                for edge in edges:
                    other_id = (
                        edge.target
                        if edge.source == current_id
                        else edge.source
                    )

                    if other_id not in visited:
                        visited.add(other_id)
                        next_frontier.add(other_id)
                        result.append(other_id)

            frontier = next_frontier

            if not frontier:
                break

        return result

    def _get_neighbors_at_depth(
        self,
        entity_name: str,
        depth: int
    ) -> List[str]:
        """Get entity names reachable within depth, excluding self."""
        entity_id = self._entity_id(entity_name)

        if entity_id is None or depth <= 0:
            return []

        neighbor_ids = self._get_neighbor_ids(
            entity_id,
            depth,
            set()
        )

        names = []

        for neighbor_id in neighbor_ids:
            name = self._entity_name(neighbor_id)
            if name is not None:
                names.append(name)

        return names

    def _get_neighbors_at_depth_with_cache(
        self,
        entity_name: str,
        depth: int,
        processed: set,
        entity_cache: dict
    ) -> List[str]:
        """
        Get neighboring entity names with caching.

        The cache remains name-keyed because this method is an internal
        compatibility path used by callers that operate on public names.
        """
        entity_id = self._entity_id(entity_name)

        if entity_id is None or depth <= 0:
            return []

        neighbor_ids = self._get_neighbor_ids(
            entity_id,
            depth,
            {
                self._entity_id(name)
                for name in processed
                if self._entity_id(name) is not None
            }
        )

        result = []

        for neighbor_id in neighbor_ids:
            name = self._entity_name(neighbor_id)
            if name is None:
                continue

            result.append(name)

            if name not in entity_cache:
                entity = self.find_entity(name)
                if entity:
                    entity_cache[name] = entity

        return result

    def get_entity_relations(self, entity_name: str) -> List[dict]:
        """Get all relations for a specific entity."""
        entity_id = self._entity_id(entity_name)

        if entity_id is None:
            return []

        edges = self.edge_store.fetch_edges_by_entity(entity_id)

        relations = []

        for edge in edges:
            source_name = self._entity_name(edge.source)
            target_name = self._entity_name(edge.target)

            relations.append({
                "source": source_name,
                "relation": edge.relation,
                "target": target_name,
                "memory_id": edge.memory_id,
            })

        return relations

    def get_entity_neighbors(
        self,
        entity_name: str,
        depth: int = 1
    ) -> List[str]:
        """Get all entity names within depth without relation details."""
        neighbors = self.neighbors(entity_name, depth=depth)
        return [item["entity_name"] for item in neighbors]

    def get_entity_connections(self, entity_name: str) -> dict:
        """Get a summary of entity connections."""
        entity_id = self._entity_id(entity_name)

        if entity_id is None:
            return {
                "entity": entity_name,
                "outgoing": [],
                "incoming": [],
                "total_edges": 0,
            }

        edges = self.edge_store.fetch_edges_by_entity(entity_id)

        outgoing = []
        incoming = []

        for edge in edges:
            if edge.source == entity_id:
                target_name = self._entity_name(edge.target)
                if target_name is not None:
                    outgoing.append({
                        "target": target_name,
                        "relation": edge.relation
                    })

            if edge.target == entity_id:
                source_name = self._entity_name(edge.source)
                if source_name is not None:
                    incoming.append({
                        "source": source_name,
                        "relation": edge.relation
                    })

        return {
            "entity": entity_name,
            "outgoing": outgoing,
            "incoming": incoming,
            "total_edges": len(edges),
        }

    def get_stats(self) -> dict:
        """Get search statistics."""
        return {
            "numpy_graph_available": (
                self.numpy_graph is not None
                and self.numpy_graph.built
            ),
            "entity_store_available": self.entity_store is not None,
        }
