from typing import List, Dict, Optional, Set
from graph.models import GraphRecord
from core.logger import debug


class RelationshipBuilder:

    def __init__(self, edge_store, entity_store):
        self.edge_store = edge_store
        self.entity_store = entity_store
        self._entity_cache = {}  # name -> entity_id (cleared per batch)

    def _normalize(self, name: str) -> str:
        """Normalize entity name for consistent lookup."""
        if not name:
            return ""
        return str(name).lower().strip()

    def get_entity_id(self, name: str) -> int:
        """
        Get or create an entity ID.
        Uses local cache to avoid repeated DB lookups.
        """
        normalized = self._normalize(name)
        if not normalized:
            return -1

        # Check local cache first
        if normalized in self._entity_cache:
            return self._entity_cache[normalized]

        # Look up in entity store
        entity = self.entity_store.get_or_create(normalized)
        self._entity_cache[normalized] = entity.id
        return entity.id

    def get_entity_ids_batch(self, names: List[str]) -> Dict[str, int]:
        """
        Get or create entity IDs for multiple names in batch.
        Uses local cache to minimize DB round trips.
        """
        result = {}
        to_lookup = []

        # Check cache first
        for name in names:
            normalized = self._normalize(name)
            if not normalized:
                continue
            if normalized in self._entity_cache:
                result[name] = self._entity_cache[normalized]
            else:
                to_lookup.append(normalized)

        if not to_lookup:
            return result

        # Look up each entity (could be batched if EntityStore had batch find)
        for normalized in to_lookup:
            entity = self.entity_store.get_or_create(normalized)
            self._entity_cache[normalized] = entity.id
            result[normalized] = entity.id

        return result

    def add_relationship(self, memory_id: int, source: str, relation: str, target: str) -> bool:
        """
        Add a single relationship to the graph.
        Returns True if added, False if invalid.
        """
        if not source or not target or not relation:
            return False

        if self._normalize(source) == self._normalize(target):
            # Skip self-referential relationships
            debug(f"[RelationshipBuilder] Skipping self-reference: {source} -> {target}")
            return False

        source_id = self.get_entity_id(source)
        target_id = self.get_entity_id(target)

        if source_id == -1 or target_id == -1:
            return False

        record = GraphRecord(
            memory_id=memory_id,
            source=source_id,
            relation=relation,
            target=target_id
        )

        return self.edge_store.insert_edge(record)

    def add_relationships_batch(self, memory_id: int, relationships: list) -> int:
        """
        Add multiple relationships in batch.
        Returns number of relationships added.
        """
        if not relationships:
            return 0

        # Clear cache for this batch
        self._entity_cache.clear()

        added = 0
        records = []

        for rel in relationships:
            if not isinstance(rel, dict):
                continue

            source = rel.get("source")
            relation = rel.get("relation")
            target = rel.get("target")

            if not source or not target or not relation:
                continue

            if self._normalize(source) == self._normalize(target):
                continue

            # Get entity IDs (with caching)
            source_id = self.get_entity_id(source)
            target_id = self.get_entity_id(target)

            if source_id == -1 or target_id == -1:
                continue

            records.append(
                GraphRecord(
                    memory_id=memory_id,
                    source=source_id,
                    relation=relation,
                    target=target_id
                )
            )
            added += 1

        # Batch insert edges
        if records:
            self.edge_store.insert_edges(records)

        return added

    def build(self, mem_id: int, relationships: list) -> int:
        """
        Build relationships for a memory.
        Returns number of relationships added.
        """
        if not relationships:
            return 0

        # Use batch path for efficiency
        return self.add_relationships_batch(mem_id, relationships)

    def get_entity_ids_for_memory(self, mem_id: int) -> set:
        """Get all entity IDs associated with a memory."""
        edges = self.edge_store.fetch_edges(mem_id)
        entity_ids = set()
        for edge in edges:
            entity_ids.add(edge.source)
            entity_ids.add(edge.target)
        return entity_ids

    def get_entity_names_for_memory(self, mem_id: int) -> List[str]:
        """Get all entity names associated with a memory."""
        entity_ids = self.get_entity_ids_for_memory(mem_id)
        names = []
        for eid in entity_ids:
            entity = self.entity_store.get(eid)
            if entity:
                names.append(entity.name)
        return names

    def get_relationship_stats(self, mem_id: int) -> dict:
        """Get relationship statistics for a memory."""
        edges = self.edge_store.fetch_edges(mem_id)
        if not edges:
            return {"total": 0}

        relations = {}
        for edge in edges:
            relations[edge.relation] = relations.get(edge.relation, 0) + 1

        return {
            "total": len(edges),
            "relations": relations,
            "unique_entities": len(set(e.source for e in edges) | set(e.target for e in edges)),
        }
