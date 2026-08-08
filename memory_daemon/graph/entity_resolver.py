from graph.models import EntityRecord
from retrieval.case_folding import fold_case
from core.logger import debug


class EntityResolver:
    def __init__(self, entity_store):
        self.entity_store = entity_store
        # Local cache for entity resolution (cleared per batch)
        self._cache = {}

    def normalize(self, entity: str) -> str:
        """Normalize entity name using case-folding."""
        if not entity:
            return ""
        return fold_case(entity).strip()

    def resolve(self, entity: str) -> int:
        """
        Resolve an entity name to an ID.
        Uses local cache to avoid repeated DB lookups.
        """
        if not entity:
            return -1

        normalized = self.normalize(entity)
        if not normalized:
            return -1

        # Check local cache first
        if normalized in self._cache:
            return self._cache[normalized]

        # Look up or create in entity store
        record = self.entity_store.get_or_create(normalized)
        self._cache[normalized] = record.id
        return record.id

    def resolve_many(self, entities: list) -> list:
        """
        Resolve multiple entity names to IDs.
        Deduplicates and returns unique IDs.
        """
        if not entities:
            return []

        ids = []
        seen = set()

        for entity in entities:
            if not entity:
                continue

            # Use resolve with local cache
            entity_id = self.resolve(entity)
            if entity_id == -1:
                continue

            if entity_id not in seen:
                seen.add(entity_id)
                ids.append(entity_id)

        return ids

    def resolve_many_batch(self, entities: list) -> dict:
        """
        Resolve multiple entities and return a mapping of name -> ID.
        More efficient than resolve_many when you need the mapping.
        """
        if not entities:
            return {}

        # Clear cache for this batch
        self._cache.clear()

        result = {}
        for entity in entities:
            if not entity:
                continue

            normalized = self.normalize(entity)
            if not normalized:
                continue

            # Use the entity store's get_or_create
            record = self.entity_store.get_or_create(normalized)
            result[entity] = record.id
            self._cache[normalized] = record.id

        return result

    def resolve_many_batch_with_type(self, entities: list, entity_type: str = None) -> dict:
        """
        Resolve multiple entities with a specific type.
        """
        if not entities:
            return {}

        self._cache.clear()
        result = {}

        for entity in entities:
            if not entity:
                continue

            normalized = self.normalize(entity)
            if not normalized:
                continue

            # Try to find existing first
            existing = self.entity_store.find(normalized)
            if existing:
                result[entity] = existing.id
                self._cache[normalized] = existing.id
            else:
                # Create with specified type
                record = self.entity_store.create(normalized, entity_type or "unknown")
                result[entity] = record.id
                self._cache[normalized] = record.id

        return result

    def clear_cache(self):
        """Clear the local resolution cache."""
        self._cache.clear()
        debug("[EntityResolver] Cache cleared")

    def get_stats(self) -> dict:
        """Return resolver statistics."""
        return {
            "cache_size": len(self._cache),
        }
