class EntityResolver:
    def __init__(self, entity_store):
        self.entity_store = entity_store

    def normalize(self, entity):
        return entity.lower().strip()

    def resolve(self, entity):
        normalized = self.normalize(entity)
        record = self.entity_store.get_or_create(normalized)
        return record.id

    def resolve_many(self, entities):
        ids = []
        seen = set()
        for entity in entities:
            entity_id = self.resolve(entity)
            if entity_id in seen:
                continue
            seen.add(entity_id)
            ids.append(entity_id)
        return ids

