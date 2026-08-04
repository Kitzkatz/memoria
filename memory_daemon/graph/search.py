from collections import deque


class GraphSearch:

    def __init__(self, edge_store, entity_store):
        self.edge_store = edge_store
        
        self.entity_store = entity_store

    def find_entity(self, name):
        return self.entity_store.find(name)

    def neighbors(self, entity_id, depth=1):
        visited = set()
        results = []
        queue = deque()
        queue.append((entity_id, 0))
        while queue:
            current, level = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            if level >= depth:
                continue
            edges = self.edge_store.fetch_edges_by_entity(current)
            for edge in edges:
                other = edge.target if edge.source == current else edge.source
                results.append({
                    "entity_id": other,
                    "relation": edge.relation,
                    "source": edge.source,
                    "target": edge.target
                })
                queue.append((other, level + 1))
        return results

    def entity_memories(self, entity_id):
        return self.edge_store.get_memory_ids_for_entity(entity_id)

    def search(self, entities, depth=1, limit=200):
        memory_ids = set()
        for name in entities:
            entity = self.find_entity(name)
            if not entity:
                continue
            entity_id = entity.id
            
            # Direct entity memories
            for mem_id in self.entity_memories(entity_id):
                memory_ids.add(mem_id)
                if len(memory_ids) >= limit:
                    return list(memory_ids)
            
            # Neighbor memories
            related = self.neighbors(entity_id, depth)
            for item in related:
                for mem_id in self.entity_memories(item["entity_id"]):
                    memory_ids.add(mem_id)
                    if len(memory_ids) >= limit:
                        return list(memory_ids)
        
        return list(memory_ids)
