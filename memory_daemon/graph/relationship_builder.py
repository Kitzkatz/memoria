from graph.models import GraphRecord


class RelationshipBuilder:

    def __init__(
        self,
        edge_store,
        entity_store,
        
    ):

        self.edge_store = edge_store
        self.entity_store = entity_store
        

    def get_entity_id(self, name):
        entity = self.entity_store.get_or_create(name)
        return entity.id

    def add_relationship(
        self,
        memory_id,
        source,
        relation,
        target
    ):

        source_id = self.get_entity_id(source)

        target_id = self.get_entity_id(target)

        #
        # Graph edge
        #

        record = GraphRecord(

            memory_id=memory_id,

            source=source_id,

            relation=relation,

            target=target_id

        )

        self.edge_store.insert_edge(record)

       

    def build(self, mem_id, relationships):
        if not relationships:
            return
        for rel in relationships:
            if not isinstance(rel, dict):
                continue
            source = rel.get("source")
            relation = rel.get("relation")
            target = rel.get("target")
            if not source or not target or not relation:
                continue
            self.add_relationship(mem_id, source, relation, target)
