from graph.models import GraphRecord


class EdgeStore:

    def __init__(self, db):
        self.db = db

    def insert_edge(self, record: GraphRecord):
        with self.db.lock:
            self.db.conn.execute(
                """
                INSERT INTO graph
                (memory_id, source, relation, target, weight, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.memory_id,
                    record.source,
                    record.relation,
                    record.target,
                    record.weight,
                    record.created_at.isoformat()
                )
            )
            self.db.conn.commit()

    def fetch_edges(self, mem_id):
        cur = self.db.conn.cursor()
        cur.execute(
            "SELECT * FROM graph WHERE memory_id = ?",
            (int(mem_id),)
        )
        return [GraphRecord(**dict(row)) for row in cur.fetchall()]

    def fetch_edges_by_entity(self, entity_id):
        cur = self.db.conn.cursor()
        cur.execute(
            "SELECT * FROM graph WHERE source = ? OR target = ?",
            (int(entity_id), int(entity_id))
        )
        return [GraphRecord(**dict(row)) for row in cur.fetchall()]

    def get_memory_ids_for_entity(self, entity_id):
        cur = self.db.conn.cursor()
        cur.execute(
            "SELECT DISTINCT memory_id FROM graph WHERE source = ? OR target = ?",
            (int(entity_id), int(entity_id))
        )
        return [row["memory_id"] for row in cur.fetchall()]
