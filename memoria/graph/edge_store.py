from graph.models import GraphRecord
from core.logger import debug


class EdgeStore:

    def __init__(self, db):
        self.db = db

    def insert_edge(self, record: GraphRecord) -> bool:
        """
        Insert a single edge into the graph.
        Returns True if inserted, False if duplicate edge already exists.
        """
        # Check for duplicate edge
        cur = self.db.conn.cursor()
        cur.execute(
            """
            SELECT id FROM graph
            WHERE memory_id = ? AND source = ? AND relation = ? AND target = ?
            """,
            (
                record.memory_id,
                record.source,
                record.relation,
                record.target
            )
        )
        if cur.fetchone():
            debug(f"[EdgeStore] Duplicate edge skipped: {record.source} -{record.relation}-> {record.target}")
            return False

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
            return True

    def insert_edges(self, records: list) -> int:
        """
        Insert multiple edges in batch.
        Returns number of edges inserted.
        """
        inserted = 0
        for record in records:
            if self.insert_edge(record):
                inserted += 1
        return inserted

    def fetch_edges(self, mem_id):
        """Fetch all edges for a memory ID."""
        cur = self.db.conn.cursor()
        cur.execute(
            "SELECT * FROM graph WHERE memory_id = ?",
            (mem_id,)
        )
        rows = cur.fetchall()
        return [GraphRecord(**dict(row)) for row in rows]

    def fetch_edges_by_entity(self, entity_name: str):
        """
        Fetch all edges where entity appears as source or target.
        Args:
            entity_name: String entity name (not ID) — graph stores source/target as TEXT
        """
        cur = self.db.conn.cursor()
        cur.execute(
            "SELECT * FROM graph WHERE source = ? OR target = ?",
            (entity_name, entity_name)
        )
        rows = cur.fetchall()
        return [GraphRecord(**dict(row)) for row in rows]

    def fetch_by_relation(self, relation: str):
        """Fetch all edges with a specific relation type."""
        cur = self.db.conn.cursor()
        cur.execute(
            "SELECT * FROM graph WHERE relation = ?",
            (relation,)
        )
        rows = cur.fetchall()
        return [GraphRecord(**dict(row)) for row in rows]

    def get_memory_ids_for_entity(self, entity_name: str):
        """Get all memory IDs associated with an entity."""
        cur = self.db.conn.cursor()
        cur.execute(
            "SELECT DISTINCT memory_id FROM graph WHERE source = ? OR target = ?",
            (entity_name, entity_name)
        )
        return [row["memory_id"] for row in cur.fetchall()]

    def get_entities_for_memory(self, mem_id):
        """Get all entities connected to a memory."""
        cur = self.db.conn.cursor()
        cur.execute(
            "SELECT DISTINCT source, target FROM graph WHERE memory_id = ?",
            (mem_id,)
        )
        entities = set()
        for row in cur.fetchall():
            entities.add(row["source"])
            entities.add(row["target"])
        return list(entities)

    def delete_edges_for_memory(self, mem_id):
        """Delete all edges for a memory (used when memory is pruned)."""
        with self.db.lock:
            self.db.conn.execute(
                "DELETE FROM graph WHERE memory_id = ?",
                (mem_id,)
            )
            self.db.conn.commit()
        debug(f"[EdgeStore] Deleted edges for memory {mem_id}")

    def delete_edge(self, memory_id, source, relation, target):
        """Delete a specific edge."""
        with self.db.lock:
            self.db.conn.execute(
                """
                DELETE FROM graph
                WHERE memory_id = ? AND source = ? AND relation = ? AND target = ?
                """,
                (memory_id, source, relation, target)
            )
            self.db.conn.commit()

    def count(self):
        """Return total number of edges."""
        cur = self.db.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM graph")
        return cur.fetchone()[0]

    def stats(self) -> dict:
        """Return edge statistics."""
        cur = self.db.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM graph")
        total = cur.fetchone()[0]

        cur.execute("SELECT COUNT(DISTINCT memory_id) FROM graph")
        memories = cur.fetchone()[0]

        cur.execute("SELECT COUNT(DISTINCT source) FROM graph")
        sources = cur.fetchone()[0]

        cur.execute("SELECT COUNT(DISTINCT target) FROM graph")
        targets = cur.fetchone()[0]

        return {
            "total_edges": total,
            "memories_with_edges": memories,
            "unique_sources": sources,
            "unique_targets": targets,
        }
