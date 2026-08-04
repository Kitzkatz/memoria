import json
from graph.models import EntityRecord


class EntityStore:
    def __init__(self, db):
        self.db = db

    def normalize(self, name):
        if not name:
            return ""
        return name.lower().strip()

    def _row_to_record(self, row):
        return EntityRecord(
            id=row["id"],
            name=row["name"],
            entity_type=row["entity_type"],
            aliases=json.loads(row["aliases"]) if row["aliases"] else []
        )

    def find(self, name):
        normalized = self.normalize(name)
        cur = self.db.conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM entities
            WHERE lower(name) = ?
               OR aliases LIKE ?
            """,
            (
                normalized,
                f'%"{normalized}"%'
            )
        )
        row = cur.fetchone()
        if row:
            return self._row_to_record(row)
        return None

    def create(self, name, entity_type="unknown"):
        normalized = self.normalize(name)
        with self.db.lock:
            cur = self.db.conn.cursor()
            cur.execute(
                """
                INSERT INTO entities
                (name, entity_type, aliases)
                VALUES (?, ?, ?)
                """,
                (
                    normalized,
                    entity_type,
                    json.dumps([normalized])
                )
            )
            self.db.conn.commit()
            entity_id = cur.lastrowid
        return self.get(entity_id)

    def get(self, entity_id):
        cur = self.db.conn.cursor()
        cur.execute(
            "SELECT * FROM entities WHERE id = ?",
            (entity_id,)
        )
        row = cur.fetchone()
        if row:
            return self._row_to_record(row)
        return None

    def get_or_create(self, name, entity_type="unknown"):
        existing = self.find(name)
        if existing:
            return existing
        return self.create(name, entity_type)

    def add_alias(self, entity_id, alias):
        entity = self.get(entity_id)
        if not entity:
            return
        aliases = entity.aliases
        normalized = self.normalize(alias)
        if normalized not in aliases:
            aliases.append(normalized)
        with self.db.lock:
            self.db.conn.execute(
                "UPDATE entities SET aliases = ? WHERE id = ?",
                (json.dumps(aliases), entity_id)
            )
            self.db.conn.commit()

