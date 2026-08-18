import json
from graph.models import EntityRecord
from retrieval.case_folding import fold_case


class EntityStore:
    def __init__(self, db):
        self.db = db

    def normalize(self, name):
        """Normalize entity name using case-folding."""
        if not name:
            return ""
        return fold_case(name).strip()

    def _row_to_record(self, row):
        return EntityRecord(
            id=row["id"],
            name=row["name"],
            entity_type=row["entity_type"],
            aliases=json.loads(row["aliases"]) if row["aliases"] else []
        )

    def find(self, name):
        """
        Find an entity by name or alias.
        Uses json_extract for proper JSON querying.
        """
        normalized = self.normalize(name)
        if not normalized:
            return None

        cur = self.db.conn.cursor()

        # Try exact match on name first
        cur.execute(
            """
            SELECT *
            FROM entities
            WHERE name = ?
            """,
            (normalized,)
        )
        row = cur.fetchone()
        if row:
            return self._row_to_record(row)

        # Try alias match using json_each (SQLite JSON function)
        try:
            cur.execute(
                """
                SELECT e.*
                FROM entities e, json_each(e.aliases) as alias
                WHERE alias.value = ?
                """,
                (normalized,)
            )
            row = cur.fetchone()
            if row:
                return self._row_to_record(row)
        except Exception:
            # Fallback to LIKE if json_each fails (older SQLite)
            cur.execute(
                """
                SELECT *
                FROM entities
                WHERE aliases LIKE ?
                """,
                (f'%"{normalized}"%',)
            )
            row = cur.fetchone()
            if row:
                return self._row_to_record(row)

        return None

    def find_by_id(self, entity_id):
        """Find entity by ID."""
        cur = self.db.conn.cursor()
        cur.execute(
            "SELECT * FROM entities WHERE id = ?",
            (entity_id,)
        )
        row = cur.fetchone()
        if row:
            return self._row_to_record(row)
        return None

    def create(self, name, entity_type="unknown", aliases=None):
        """Create a new entity."""
        normalized = self.normalize(name)
        if not normalized:
            return None

        # Use provided aliases or default
        alias_list = aliases or [normalized]
        # Ensure normalized name is in aliases
        if normalized not in alias_list:
            alias_list.append(normalized)

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
                    json.dumps(alias_list)
                )
            )
            self.db.conn.commit()
            entity_id = cur.lastrowid

        return self.find_by_id(entity_id)

    def get(self, entity_id):
        """Alias for find_by_id."""
        return self.find_by_id(entity_id)

    def get_or_create(self, name, entity_type="unknown", aliases=None):
        """Get existing entity or create new one."""
        existing = self.find(name)
        if existing:
            return existing
        return self.create(name, entity_type, aliases)

    def add_alias(self, entity_id, alias):
        """Add an alias to an existing entity."""
        entity = self.find_by_id(entity_id)
        if not entity:
            return False

        normalized = self.normalize(alias)
        if not normalized or normalized in entity.aliases:
            return False

        aliases = entity.aliases.copy()
        aliases.append(normalized)

        with self.db.lock:
            self.db.conn.execute(
                "UPDATE entities SET aliases = ? WHERE id = ?",
                (json.dumps(aliases), entity_id)
            )
            self.db.conn.commit()

        return True

    def remove_alias(self, entity_id, alias):
        """Remove an alias from an entity."""
        entity = self.find_by_id(entity_id)
        if not entity:
            return False

        normalized = self.normalize(alias)
        if not normalized or normalized == entity.name:
            # Don't remove the primary name
            return False

        if normalized not in entity.aliases:
            return False

        aliases = entity.aliases.copy()
        aliases.remove(normalized)

        with self.db.lock:
            self.db.conn.execute(
                "UPDATE entities SET aliases = ? WHERE id = ?",
                (json.dumps(aliases), entity_id)
            )
            self.db.conn.commit()

        return True

    def update_type(self, entity_id, entity_type):
        """Update entity type."""
        with self.db.lock:
            self.db.conn.execute(
                "UPDATE entities SET entity_type = ? WHERE id = ?",
                (entity_type, entity_id)
            )
            self.db.conn.commit()

    def list_all(self, limit=1000):
        """List all entities."""
        cur = self.db.conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM entities
            ORDER BY id
            LIMIT ?
            """,
            (limit,)
        )
        rows = cur.fetchall()
        return [self._row_to_record(row) for row in rows]

    def find_by_type(self, entity_type, limit=100):
        """Find entities by type."""
        cur = self.db.conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM entities
            WHERE entity_type = ?
            LIMIT ?
            """,
            (entity_type, limit)
        )
        rows = cur.fetchall()
        return [self._row_to_record(row) for row in rows]

    def count(self):
        """Return total number of entities."""
        cur = self.db.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM entities")
        return cur.fetchone()[0]

    def stats(self):
        """Return entity statistics."""
        cur = self.db.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM entities")
        total = cur.fetchone()[0]

        cur.execute(
            "SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type"
        )
        types = {row["entity_type"]: row["COUNT(*)"] for row in cur.fetchall()}

        return {
            "total_entities": total,
            "types": types,
        }
