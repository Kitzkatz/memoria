import os
import sqlite3
import json
import threading
from datetime import datetime

from cache.config import settings
from core.logger import debug


class MemoryDB:

    def __init__(self):

        self.lock = threading.Lock()

        self.conn = sqlite3.connect(
            settings.DB_PATH,
            check_same_thread=False
        )

        # IMPORTANT: enables dict-style access
        self.conn.row_factory = sqlite3.Row

        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")

        debug("[DB] Using:", os.path.abspath(settings.DB_PATH))

        self._init()

    # -------------------------
    # SCHEMA
    # -------------------------

    def _init(self):

        with self.conn:

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    text TEXT NOT NULL,

                    normalized_text TEXT,

                    tokens TEXT,

                    token_count INTEGER DEFAULT 0,

                    memory_type TEXT,

                    metadata TEXT,

                    entities TEXT,

                    relationships TEXT,

                    importance REAL DEFAULT 0.5,

                    created_at TEXT,

                    last_accessed TEXT,

                    tombstone INTEGER DEFAULT 0

                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    name TEXT UNIQUE,

                    entity_type TEXT,

                    aliases TEXT

                )
            """)

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS graph (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    memory_id INTEGER NOT NULL,

                    source TEXT NOT NULL,

                    relation TEXT NOT NULL,

                    target TEXT NOT NULL,

                    weight REAL DEFAULT 1.0,

                    created_at TEXT,

                    FOREIGN KEY(memory_id)
                        REFERENCES memories(id)
                )
            """)
            
            #entity links i want to say is v4
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS entity_links (
                    entity_id INTEGER,
                    memory_id INTEGER
                )
            """)

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS goals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal TEXT,
                    progress TEXT,
                    status TEXT DEFAULT 'active'
                )
            """)

        self.conn.commit()

    # -------------------------
    # INSERT
    # -------------------------

    def insert(self, record):

        with self.lock:

            cur = self.conn.cursor()

            now = datetime.utcnow().isoformat()
            normalized_text = record.normalized_text or record.text
            tokens = record.tokens or normalized_text.split()
            token_count = record.token_count or len(tokens)

            cur.execute("""
                INSERT INTO memories (

                    text,
                    normalized_text,
                    tokens,
                    token_count,
                    memory_type,
                    metadata,
                    entities,
                    relationships,
                    importance,
                    created_at,
                    last_accessed

                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (

                record.text,

                normalized_text,

                json.dumps(tokens),

                int(token_count),

                record.memory_type,

                json.dumps(record.metadata or {}),
                json.dumps(record.entities or []),

                json.dumps(record.relationships or []),

                float(record.importance or 0.5),

                now,

                now

            ))

            self.conn.commit()

            return cur.lastrowid
        # -------------------------
    # BATCH INSERT
    # -------------------------

    def insert_many(self, records):

        ids = []

        with self.lock:

            cur = self.conn.cursor()

            now = datetime.utcnow().isoformat()

            #
            # Find next available ID
            #

            cur.execute(
                "SELECT COALESCE(MAX(id), 0) FROM memories"
            )

            next_id = cur.fetchone()[0] + 1

            rows = []

            for offset, record in enumerate(records):

                mem_id = next_id + offset

                ids.append(mem_id)

                normalized_text = (
                    record.normalized_text
                    or record.text
                )

                tokens = (
                    record.tokens
                    or normalized_text.split()
                )

                token_count = (
                    record.token_count
                    or len(tokens)
                )

                rows.append((

                    mem_id,

                    record.text,

                    normalized_text,

                    json.dumps(tokens),

                    int(token_count),

                    record.memory_type,

                    json.dumps(record.metadata or {}),
                    
                    json.dumps(record.entities or []),

                    json.dumps(record.relationships or []),

                    float(record.importance or 0.5),

                    now,

                    now

                ))

            cur.executemany("""

                INSERT INTO memories (

                    id,
                    text,
                    normalized_text,
                    tokens,
                    token_count,
                    memory_type,
                    metadata,
                    entities,
                    relationships,
                    importance,
                    created_at,
                    last_accessed

                )

                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

            """, rows)

            self.conn.commit()

        return ids


         
    # -------------------------
    # FETCH
    # -------------------------

    def fetch(self, mem_id):

        debug("\n========== FETCH ==========")
        debug("Requested ID:", mem_id)

        cur = self.conn.cursor()

        cur.execute("""
            SELECT *
            FROM memories
            WHERE id = ?
              AND tombstone = 0
        """, (int(mem_id),))

        row = cur.fetchone()
        

        if row is None:
            return None
        tokens = json.loads(row["tokens"]) if row["tokens"] else []
        normalized_text = row["normalized_text"] if row["normalized_text"] is not None else row["text"]
        return {

            "id": row["id"],

            "text": row["text"],

            "normalized_text": normalized_text,
            "tokens": tokens,
            "token_count": row["token_count"] or len(tokens),

            "memory_type": row["memory_type"],

            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},

            "entities": json.loads(row["entities"]) if row["entities"] else [],

            "relationships": json.loads(row["relationships"]) if row["relationships"] else [],

            "importance": float(row["importance"] or 0.5),

            "created_at": row["created_at"],

            "last_accessed": row["last_accessed"],

            "tombstone": row["tombstone"]

        }

    def fetch_many(self, ids):

        if not ids:
            return {}

        placeholders = ",".join("?" for _ in ids)

        cur = self.conn.cursor()

        cur.execute(
            f"""
            SELECT *
            FROM memories
            WHERE id IN ({placeholders})
              AND tombstone = 0
            """,
            [int(i) for i in ids]
        )

        rows = cur.fetchall()

        result = {}

        for row in rows:

            tokens = json.loads(row["tokens"]) if row["tokens"] else []
            normalized_text = row["normalized_text"] if row["normalized_text"] is not None else row["text"]

            result[row["id"]] = {

                "id": row["id"],
                "text": row["text"],
                "normalized_text": normalized_text,
                "tokens": tokens,
                "token_count": row["token_count"] or len(tokens),
                "memory_type": row["memory_type"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "entities": json.loads(row["entities"]) if row["entities"] else [],
                "relationships": json.loads(row["relationships"]) if row["relationships"] else [],
                "importance": float(row["importance"] or 0.5),
                "created_at": row["created_at"],
                "last_accessed": row["last_accessed"],
                "tombstone": row["tombstone"]
            }

        return result


    def fetch_all(self):

        cur = self.conn.cursor()

        cur.execute("""
            SELECT *
            FROM memories
            WHERE tombstone = 0
        """)

        rows = cur.fetchall()
        

        result = []

        for row in rows:
            tokens = json.loads(row["tokens"]) if row["tokens"] else []
            normalized_text = row["normalized_text"] if row["normalized_text"] is not None else row["text"]

            result.append({

                "id": row["id"],

                "text": row["text"],

                "normalized_text": normalized_text,
                "tokens": tokens,
                "token_count": row["token_count"] or len(tokens),

                "memory_type": row["memory_type"],

                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},

                "entities": json.loads(row["entities"]) if row["entities"] else [],

                "relationships": json.loads(row["relationships"]) if row["relationships"] else [],

                "importance": float(row["importance"] or 0.5),

                "created_at": row["created_at"],

                "last_accessed": row["last_accessed"],

                "tombstone": row["tombstone"]

            })

        return result
    def search_attribute(self, subject, attribute):
        if not subject or not attribute:
            return []

        # Use the virtual columns (subject, attribute) which are indexed
        cur = self.conn.execute(
            """
            SELECT *
            FROM memories
            WHERE subject = ?
              AND attribute = ?
              AND tombstone = 0
            """,
            (subject, attribute)
        )
        rows = cur.fetchall()
        debug("ATTRIBUTE HITS:", len(rows))
        return rows

        
    # -------------------------
    # UPDATE
    # -------------------------

    def update(self, mem_id, **kwargs):

        allowed = {
            "text",
            "normalized_text",
            "tokens",
            "token_count",
            "importance",
            "memory_type",
            "metadata",
            "entities",
            "relationships",
            "last_accessed"
        }

        fields = []
        values = []

        for k, v in kwargs.items():

            if k not in allowed:
                continue

            fields.append(f"{k}=?")

            if k == "tokens":
                values.append(json.dumps(v or []))
            elif k =="metadata":
                values.append(json.dumps(v or {}))

            elif k == "entities":
                values.append(json.dumps(v or []))

            elif k == "relationships":
                values.append(json.dumps(v or []))
            
            else:
                values.append(json.dumps(v) if isinstance(v, (dict, list)) else v)

        if not fields:
            return

        values.append(mem_id)

        query = f"""
            UPDATE memories
            SET {", ".join(fields)}
            WHERE id=?
        """

        with self.conn:
            self.conn.execute(query, values)
    
    # -------------------------
    # DELETE (soft)
    # -------------------------

    def delete(self, mem_id):

        with self.conn:
            self.conn.execute("""
                UPDATE memories
                SET tombstone = 1
                WHERE id = ?
            """, (int(mem_id),))

    # -------------------------
    # HELPERS
    # -------------------------

    def count(self):

        cur = self.conn.cursor()

        cur.execute("""
            SELECT COUNT(*)
            FROM memories
            WHERE tombstone = 0
        """)

        return cur.fetchone()[0]

    def latest(self, limit=10):

        cur = self.conn.cursor()

        cur.execute("""
            SELECT id, text, importance
            FROM memories
            WHERE tombstone = 0
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))

        return cur.fetchall()

    # -------------------------
    # TIME HELPERS
    # -------------------------

    def touch(self, mem_id, delta_importance=0.01):

        with self.conn:

            self.conn.execute("""
                UPDATE memories
                SET
                    last_accessed = ?,
                    importance = importance + ?
                WHERE id = ?
            """, (
                datetime.utcnow().isoformat(),
                delta_importance,
                int(mem_id)
            ))

    def decay_memories(self, decay_rate=0.001):

        with self.conn:

            self.conn.execute("""
                UPDATE memories
                SET importance = MAX(0.1, importance - ?)
                WHERE last_accessed < datetime('now', '-1 day')
            """, (decay_rate,))

    def schema(self):

        cur = self.conn.cursor()

        cur.execute("PRAGMA table_info(memories)")

        return [dict(r) for r in cur.fetchall()]


    def integrity_check(self):

        cur = self.conn.cursor()

        cur.execute("PRAGMA integrity_check")

        return cur.fetchone()[0]

    def sanity_check(self):

        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM memories")

        db_count = cur.fetchone()[0]

        cur.execute("PRAGMA table_info(memories)")
        schema = cur.fetchall()

        return {
            "db_count": db_count,
            "columns": [c[1] for c in schema]
        }
