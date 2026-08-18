import json
from datetime import datetime, timezone

from core.logger import debug


def insert(conn, lock, record):
    """Insert a single memory record."""
    with lock:
        cur = conn.cursor()

        now = datetime.now(timezone.utc).isoformat()
        normalized_text = record.normalized_text or record.text
        tokens = record.tokens or normalized_text.split()
        token_count = record.token_count or len(tokens)

        # Main insert
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

        mem_id = cur.lastrowid

        # Insert into type-specific table
        mem_type = record.memory_type or "general"
        if mem_type != "general":
            type_table = f"memories_{mem_type}"
            cur.execute(f"""
                INSERT INTO {type_table} (
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
            """, (
                mem_id,
                record.text,
                normalized_text,
                json.dumps(tokens),
                int(token_count),
                mem_type,
                json.dumps(record.metadata or {}),
                json.dumps(record.entities or []),
                json.dumps(record.relationships or []),
                float(record.importance or 0.5),
                now,
                now
            ))

        conn.commit()
        return mem_id


def insert_many(conn, lock, records):
    """Insert multiple memory records."""
    ids = []

    with lock:
        cur = conn.cursor()

        now = datetime.now(timezone.utc).isoformat()

        # Find next available ID
        cur.execute("SELECT COALESCE(MAX(id), 0) FROM memories")
        next_id = cur.fetchone()[0] + 1

        rows = []
        type_rows = []

        for offset, record in enumerate(records):
            mem_id = next_id + offset
            ids.append(mem_id)

            normalized_text = record.normalized_text or record.text
            tokens = record.tokens or normalized_text.split()
            token_count = record.token_count or len(tokens)

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

            mem_type = record.memory_type or "general"
            if mem_type != "general":
                type_table = f"memories_{mem_type}"
                type_rows.append({
                    "table": type_table,
                    "data": (
                        mem_id,
                        record.text,
                        normalized_text,
                        json.dumps(tokens),
                        int(token_count),
                        mem_type,
                        json.dumps(record.metadata or {}),
                        json.dumps(record.entities or []),
                        json.dumps(record.relationships or []),
                        float(record.importance or 0.5),
                        now,
                        now
                    )
                })

        # Main insert
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

        # Type table inserts
        for type_row in type_rows:
            table = type_row["table"]
            data = type_row["data"]
            cur.execute(f"""
                INSERT INTO {table} (
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
            """, data)

        conn.commit()

    return ids


def fetch(conn, mem_id):
    """Fetch a single memory by ID."""
    debug("\n========== FETCH ==========")
    debug("Requested ID:", mem_id)

    cur = conn.cursor()
    cur.execute("""
        SELECT *
        FROM memories
        WHERE id = ?
          AND tombstone = 0
    """, (int(mem_id),))

    row = cur.fetchone()

    if row is None:
        return None

    return _row_to_dict(row)


def fetch_many(conn, ids):
    """Fetch multiple memories by ID."""
    if not ids:
        return {}

    placeholders = ",".join("?" for _ in ids)
    cur = conn.cursor()
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
        result[row["id"]] = _row_to_dict(row)

    return result


def fetch_all(conn):
    """Fetch all non-tombstone memories."""
    cur = conn.cursor()
    cur.execute("""
        SELECT *
        FROM memories
        WHERE tombstone = 0
    """)

    rows = cur.fetchall()
    return [_row_to_dict(row) for row in rows]


def fetch_many_by_type(conn, mem_type: str, limit: int = 50):
    """Fetch top N memories of a specific type by importance."""
    table = f"memories_{mem_type}"
    cur = conn.cursor()
    cur.execute(f"""
        SELECT *
        FROM {table}
        WHERE tombstone = 0
        ORDER BY importance DESC, created_at DESC
        LIMIT ?
    """, (limit,))

    rows = cur.fetchall()
    return [_row_to_dict(row) for row in rows]


def search_attribute(conn, subject, attribute):
    """Search memories by subject and attribute."""
    if not subject or not attribute:
        return []

    try:
        cur = conn.execute(
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
    except sqlite3.OperationalError as e:
        if "no such column" in str(e):
            cur = conn.execute(
                """
                SELECT *
                FROM memories
                WHERE json_extract(metadata,'$.subject') = ?
                  AND json_extract(metadata,'$.attribute') = ?
                  AND tombstone = 0
                """,
                (subject, attribute)
            )
            rows = cur.fetchall()
        else:
            raise

    debug("ATTRIBUTE HITS:", len(rows))
    return rows


def update(conn, mem_id, **kwargs):
    """Update a memory record."""
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
        elif k == "metadata":
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

    with conn:
        conn.execute(query, values)


def delete(conn, mem_id):
    """Soft delete a memory."""
    with conn:
        conn.execute("""
            UPDATE memories
            SET tombstone = 1
            WHERE id = ?
        """, (int(mem_id),))


def count(conn):
    """Count non-tombstone memories."""
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*)
        FROM memories
        WHERE tombstone = 0
    """)
    return cur.fetchone()[0]


def latest(conn, limit=10):
    """Get the latest memories."""
    cur = conn.cursor()
    cur.execute("""
        SELECT id, text, importance
        FROM memories
        WHERE tombstone = 0
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    return cur.fetchall()


def _row_to_dict(row):
    """Convert a SQLite row to a dictionary."""
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
