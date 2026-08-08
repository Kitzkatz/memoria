from datetime import datetime, timezone


def touch(conn, mem_id, delta_importance=0.01):
    """Update last_accessed and boost importance."""
    with conn:
        conn.execute("""
            UPDATE memories
            SET
                last_accessed = ?,
                importance = importance + ?
            WHERE id = ?
        """, (
            datetime.now(timezone.utc).isoformat(),
            delta_importance,
            int(mem_id)
        ))


def decay_memories(conn, decay_rate=0.001):
    """Decay importance of old memories."""
    with conn:
        conn.execute("""
            UPDATE memories
            SET importance = MAX(0.1, importance - ?)
            WHERE last_accessed < datetime('now', '-1 day')
        """, (decay_rate,))


def schema(conn):
    """Get the schema of the memories table."""
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(memories)")
    return [dict(r) for r in cur.fetchall()]


def integrity_check(conn):
    """Run SQLite integrity check."""
    cur = conn.cursor()
    cur.execute("PRAGMA integrity_check")
    return cur.fetchone()[0]


def sanity_check(conn):
    """Get DB stats."""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM memories")
    db_count = cur.fetchone()[0]

    cur.execute("PRAGMA table_info(memories)")
    schema = cur.fetchall()

    return {
        "db_count": db_count,
        "columns": [c[1] for c in schema]
    }
