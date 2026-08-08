"""
MemoryDB — Facade for all database operations.
This is the public interface that all other modules import.
"""

from db.connection import DBConnection
from db.schema import init_schema
from db import crud
from db import maintenance



class MemoryDB:
    """
    Main database interface for the memory system.
    Delegates all operations to focused modules.
    """

    def __init__(self):
        self._conn = DBConnection()
        init_schema(self._conn.conn, self._conn.lock)

    # ---- Connection ----

    @property
    def conn(self):
        """Raw connection (for advanced use only)."""
        return self._conn.conn

    @property
    def lock(self):
        """Connection lock (for advanced use only)."""
        return self._conn.lock

    # ---- CRUD ----

    def insert(self, record):
        return crud.insert(self._conn.conn, self._conn.lock, record)

    def insert_many(self, records):
        return crud.insert_many(self._conn.conn, self._conn.lock, records)

    def fetch(self, mem_id):
        return crud.fetch(self._conn.conn, mem_id)

    def fetch_many(self, ids):
        return crud.fetch_many(self._conn.conn, ids)

    def fetch_all(self):
        return crud.fetch_all(self._conn.conn)

    def fetch_many_by_type(self, mem_type: str, limit: int = 50):
        return crud.fetch_many_by_type(self._conn.conn, mem_type, limit)

    def search_attribute(self, subject, attribute):
        return crud.search_attribute(self._conn.conn, subject, attribute)

    def update(self, mem_id, **kwargs):
        return crud.update(self._conn.conn, mem_id, **kwargs)

    def delete(self, mem_id):
        return crud.delete(self._conn.conn, mem_id)

    def count(self):
        return crud.count(self._conn.conn)

    def latest(self, limit=10):
        return crud.latest(self._conn.conn, limit)

    # ---- Maintenance ----

    def touch(self, mem_id, delta_importance=0.01):
        return maintenance.touch(self._conn.conn, mem_id, delta_importance)

    def decay_memories(self, decay_rate=0.001):
        return maintenance.decay_memories(self._conn.conn, decay_rate)

    def schema(self):
        return maintenance.schema(self._conn.conn)

    def integrity_check(self):
        return maintenance.integrity_check(self._conn.conn)

    def sanity_check(self):
        return maintenance.sanity_check(self._conn.conn)

    # ---- Utility ----

    def close(self):
        """Close the database connection."""
        self._conn.close()
