import os
import sqlite3
import threading

from cache.config import settings
from core.logger import debug


class DBConnection:
    """Manages SQLite connection with WAL and proper pragmas."""

    def __init__(self):
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(
            settings.DB_PATH,
            check_same_thread=False
        )
        self.conn.row_factory = sqlite3.Row

        # WAL mode for better concurrency
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")

        debug("[DB] Using:", os.path.abspath(settings.DB_PATH))

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()

    def execute(self, sql, params=None):
        """Execute a SQL statement with parameters."""
        with self.lock:
            cur = self.conn.cursor()
            if params:
                cur.execute(sql, params)
            else:
                cur.execute(sql)
            return cur

    def commit(self):
        """Commit the current transaction."""
        with self.lock:
            self.conn.commit()

    def cursor(self):
        """Get a cursor (use within a with block for safety)."""
        return self.conn.cursor()
