# db/clear.py

"""
Clear all data from the database while preserving schema.
Used for per-question isolation in benchmarks.
"""

import sqlite3
from pathlib import Path
from cache.config import settings
from core.logger import debug


def clear_db():
    """
    Clear all data from all tables but keep schema.
    This closes any existing connections and reopens fresh.
    """
    db_path = Path(settings.DB_PATH)
    
    # Force close any existing connections by using a fresh one
    conn = sqlite3.connect(str(db_path))
    
    # Get all table names
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    
    # Disable foreign key checks for speed
    conn.execute("PRAGMA foreign_keys = OFF")
    
    # Delete all data from each table
    for (table_name,) in tables:
        conn.execute(f"DELETE FROM {table_name}")
    
    # Reset autoincrement counters
    for (table_name,) in tables:
        conn.execute(f"DELETE FROM sqlite_sequence WHERE name='{table_name}'")
    
    conn.execute("PRAGMA foreign_keys = ON")
    conn.commit()
    conn.close()
    
    debug(f"[DB] Cleared {len(tables)} tables")
    return len(tables)
