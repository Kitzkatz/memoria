from core.logger import debug


def init_schema(conn, lock):
    """Initialize all tables and indexes."""
    with conn:
        # Main memories table
        conn.execute("""
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
                tombstone INTEGER DEFAULT 0,
                subject TEXT GENERATED ALWAYS AS (json_extract(metadata, '$.subject')) VIRTUAL,
                attribute TEXT GENERATED ALWAYS AS (json_extract(metadata, '$.attribute')) VIRTUAL
            )
        """)

        # Entities table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                entity_type TEXT,
                aliases TEXT
            )
        """)

        # Graph table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS graph (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER NOT NULL,
                source TEXT NOT NULL,
                relation TEXT NOT NULL,
                target TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                created_at TEXT,
                FOREIGN KEY(memory_id) REFERENCES memories(id)
            )
        """)

        # Entity links
        conn.execute("""
            CREATE TABLE IF NOT EXISTS entity_links (
                entity_id INTEGER,
                memory_id INTEGER
            )
        """)

        # Goals
        conn.execute("""
            CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal TEXT,
                progress TEXT,
                status TEXT DEFAULT 'active'
            )
        """)

        # --- Type-specific tables (V4) ---
        type_tables = ["semantic", "episodic", "procedural", "code", "science", "relevance"]
        for mem_type in type_tables:
            table_name = f"memories_{mem_type}"
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    normalized_text TEXT,
                    tokens TEXT,
                    token_count INTEGER DEFAULT 0,
                    memory_type TEXT DEFAULT '{mem_type}',
                    metadata TEXT,
                    entities TEXT,
                    relationships TEXT,
                    importance REAL DEFAULT 0.5,
                    created_at TEXT,
                    last_accessed TEXT,
                    tombstone INTEGER DEFAULT 0,
                    subject TEXT GENERATED ALWAYS AS (json_extract(metadata, '$.subject')) VIRTUAL,
                    attribute TEXT GENERATED ALWAYS AS (json_extract(metadata, '$.attribute')) VIRTUAL
                )
            """)

            # Indexes for type tables
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_subject_{mem_type} ON {table_name}(subject)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_attribute_{mem_type} ON {table_name}(attribute)")

        # Main table indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_subject ON memories(subject)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_attribute ON memories(attribute)")

        conn.commit()

    debug("[DB] Schema initialized")
