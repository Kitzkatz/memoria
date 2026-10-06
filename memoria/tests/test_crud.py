import json
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from db import crud, schema


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


@pytest.fixture
def db_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    lock = threading.RLock()

    schema.init_schema(conn, lock)

    yield conn, lock

    conn.close()


def make_record(
    text="Test memory",
    memory_type="general",
    normalized_text=None,
    tokens=None,
    token_count=None,
    metadata=None,
    entities=None,
    relationships=None,
    importance=0.5,
):
    return SimpleNamespace(
        text=text,
        normalized_text=normalized_text,
        tokens=tokens,
        token_count=token_count,
        memory_type=memory_type,
        metadata=metadata or {},
        entities=entities or [],
        relationships=relationships or [],
        importance=importance,
    )


def seed_memory(
    conn,
    lock,
    text="Seed memory",
    memory_type="general",
    metadata=None,
    importance=0.5,
):
    record = make_record(
        text=text,
        memory_type=memory_type,
        metadata=metadata,
        importance=importance,
    )
    return crud.insert(conn, lock, record)


def test_insert_and_fetch(db_conn):
    conn, lock = db_conn

    record = make_record(
        text="Hello Memoria",
        metadata={"subject": "test", "attribute": "value"},
        importance=0.8,
    )

    mem_id = crud.insert(conn, lock, record)

    require(isinstance(mem_id, int), f"Expected integer ID, got {type(mem_id)}")

    fetched = crud.fetch(conn, mem_id)

    require(fetched is not None, "Inserted memory could not be fetched")
    require(fetched["id"] == mem_id, "Fetched ID does not match inserted ID")
    require(fetched["text"] == "Hello Memoria", "Fetched text is incorrect")
    require(
        fetched["normalized_text"] == "Hello Memoria",
        "Normalized text default is incorrect",
    )
    require(
        fetched["tokens"] == ["Hello", "Memoria"],
        f"Unexpected tokens: {fetched['tokens']}",
    )
    require(fetched["token_count"] == 2, "Token count is incorrect")
    require(
        fetched["metadata"] == {"subject": "test", "attribute": "value"},
        "Metadata was not persisted correctly",
    )
    require(fetched["importance"] == 0.8, "Importance was not persisted correctly")


def test_insert_preserves_explicit_normalized_values(db_conn):
    conn, lock = db_conn

    record = make_record(
        text="Original Text",
        normalized_text="original text",
        tokens=["original", "text"],
        token_count=2,
    )

    mem_id = crud.insert(conn, lock, record)
    fetched = crud.fetch(conn, mem_id)

    require(fetched["normalized_text"] == "original text", "Normalized text changed")
    require(fetched["tokens"] == ["original", "text"], "Explicit tokens changed")
    require(fetched["token_count"] == 2, "Explicit token count changed")


def test_insert_many_and_fetch_many(db_conn):
    conn, lock = db_conn

    records = [
        make_record(text="Memory one"),
        make_record(text="Memory two"),
        make_record(text="Memory three"),
    ]

    ids = crud.insert_many(conn, lock, records)

    require(len(ids) == 3, f"Expected 3 IDs, got {ids}")
    require(len(set(ids)) == 3, f"IDs are not unique: {ids}")

    fetched = crud.fetch_many(conn, ids)

    require(set(fetched) == set(ids), "fetch_many did not return all inserted IDs")
    require(fetched[ids[0]]["text"] == "Memory one", "First memory is incorrect")
    require(fetched[ids[1]]["text"] == "Memory two", "Second memory is incorrect")
    require(fetched[ids[2]]["text"] == "Memory three", "Third memory is incorrect")


def test_fetch_many_empty(db_conn):
    conn, _ = db_conn

    result = crud.fetch_many(conn, [])

    require(result == {}, f"Expected empty dict, got {result}")


def test_fetch_missing_returns_none(db_conn):
    conn, _ = db_conn

    result = crud.fetch(conn, 999999)

    require(result is None, f"Expected None for missing memory, got {result}")


def test_fetch_all_excludes_tombstones(db_conn):
    conn, lock = db_conn

    first = seed_memory(conn, lock, text="First")
    second = seed_memory(conn, lock, text="Second")

    crud.delete(conn, second)

    rows = crud.fetch_all(conn)

    ids = [row["id"] for row in rows]

    require(first in ids, "Live memory missing from fetch_all")
    require(second not in ids, "Deleted memory returned by fetch_all")


def test_fetch_many_by_type(db_conn):
    conn, lock = db_conn

    semantic_id = seed_memory(
        conn,
        lock,
        text="Semantic memory",
        memory_type="semantic",
        importance=0.9,
    )
    seed_memory(
        conn,
        lock,
        text="Other semantic memory",
        memory_type="semantic",
        importance=0.5,
    )

    rows = crud.fetch_many_by_type(conn, "semantic", limit=10)

    ids = [row["id"] for row in rows]

    require(semantic_id in ids, "Semantic memory missing from typed query")
    require(
        all(row["memory_type"] == "semantic" for row in rows),
        "Typed query returned a non-semantic memory",
    )


def test_update_single_field(db_conn):
    conn, lock = db_conn

    mem_id = seed_memory(conn, lock, text="Original")

    crud.update(conn, mem_id, text="Updated")

    fetched = crud.fetch(conn, mem_id)

    require(fetched["text"] == "Updated", "Text update did not persist")


def test_update_multiple_fields(db_conn):
    conn, lock = db_conn

    mem_id = seed_memory(
        conn,
        lock,
        text="Original",
        metadata={"old": "value"},
        importance=0.4,
    )

    crud.update(
        conn,
        mem_id,
        text="Updated",
        normalized_text="updated",
        tokens=["updated"],
        token_count=1,
        importance=0.95,
        metadata={"new": "value"},
        entities=["entity-a"],
        relationships=["relation-a"],
        last_accessed="2026-10-05T12:00:00+00:00",
    )

    fetched = crud.fetch(conn, mem_id)

    require(fetched["text"] == "Updated", "Updated text is incorrect")
    require(
        fetched["normalized_text"] == "updated",
        "Updated normalized text is incorrect",
    )
    require(fetched["tokens"] == ["updated"], "Updated tokens are incorrect")
    require(fetched["token_count"] == 1, "Updated token count is incorrect")
    require(fetched["importance"] == 0.95, "Updated importance is incorrect")
    require(fetched["metadata"] == {"new": "value"}, "Updated metadata is incorrect")
    require(fetched["entities"] == ["entity-a"], "Updated entities are incorrect")
    require(
        fetched["relationships"] == ["relation-a"],
        "Updated relationships are incorrect",
    )
    require(
        fetched["last_accessed"] == "2026-10-05T12:00:00+00:00",
        "Updated last_accessed is incorrect",
    )


def test_update_ignores_unknown_fields(db_conn):
    conn, lock = db_conn

    mem_id = seed_memory(conn, lock, text="Original", importance=0.5)

    result = crud.update(
        conn,
        mem_id,
        text="Updated",
        definitely_not_a_real_field="ignored",
    )

    require(result is None, f"Unexpected update return value: {result}")

    fetched = crud.fetch(conn, mem_id)

    require(fetched["text"] == "Updated", "Valid field was not updated")
    require(
        "definitely_not_a_real_field" not in fetched,
        "Unknown field somehow appeared in memory",
    )


def test_update_with_only_unknown_fields_does_nothing(db_conn):
    conn, lock = db_conn

    mem_id = seed_memory(conn, lock, text="Original")

    crud.update(
        conn,
        mem_id,
        definitely_not_a_real_field="ignored",
    )

    fetched = crud.fetch(conn, mem_id)

    require(fetched["text"] == "Original", "Memory changed from unknown field")


def test_update_missing_memory_does_nothing(db_conn):
    conn, _ = db_conn

    result = crud.update(conn, 999999, text="Should not exist")

    require(result is None, f"Unexpected return value: {result}")

    row = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE id = ?",
        (999999,),
    ).fetchone()

    require(row[0] == 0, "Missing memory somehow appeared")


def test_update_general_to_typed(db_conn):
    conn, lock = db_conn

    mem_id = seed_memory(
        conn,
        lock,
        text="Move me",
        memory_type="general",
    )

    crud.update(conn, mem_id, memory_type="semantic")

    canonical = crud.fetch(conn, mem_id)

    require(
        canonical["memory_type"] == "semantic",
        "Canonical memory_type did not change",
    )

    old_shadow = conn.execute(
        "SELECT id FROM memories WHERE id = ?",
        (mem_id,),
    ).fetchone()

    require(old_shadow is not None, "Canonical memory disappeared")

    new_shadow = conn.execute(
        "SELECT * FROM memories_semantic WHERE id = ?",
        (mem_id,),
    ).fetchone()

    require(
        new_shadow is not None,
        "New semantic shadow row was not created",
    )
    require(
        new_shadow["text"] == "Move me",
        "New semantic shadow row has wrong text",
    )
    require(
        new_shadow["memory_type"] == "semantic",
        "New semantic shadow row has wrong type",
    )


def test_update_typed_memory_same_type_syncs_shadow(db_conn):
    conn, lock = db_conn

    mem_id = seed_memory(
        conn,
        lock,
        text="Original semantic",
        memory_type="semantic",
        importance=0.4,
    )

    crud.update(
        conn,
        mem_id,
        text="Updated semantic",
        importance=0.9,
    )

    canonical = crud.fetch(conn, mem_id)

    require(
        canonical["text"] == "Updated semantic",
        "Canonical typed memory was not updated",
    )
    require(
        canonical["importance"] == 0.9,
        "Canonical importance was not updated",
    )

    shadow = conn.execute(
        "SELECT * FROM memories_semantic WHERE id = ?",
        (mem_id,),
    ).fetchone()

    require(shadow is not None, "Semantic shadow row disappeared")
    require(
        shadow["text"] == "Updated semantic",
        "Semantic shadow text was not synchronized",
    )
    require(
        shadow["importance"] == 0.9,
        "Semantic shadow importance was not synchronized",
    )


def test_update_typed_to_different_type_moves_shadow(db_conn):
    conn, lock = db_conn

    mem_id = seed_memory(
        conn,
        lock,
        text="Move typed memory",
        memory_type="semantic",
    )

    crud.update(conn, mem_id, memory_type="episodic")

    canonical = crud.fetch(conn, mem_id)

    require(
        canonical["memory_type"] == "episodic",
        "Canonical memory_type did not change",
    )

    old_shadow = conn.execute(
        "SELECT id FROM memories_semantic WHERE id = ?",
        (mem_id,),
    ).fetchone()

    require(
        old_shadow is None,
        "Old semantic shadow row was not removed",
    )

    new_shadow = conn.execute(
        "SELECT * FROM memories_episodic WHERE id = ?",
        (mem_id,),
    ).fetchone()

    require(
        new_shadow is not None,
        "New episodic shadow row was not created",
    )
    require(
        new_shadow["memory_type"] == "episodic",
        "New episodic shadow has wrong memory type",
    )


def test_update_typed_to_general_removes_shadow(db_conn):
    conn, lock = db_conn

    mem_id = seed_memory(
        conn,
        lock,
        text="Return to general",
        memory_type="semantic",
    )

    crud.update(conn, mem_id, memory_type="general")

    canonical = crud.fetch(conn, mem_id)

    require(
        canonical["memory_type"] == "general",
        "Canonical memory_type did not become general",
    )

    shadow = conn.execute(
        "SELECT id FROM memories_semantic WHERE id = ?",
        (mem_id,),
    ).fetchone()

    require(
        shadow is None,
        "Semantic shadow row was not removed",
    )


def test_delete_soft_deletes_memory_and_shadow(db_conn):
    conn, lock = db_conn

    mem_id = seed_memory(
        conn,
        lock,
        text="Delete me",
        memory_type="semantic",
    )

    crud.delete(conn, mem_id)

    fetched = crud.fetch(conn, mem_id)

    require(fetched is None, "Deleted memory is still fetchable")

    canonical = conn.execute(
        "SELECT tombstone FROM memories WHERE id = ?",
        (mem_id,),
    ).fetchone()

    require(canonical is not None, "Canonical row was physically deleted")
    require(canonical["tombstone"] == 1, "Canonical tombstone was not set")

    shadow = conn.execute(
        "SELECT tombstone FROM memories_semantic WHERE id = ?",
        (mem_id,),
    ).fetchone()

    require(shadow is not None, "Shadow row was physically deleted")
    require(shadow["tombstone"] == 1, "Shadow tombstone was not set")


def test_delete_missing_memory_does_nothing(db_conn):
    conn, _ = db_conn

    crud.delete(conn, 999999)

    count = conn.execute(
        "SELECT COUNT(*) FROM memories",
    ).fetchone()[0]

    require(count == 0, f"Unexpected rows after missing delete: {count}")


def test_count_excludes_tombstones(db_conn):
    conn, lock = db_conn

    first = seed_memory(conn, lock, text="First")
    seed_memory(conn, lock, text="Second")

    require(crud.count(conn) == 2, "Initial count is incorrect")

    crud.delete(conn, first)

    require(
        crud.count(conn) == 1,
        "Count includes tombstoned memory",
    )


def test_latest_returns_newest_live_memories(db_conn):
    conn, lock = db_conn

    first = seed_memory(conn, lock, text="First", importance=0.1)
    second = seed_memory(conn, lock, text="Second", importance=0.2)
    third = seed_memory(conn, lock, text="Third", importance=0.3)

    rows = crud.latest(conn, limit=2)

    ids = [row["id"] for row in rows]

    require(ids == [third, second], f"Unexpected latest IDs: {ids}")


def test_latest_excludes_tombstones(db_conn):
    conn, lock = db_conn

    first = seed_memory(conn, lock, text="First")
    second = seed_memory(conn, lock, text="Second")

    crud.delete(conn, second)

    rows = crud.latest(conn, limit=10)
    ids = [row["id"] for row in rows]

    require(first in ids, "Live memory missing from latest")
    require(second not in ids, "Tombstoned memory returned by latest")


def test_insert_many_typed_memories_create_shadow_rows(db_conn):
    conn, lock = db_conn

    records = [
        make_record(text="Semantic one", memory_type="semantic"),
        make_record(text="Code one", memory_type="code"),
        make_record(text="Science one", memory_type="science"),
    ]

    ids = crud.insert_many(conn, lock, records)

    require(len(ids) == 3, f"Expected three IDs, got {ids}")

    for mem_id, mem_type in zip(
        ids,
        ["semantic", "code", "science"],
    ):
        row = conn.execute(
            f"SELECT * FROM memories_{mem_type} WHERE id = ?",
            (mem_id,),
        ).fetchone()

        require(
            row is not None,
            f"Missing {mem_type} shadow row for memory {mem_id}",
        )


def test_update_metadata_preserves_json_round_trip(db_conn):
    conn, lock = db_conn

    mem_id = seed_memory(
        conn,
        lock,
        metadata={"subject": "old", "nested": {"value": 1}},
    )

    new_metadata = {
        "subject": "new",
        "nested": {"value": 2},
        "tags": ["one", "two"],
    }

    crud.update(conn, mem_id, metadata=new_metadata)

    raw = conn.execute(
        "SELECT metadata FROM memories WHERE id = ?",
        (mem_id,),
    ).fetchone()

    require(raw is not None, "Updated memory disappeared")

    stored = json.loads(raw["metadata"])

    require(
        stored == new_metadata,
        f"Metadata JSON did not round-trip: {stored}",
    )
