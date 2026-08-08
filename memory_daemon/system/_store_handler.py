"""
Store handling for MemorySystem.
"""

import time

from core.logger import debug


def handle_store(system, text):
    """Store a single memory."""
    t0 = time.perf_counter()
    record = system.extractor.extract(text)
    debug("extract:", time.perf_counter() - t0)

    debug("\n[TEST EXTRACTOR OUTPUT]")
    debug("TEXT:", record.text)
    debug("TYPE:", record.memory_type)
    debug("META:", record.metadata)
    debug("IMPORTANCE:", record.importance)
    debug("TEXT:", record.text)
    debug("NORMALIZED:", record.normalized_text)
    debug("TOKENS:", record.tokens)
    debug("TOKEN COUNT:", record.token_count)

    t0 = time.perf_counter()
    record.importance = system.scorer.score(record.text, record.metadata)
    debug("score:", time.perf_counter() - t0)

    t0 = time.perf_counter()
    debug(id(system.vector_store))
    vec = system.embedder.embed(record.normalized_text)
    debug("embed:", time.perf_counter() - t0)

    t0 = time.perf_counter()
    mem_id = system.db.insert(record)
    system.relationship_builder.build(mem_id, record.relationships)
    debug("\n[TEST DB INSERT]")
    row = system.db.fetch(mem_id)
    debug(f"[CACHE] {system.embedding_cache.count()} vectors")
    debug(row)
    debug("db:", time.perf_counter() - t0)

    system.embedding_cache.add(mem_id, vec)
    system.vector_store.add(mem_id, vec, persist=True)

    t0 = time.perf_counter()
    debug("\n[TEST FAISS SYNC]")
    debug("FAISS COUNT:", system.vector_store.count())
    debug("DB COUNT:", system.db.count())
    debug("faiss:", time.perf_counter() - t0)

    debug(f"Stored memory {mem_id}")
    return mem_id


def handle_store_many(system, texts):
    """Store multiple memories."""
    overall_start = time.perf_counter()
    total = len(texts)

    debug()
    debug("=" * 60)
    debug("[STORE MANY]")
    debug("=" * 60)
    debug(f"Loading {total} memories")

    records = []
    vectors = []

    for text in texts:
        record = system.extractor.extract(text)
        record.importance = system.scorer.score(record.text, record.metadata)
        vector = system.embedder.embed(record.normalized_text)
        records.append(record)
        vectors.append(vector)

    debug(f"[READY] {len(records)} records")

    ids = system.db.insert_many(records)
    for mem_id, record in zip(ids, records):
        system.relationship_builder.build(mem_id, record.relationships)

    system.embedding_cache.add_many(ids, vectors)
    system.vector_store.add_many(ids, vectors, persist=False)

    debug("[DB] Insert complete")
    system.vector_store.save()

    runtime = time.perf_counter() - overall_start
    debug(f"[COMPLETE] {len(ids)} memories in {runtime:.2f}s")
    return ids
