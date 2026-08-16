"""
Store handling for MemorySystem.
"""

import time

from core.logger import debug


def handle_store(system, text, metadata=None):
    """Store a single memory."""
    t0 = time.perf_counter()
    record = system.extractor.extract(text)
    debug("extract:", time.perf_counter() - t0)

    # Merge any provided metadata
    if metadata:
        record.metadata.update(metadata)

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


def handle_store_many(system, texts, metadatas=None):
    """Store multiple memories with per‑stage timing."""
    overall_start = time.perf_counter()
    total = len(texts)

    debug()
    debug("=" * 60)
    debug("[STORE MANY]")
    debug("=" * 60)
    debug(f"Loading {total} memories")

    records = []
    vectors = []

    # --- Stage 1: Extract, Score, Embed (per record) ---
    extract_time = 0.0
    score_time = 0.0
    embed_time = 0.0

    for i, text in enumerate(texts):
        t0 = time.perf_counter()
        record = system.extractor.extract(text)
        extract_time += time.perf_counter() - t0

        if metadatas and i < len(metadatas) and metadatas[i]:
            record.metadata.update(metadatas[i])

        t0 = time.perf_counter()
        record.importance = system.scorer.score(record.text, record.metadata)
        score_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        vector = system.embedder.embed(record.normalized_text)
        embed_time += time.perf_counter() - t0

        records.append(record)
        vectors.append(vector)

    debug(f"[EXTRACT] total: {extract_time:.4f}s, avg: {extract_time/total:.4f}s")
    debug(f"[SCORE]   total: {score_time:.4f}s, avg: {score_time/total:.4f}s")
    debug(f"[EMBED]   total: {embed_time:.4f}s, avg: {embed_time/total:.4f}s")
    debug(f"[READY] {len(records)} records")

    # --- Stage 2: DB Insert ---
    t0 = time.perf_counter()
    ids = system.db.insert_many(records)
    db_time = time.perf_counter() - t0
    debug(f"[DB INSERT] {db_time:.4f}s")

    # --- Stage 3: Relationship Building ---
    t0 = time.perf_counter()
    for mem_id, record in zip(ids, records):
        system.relationship_builder.build(mem_id, record.relationships)
    rel_time = time.perf_counter() - t0
    debug(f"[REL BUILD] {rel_time:.4f}s")

    # --- Stage 4: Cache + Vector Store Add ---
    t0 = time.perf_counter()
    system.embedding_cache.add_many(ids, vectors)
    system.vector_store.add_many(ids, vectors, persist=False)
    cache_time = time.perf_counter() - t0
    debug(f"[CACHE+ADD] {cache_time:.4f}s")

    debug("[DB] Insert complete")

    # --- Stage 5: FAISS Save ---
    t0 = time.perf_counter()
    system.vector_store.save()
    save_time = time.perf_counter() - t0
    debug(f"[FAISS SAVE] {save_time:.4f}s")

    runtime = time.perf_counter() - overall_start
    debug(f"[COMPLETE] {len(ids)} memories in {runtime:.2f}s")
    return ids
