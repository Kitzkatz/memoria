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


def handle_store_many(system, texts, metadatas=None, skip_embedding_build=False):
    """
    Store multiple memories.

    Args:
        system: MemorySystem instance
        texts: List of memory texts
        metadatas: Optional list of metadata dicts (one per text)
        skip_embedding_build: If True, skip embedding computation and vector store operations
                               (assumes FAISS index is already loaded from cache).
    """
    overall_start = time.perf_counter()
    total = len(texts)

    debug(category="store")
    debug("=" * 60, category="store")
    debug("[STORE MANY]", category="store")
    debug("=" * 60, category="store")
    debug(f"Loading {total} memories", category="store")

    records = []
    normalized_texts = []

    extract_time = 0.0
    score_time = 0.0

    for i, text in enumerate(texts):
        t0 = time.perf_counter()
        record = system.extractor.extract(text)
        extract_time += time.perf_counter() - t0

        if metadatas and i < len(metadatas) and metadatas[i]:
            record.metadata.update(metadatas[i])

        t0 = time.perf_counter()
        record.importance = system.scorer.score(record.text, record.metadata)
        score_time += time.perf_counter() - t0

        records.append(record)
        normalized_texts.append(record.normalized_text)

    debug(f"[EXTRACT] total: {extract_time:.4f}s, avg: {extract_time/total:.4f}s", category="store")
    debug(f"[SCORE]   total: {score_time:.4f}s, avg: {score_time/total:.4f}s", category="store")

    # ---- Embedding ----
    vectors = None
    if not skip_embedding_build and not system.embedder.skip:
        t0 = time.perf_counter()
        vectors = system.embedder.embed_many(normalized_texts)
        embed_time = time.perf_counter() - t0
        debug(f"[EMBED]   total: {embed_time:.4f}s, avg: {embed_time/total:.4f}s", category="store")
    else:
        if skip_embedding_build:
            debug("[EMBED]   skipped (using cached FAISS index)", category="store")
        else:
            debug("[EMBED]   skipped (embedding disabled)", category="store")
    debug(f"[READY] {len(records)} records", category="store")

    # ---- DB Insert ----
    t0 = time.perf_counter()
    ids = system.db.insert_many(records)
    db_time = time.perf_counter() - t0
    debug(f"[DB INSERT] {db_time:.4f}s", category="store")

    # ---- Rebuild Inverted Index and BM25 ----
    if hasattr(system, 'inverted_index') and system.inverted_index:
        t0 = time.perf_counter()
        system.inverted_index.build()   # rebuilds index from DB
        inverted_index_time = time.perf_counter() - t0
        debug(f"[INVERTED INDEX] rebuilt in {inverted_index_time:.4f}s", category="store")
    else:
        debug("[INVERTED INDEX] skipped (not available)", category="store")

    if hasattr(system, 'bm25_ranker') and system.bm25_ranker:
        t0 = time.perf_counter()
        corpus_tokens = [r.tokens for r in records]
        system.bm25_ranker.build(corpus_tokens)
        bm25_time = time.perf_counter() - t0
        debug(f"[BM25] rebuilt in {bm25_time:.4f}s", category="store")
    else:
        debug("[BM25] skipped (not available)", category="store")

    # ---- Relationship Building ----
    t0 = time.perf_counter()
    for mem_id, record in zip(ids, records):
        system.relationship_builder.build(mem_id, record.relationships)
    rel_time = time.perf_counter() - t0
    debug(f"[REL BUILD] {rel_time:.4f}s", category="store")

    # ---- Cache + Vector Store Add (skip if embedding build is skipped) ----
    if not skip_embedding_build and not system.embedder.skip:
        t0 = time.perf_counter()
        system.embedding_cache.add_many(ids, vectors)
        system.vector_store.add_many(ids, vectors, persist=False)
        cache_time = time.perf_counter() - t0
        debug(f"[CACHE+ADD] {cache_time:.4f}s", category="store")
    else:
        if skip_embedding_build:
            debug("[CACHE+ADD] skipped (using cached FAISS index)", category="store")
        else:
            debug("[CACHE+ADD] skipped (embedding disabled)", category="store")

    debug("[DB] Insert complete", category="store")

    # ---- FAISS Save (skip if embedding build is skipped) ----
    if not skip_embedding_build and not system.embedder.skip:
        t0 = time.perf_counter()
        system.vector_store.save()
        save_time = time.perf_counter() - t0
        debug(f"[FAISS SAVE] {save_time:.4f}s", category="store")
    else:
        if skip_embedding_build:
            debug("[FAISS SAVE] skipped (using cached FAISS index)", category="store")
        else:
            debug("[FAISS SAVE] skipped (embedding disabled)", category="store")

    runtime = time.perf_counter() - overall_start
    debug(f"[COMPLETE] {len(ids)} memories in {runtime:.2f}s", category="store")
    return ids
