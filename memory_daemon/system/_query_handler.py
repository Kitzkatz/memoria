"""
Query handling for MemorySystem.
EXACT COPY of the original query method logic, extracted for decomposition.
"""

import time

from cache.config import settings
from core.logger import debug

from ranking.models import CandidateRecord
from system._response_builder import build_response, build_diagnostics_v3


# How many candidates to keep before expensive ranking
RANKING_CANDIDATE_LIMIT = 300


def handle_query(system, text):
    """
    Handle a query using V4 blackboard path or V3 fallback.
    Mirrors the original MemorySystem.query() method exactly.
    """
    overall_start = time.perf_counter()
    query = system.query_processor.process(text)

    # Embedding
    t0_embed = time.perf_counter()
    vec = system.embedder.embed(query.normalized_text)
    embedding_ms = (time.perf_counter() - t0_embed) * 1000

    if system.use_blackboard and system.scheduler:
        return _handle_query_blackboard(system, query, vec, embedding_ms, overall_start, text)
    else:
        return _handle_query_v3_fallback(system, query, vec, embedding_ms, overall_start)


def _handle_query_blackboard(system, query, vec, embedding_ms, overall_start, text):
    """V4 query path using blackboard workers. EXACT COPY."""
    t0_retrieval = time.perf_counter()

    # --- Step 1: Get routing configuration for memory type ---
    memory_type_hint = query.metadata.get("memory_type_hint", "general")

    if hasattr(system, 'router') and system.router:
        route = system.router.route(memory_type_hint)
        workers_to_use = route.get("workers", ["faiss", "bm25", "graph"])
        graph_depth = route.get("graph_depth", getattr(settings, "GRAPH_DEPTH", 2))
        signals = route.get("signals", {})
        pool = route.get("pool", "memories")
        fallback_pools = route.get("fallback_pools", [])
        debug(f"[MemorySystem] Routing: type='{memory_type_hint}', workers={workers_to_use}, depth={graph_depth}")
    else:
        workers_to_use = ["faiss", "bm25", "graph", "phrase", "attribute"]
        graph_depth = getattr(settings, "GRAPH_DEPTH", 2)
        signals = {}
        pool = "memories"
        fallback_pools = []

    # --- Step 2: Check relevance pool first ---
    relevance_candidates = []
    if memory_type_hint != "general" and hasattr(system, 'relevance_manager'):
        relevance_ids = system.relevance_manager.get_top_relevant_by_type(memory_type_hint, limit=settings.TOP_K)
        if relevance_ids:
            rows = system.db.fetch_many(relevance_ids)
            for mem_id, row in rows.items():
                if row:
                    memory = system.retrieval._build_memory_record(row)
                    embedding = system.embedding_cache.get(mem_id) or system.vector_store.get(mem_id)
                    relevance_candidates.append(
                        CandidateRecord(
                            memory=memory,
                            distance=0.0,
                            embedding=embedding,
                            graph_hit=False
                        )
                    )
            debug(f"[MemorySystem] Relevance pool returned {len(relevance_candidates)} candidates")
        else:
            if pool != "memories":
                type_rows = system.db.fetch_many_by_type(memory_type_hint, limit=settings.TOP_K)
            else:
                type_rows = system.db.fetch_many(list(range(1, settings.TOP_K + 1)))
            for row in type_rows:
                if row:
                    memory = system.retrieval._build_memory_record(row)
                    embedding = system.embedding_cache.get(row["id"]) or system.vector_store.get(row["id"])
                    relevance_candidates.append(
                        CandidateRecord(
                            memory=memory,
                            distance=0.0,
                            embedding=embedding,
                            graph_hit=False
                        )
                    )
            debug(f"[MemorySystem] Type table fallback returned {len(relevance_candidates)} candidates")

    # --- Step 3: If enough relevance candidates, skip workers ---
    if len(relevance_candidates) >= settings.TOP_K:
        candidates = relevance_candidates
        debug("[MemorySystem] Using relevance pool only (skipping workers)")
    else:
        # --- Step 4: Get shards ---
        use_sharding = getattr(settings, "USE_SHARDING", False)
        if use_sharding and hasattr(system, 'shard_manager'):
            shards = system.shard_manager.get_shards_for_query(query, memory_type_hint)
            num_shards = len(shards)
            debug(f"[MemorySystem] Using type sharding: {num_shards} shard(s) for type '{memory_type_hint}'")
        else:
            shards = [0]
            num_shards = 1

        # --- Step 5: Submit tasks per shard ---
        task_ids = []
        top_k_per_shard = getattr(settings, "TOP_K_PER_SHARD", 300)
        graph_limit_per_shard = settings.GRAPH_TOP_K // num_shards if num_shards > 1 else settings.GRAPH_TOP_K

        for shard_id in shards:
            if "faiss" in workers_to_use:
                task_ids.append(
                    system.scheduler.submit(
                        "faiss",
                        {"vector": vec, "top_k": top_k_per_shard,
                         "shard_id": shard_id, "num_shards": num_shards}
                    )
                )
            if system.bm25_ranker and "bm25" in workers_to_use:
                task_ids.append(
                    system.scheduler.submit(
                        "bm25",
                        {"tokens": query.tokens, "limit": top_k_per_shard,
                         "shard_id": shard_id, "num_shards": num_shards}
                    )
                )
            if query.entities and "graph" in workers_to_use:
                task_ids.append(
                    system.scheduler.submit(
                        "graph",
                        {"entities": query.entities, "limit": graph_limit_per_shard,
                         "shard_id": shard_id, "num_shards": num_shards, "depth": graph_depth}
                    )
                )
            phrases = query.metadata.get("phrases", [])
            if phrases and system.inverted_index and "phrase" in workers_to_use:
                task_ids.append(
                    system.scheduler.submit(
                        "phrase",
                        {"phrases": phrases, "limit": 100 // num_shards,
                         "shard_id": shard_id, "num_shards": num_shards}
                    )
                )
            subject = query.metadata.get("subject")
            attribute = query.metadata.get("attribute")
            if subject and attribute and "attribute" in workers_to_use:
                task_ids.append(
                    system.scheduler.submit(
                        "attribute",
                        {"subject": subject, "attribute": attribute,
                         "shard_id": shard_id, "num_shards": num_shards}
                    )
                )

        t_wait = time.perf_counter()
        completed = system.scheduler.wait_for_tasks(task_ids, timeout=0.05)
        debug(f"Scheduler wait {(time.perf_counter()-t_wait)*1000:.2f}ms ({'complete' if completed else 'timeout'})")

        worker_results = system.scheduler.results(task_ids)

        mem_ids = set()
        source_map = {}

        for result in worker_results:
            if not result:
                continue
            source = result["source"]
            candidates = result["candidates"]

            if source == "faiss":
                for mem_id, dist in candidates:
                    mem_ids.add(mem_id)
                    source_map[mem_id] = ("faiss", float(dist), False)
            elif source == "bm25":
                for mem_id, score in candidates:
                    mem_ids.add(mem_id)
                    source_map[mem_id] = ("bm25", 1.0 / (score + 1e-6), False)
            elif source == "graph":
                for mem_id in candidates:
                    mem_ids.add(mem_id)
                    source_map[mem_id] = ("graph", 0.0, True)
            elif source == "phrase":
                for mem_id, score in candidates:
                    mem_ids.add(mem_id)
                    source_map[mem_id] = ("phrase", 0.0, False)
            elif source == "attribute":
                for mem_id, dist in candidates:
                    mem_ids.add(mem_id)
                    source_map[mem_id] = ("attribute", float(dist), False)

        # --- OPTIMIZATION: Cap mem_ids BEFORE DB fetch and Candidate building ---
        mem_id_list = list(mem_ids)
        original_count = len(mem_id_list)
        if len(mem_id_list) > RANKING_CANDIDATE_LIMIT:
            mem_id_list = mem_id_list[:RANKING_CANDIDATE_LIMIT]
            debug(f"[MemorySystem] Capped mem_ids from {original_count} to {len(mem_id_list)} before DB fetch", category="system")

        t_db = time.perf_counter()
        rows = system.db.fetch_many(mem_id_list)
        database_ms = (time.perf_counter() - t_db) * 1000

        # --- Now build CandidateRecords only for the capped list ---
        worker_candidates = []
        for mem_id in mem_id_list:
            row = rows.get(mem_id)
            if row is None:
                continue
            source, dist, graph_hit = source_map.get(mem_id, (None, 0.0, False))
            memory = system.retrieval._build_memory_record(row)
            embedding = system.embedding_cache.get(mem_id) or system.vector_store.get(mem_id)
            worker_candidates.append(
                CandidateRecord(
                    memory=memory,
                    distance=dist,
                    embedding=embedding,
                    graph_hit=graph_hit,
                )
            )

        # --- Step 6: Merge candidates ---
        candidates = relevance_candidates + worker_candidates
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c.memory.id not in seen:
                seen.add(c.memory.id)
                unique_candidates.append(c)
        candidates = unique_candidates

        debug(f"[MemorySystem] Total candidates for ranking: {len(candidates)} (relevance: {len(relevance_candidates)}, workers: {len(worker_candidates)})")

    # --- Step 7: Pass signals ---
    if signals:
        query.metadata["routing_signals"] = signals
        query.metadata["routing_pool"] = pool

    # --- Step 8: Type Filtering ---
    if memory_type_hint and memory_type_hint != "general":
        filtered = [c for c in candidates if c.memory.memory_type == memory_type_hint]
        if filtered:
            candidates = filtered
            debug(f"[MemorySystem] Final type filter: {len(candidates)} candidates of type '{memory_type_hint}'")
        else:
            debug(f"[MemorySystem] No candidates of type '{memory_type_hint}', keeping all")

    faiss_ms = (time.perf_counter() - t0_retrieval) * 1000

    t_rank = time.perf_counter()
    results, ranking_diag = system.pipeline.run(query, candidates)
    ranking_ms = (time.perf_counter() - t_rank) * 1000

    # --- Build response (only top 10) ---
    t_format = time.perf_counter()
    response = build_response(results, limit=10)
    formatting_ms = (time.perf_counter() - t_format) * 1000

    # Record feedback using the original text
    system.query_history.record(text, response)
    if response:
        top_result = response[0]
        system.feedback.record_click(top_result["id"], text)

    total_query_ms = (time.perf_counter() - overall_start) * 1000

    return {
        "results": response,
        "diagnostics": {
            "candidate_count": len(candidates),
            "returned_count": len(response),
            "embedding_ms": round(embedding_ms, 3),
            "faiss_ms": round(faiss_ms, 3),
            "database_ms": round(database_ms, 3),
            "ranking_ms": round(ranking_ms, 3),
            "before_mmr": ranking_diag.get("before_mmr"),
            "after_mmr": ranking_diag.get("after_mmr"),
            "mmr_changed": ranking_diag.get("mmr_changed", False),
            "mmr_moves": ranking_diag.get("mmr_moves", 0),
            "formatting_ms": round(formatting_ms, 3),
            "total_query_ms": round(total_query_ms, 3),
        },
    }


def _handle_query_v3_fallback(system, query, vec, embedding_ms, overall_start):
    """V3 fallback path. EXACT COPY."""
    t0_faiss = time.perf_counter()
    ids, distances = system.vector_store.search(vec)
    faiss_ms = (time.perf_counter() - t0_faiss) * 1000

    t0_db = time.perf_counter()
    candidates = system.retrieval.retrieve(query, ids, distances)
    database_ms = (time.perf_counter() - t0_db) * 1000

    # --- OPTIMIZATION: Cap candidates before expensive ranking ---
    original_count = len(candidates)
    if len(candidates) > RANKING_CANDIDATE_LIMIT:
        candidates = candidates[:RANKING_CANDIDATE_LIMIT]
        debug(f"[MemorySystem] Capped candidates from {original_count} to {len(candidates)} for ranking (V3 fallback)", category="system")

    debug("passing to pipeline:", len(candidates))

    t0_rank = time.perf_counter()
    results, ranking_diag = system.pipeline.run(query, candidates)
    ranking_ms = (time.perf_counter() - t0_rank) * 1000

    t_format = time.perf_counter()
    response = build_response(results, limit=10)
    formatting_ms = (time.perf_counter() - t_format) * 1000

    total_query_ms = (time.perf_counter() - overall_start) * 1000

    return {
        "results": response,
        "diagnostics": build_diagnostics_v3(
            candidates, response, embedding_ms, faiss_ms, database_ms,
            ranking_ms, formatting_ms, total_query_ms
        )
    }
