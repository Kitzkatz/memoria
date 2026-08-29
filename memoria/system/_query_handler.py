"""
Query handling for MemorySystem.

V4 uses the scheduler's completion-policy system rather than a fixed
polling/quiet-period wait.

Retrieval policy:

- Require ALL submitted sources to complete.
- Apply a hard retrieval deadline as a safety ceiling.
- Workers still running after the policy finishes are not included in
  this query's result set.
"""

import time

from cache.config import settings
from core.logger import debug

from ranking.models import CandidateRecord
from system._response_builder import (
    build_response,
    build_diagnostics_v3,
)

from blackboard.scheduler import SourceCoveragePolicy

# ---- NEW IMPORT ----
from routing.matrix import get_workers_for_type


# How many candidates to keep before expensive ranking.
RANKING_CANDIDATE_LIMIT = 200


# ---------------------------------------------------------------------------
# Helper: Sort candidates by retrieval score (skip ranking)
# ---------------------------------------------------------------------------

def _sort_candidates_by_retrieval_score(candidates):
    """
    Sort candidates using the score already present from the retriever/fusion.
    If base_score is not set, fall back to inverse distance.
    Also set final_score = base_score for response builder.
    """
    for c in candidates:
        # Compute base_score if missing
        if not hasattr(c, 'base_score') or c.base_score is None:
            if c.distance is not None:
                c.base_score = 1.0 / (1.0 + c.distance)
            else:
                c.base_score = 0.0
        # Ensure final_score is set to base_score
        c.final_score = c.base_score
    candidates.sort(key=lambda c: c.base_score, reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# V4 retrieval completion policy
# ---------------------------------------------------------------------------

def _build_retrieval_policy(submitted_sources):
    """
    Build the completion policy for one retrieval pass.

    Policy:
        - Require ALL submitted sources to complete.
        - If only one source was submitted, that source is enough.
        - Do not finish early just because some sources are slow.
    """
    available_sources = set(submitted_sources)

    if not available_sources:
        return SourceCoveragePolicy(
            required_sources=set(),
            min_sources=1,
        )

    required_sources = set()

    # FAISS is mandatory when submitted
    if "faiss" in available_sources:
        required_sources.add("faiss")

    # Require ALL sources to complete
    min_sources = len(available_sources)

    return SourceCoveragePolicy(
        required_sources=required_sources,
        min_sources=min_sources,
    )


# ---------------------------------------------------------------------------
# Query entry point
# ---------------------------------------------------------------------------

def handle_query(system, text):
    """
    Handle a query using the V4 blackboard path or V3 fallback.
    """

    overall_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Query processing
    # ------------------------------------------------------------------

    t_query_process = time.perf_counter()

    query = system.query_processor.process(text)

    query_process_ms = (
        time.perf_counter() - t_query_process
    ) * 1000

    # ---- Query expansion ----
    if hasattr(system, 'query_expander') and system.query_expander:
        if getattr(settings, "USE_QUERY_EXPANSION", True):
            original_tokens = query.tokens.copy()
            expanded_tokens = system.query_expander.expand(original_tokens)
            if expanded_tokens != original_tokens:
                query.tokens = expanded_tokens
                query.metadata["original_tokens"] = original_tokens
                query.metadata["expanded_tokens"] = expanded_tokens
                query.metadata["expansion_applied"] = True
                debug(f"[QueryExpander] Expanded tokens: {original_tokens} -> {expanded_tokens}")

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    t0_embed = time.perf_counter()

    vec = system.embedder.embed(
        query.normalized_text
    )

    embedding_ms = (
        time.perf_counter() - t0_embed
    ) * 1000

    # ------------------------------------------------------------------
    # Query path selection
    # ------------------------------------------------------------------

    if system.use_blackboard and system.scheduler:
        return _handle_query_blackboard(
            system,
            query,
            vec,
            embedding_ms,
            overall_start,
            text,
            query_process_ms,
        )

    return _handle_query_v3_fallback(
        system,
        query,
        vec,
        embedding_ms,
        overall_start,
        text,
        query_process_ms,
    )


# ---------------------------------------------------------------------------
# V4 blackboard query path
# ---------------------------------------------------------------------------

def _handle_query_blackboard(
    system,
    query,
    vec,
    embedding_ms,
    overall_start,
    text,
    query_process_ms,
):
    """
    V4 query path using blackboard workers and scheduler completion policy.
    """

    # ------------------------------------------------------------------
    # Retrieval diagnostics initialized up front.
    #
    # This guarantees that the diagnostic contract remains valid even
    # when the relevance pool satisfies the query and workers are skipped.
    # ------------------------------------------------------------------

    execution = None
    scheduler_wait_ms = 0.0

    submitted_sources = set()
    completed_sources = set()
    pending_sources = set()
    failed_sources = set()

    task_source_map = {}

    # ------------------------------------------------------------------
    # Retrieval boundary
    #
    # Everything from routing through candidate construction/type
    # filtering belongs to the V4 retrieval section.
    # ------------------------------------------------------------------

    t0_retrieval = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1: Routing
    # ------------------------------------------------------------------

    memory_type_hint = query.metadata.get(
        "memory_type_hint",
        "general",
    )

    # ---- Plugin hook: pre-routing ----
    if system.plugin_manager:
        try:
            system.plugin_manager.memoria_routing_pre(query, memory_type_hint)
        except Exception as e:
            debug(f"[Plugin] routing_pre error: {e}")

    if getattr(settings, "USE_ROUTING", True) and hasattr(system, "router") and system.router:
        route = system.router.route(memory_type_hint)
        # ---- USE FUSION-AWARE WORKER LIST ----
        workers_to_use = getattr(settings, "WORKERS_TO_USE", get_workers_for_type(memory_type_hint))
        graph_depth = route.get("graph_depth", getattr(settings, "GRAPH_DEPTH", 2))
        signals = route.get("signals", {})
        pool = route.get("pool", "memories")
        fallback_pools = route.get("fallback_pools", [])

        debug(
            f"[MemorySystem] Routing: "
            f"type='{memory_type_hint}', "
            f"workers={workers_to_use}, "
            f"depth={graph_depth}"
        )

    else:
        # ---- FALLBACK: use general fusion-aware workers ----
        workers_to_use = get_workers_for_type("general")
        graph_depth = getattr(settings, "GRAPH_DEPTH", 2)
        signals = {}
        pool = "memories"
        fallback_pools = []

    # Keep this available for future routing diagnostics without changing
    # current behavior.
    _ = fallback_pools

    # ---- Plugin hook: post-routing ----
    if system.plugin_manager:
        try:
            route_info = {
                "workers": workers_to_use,
                "graph_depth": graph_depth,
                "signals": signals,
                "pool": pool,
                "fallback_pools": fallback_pools,
            }
            system.plugin_manager.memoria_routing_post(route_info)
        except Exception as e:
            debug(f"[Plugin] routing_post error: {e}")

    # ------------------------------------------------------------------
    # Step 2: Relevance pool
    # ------------------------------------------------------------------

    relevance_candidates = []

    if (
        memory_type_hint != "general"
        and hasattr(
            system,
            "relevance_manager",
        )
    ):
        relevance_ids = (
            system.relevance_manager
            .get_top_relevant_by_type(
                memory_type_hint,
                limit=settings.TOP_K,
            )
        )

        if relevance_ids:
            rows = system.db.fetch_many(
                relevance_ids
            )

            for mem_id, row in rows.items():
                if row:
                    memory = (
                        system.retrieval
                        ._build_memory_record(row)
                    )

                    embedding = (
                        system.embedding_cache.get(mem_id)
                        or system.vector_store.get(mem_id)
                    )

                    relevance_candidates.append(
                        CandidateRecord(
                            memory=memory,
                            distance=0.0,
                            embedding=embedding,
                            graph_hit=False,
                        )
                    )

            debug(
                f"[MemorySystem] Relevance pool "
                f"returned {len(relevance_candidates)} candidates"
            )

        else:
            if pool != "memories":
                type_rows = (
                    system.db.fetch_many_by_type(
                        memory_type_hint,
                        limit=settings.TOP_K,
                    )
                )
            else:
                type_rows = system.db.fetch_many(
                    list(
                        range(
                            1,
                            settings.TOP_K + 1,
                        )
                    )
                )

            for row in type_rows:
                if row:
                    memory = (
                        system.retrieval
                        ._build_memory_record(row)
                    )

                    embedding = (
                        system.embedding_cache.get(row["id"])
                        or system.vector_store.get(row["id"])
                    )

                    relevance_candidates.append(
                        CandidateRecord(
                            memory=memory,
                            distance=0.0,
                            embedding=embedding,
                            graph_hit=False,
                        )
                    )

            debug(
                f"[MemorySystem] Type table fallback "
                f"returned {len(relevance_candidates)} candidates"
            )

    # ------------------------------------------------------------------
    # Step 3: Relevance pool can satisfy query by itself
    # ------------------------------------------------------------------

    if len(relevance_candidates) >= settings.TOP_K:
        candidates = relevance_candidates

        debug(
            "[MemorySystem] Using relevance pool only "
            "(skipping workers)"
        )

        # No workers were submitted, so skip scheduler hooks.
        # We still call pre/post retrieval hooks to allow plugins to modify candidates.
        # ---- Plugin hook: pre-retrieval (with empty list) ----
        if system.plugin_manager:
            try:
                system.plugin_manager.memoria_retrieval_pre(query, [])
            except Exception as e:
                debug(f"[Plugin] retrieval_pre error: {e}")

        # ---- Plugin hook: post-retrieval ----
        if system.plugin_manager:
            try:
                system.plugin_manager.memoria_retrieval_post(query, candidates)
            except Exception as e:
                debug(f"[Plugin] retrieval_post error: {e}")

    else:
        # --------------------------------------------------------------
        # Step 4: Determine shards
        # --------------------------------------------------------------

        use_sharding = getattr(
            settings,
            "USE_SHARDING",
            False,
        )

        if (
            use_sharding
            and hasattr(
                system,
                "shard_manager",
            )
        ):
            shards = (
                system.shard_manager.get_shards_for_query(
                    query,
                    memory_type_hint,
                )
            )

            num_shards = len(shards)

            debug(
                f"[MemorySystem] Using type sharding: "
                f"{num_shards} shard(s) "
                f"for type '{memory_type_hint}'"
            )

        else:
            shards = [0]
            num_shards = 1

        # --------------------------------------------------------------
        # Step 5: Submit retrieval tasks
        # --------------------------------------------------------------

        task_ids = []

        top_k_per_shard = getattr(
            settings,
            "TOP_K_PER_SHARD",
            300,
        )

        graph_limit_per_shard = (
            settings.GRAPH_TOP_K // num_shards
            if num_shards > 1
            else settings.GRAPH_TOP_K
        )

        phrases = query.metadata.get(
            "phrases",
            [],
        )

        subject = query.metadata.get(
            "subject"
        )

        attribute = query.metadata.get(
            "attribute"
        )

        for shard_id in shards:

            # ----------------------------------------------------------
            # FAISS
            # ----------------------------------------------------------

            if "faiss" in workers_to_use:
                task_id = system.scheduler.submit(
                    "faiss",
                    {
                        "vector": vec,
                        "top_k": top_k_per_shard,
                        "shard_id": shard_id,
                        "num_shards": num_shards,
                    },
                )

                task_ids.append(task_id)
                task_source_map[task_id] = "faiss"
                submitted_sources.add("faiss")

            # ----------------------------------------------------------
            # BM25
            # ----------------------------------------------------------

            if (
                system.bm25_ranker
                and "bm25" in workers_to_use
            ):
                task_id = system.scheduler.submit(
                    "bm25",
                    {
                        "tokens": query.tokens,
                        "limit": top_k_per_shard,
                        "shard_id": shard_id,
                        "num_shards": num_shards,
                    },
                )

                task_ids.append(task_id)
                task_source_map[task_id] = "bm25"
                submitted_sources.add("bm25")

            # ----------------------------------------------------------
            # Graph
            # ----------------------------------------------------------

            if (
                query.entities
                and "graph" in workers_to_use
            ):
                task_id = system.scheduler.submit(
                    "graph",
                    {
                        "entities": query.entities,
                        "limit": graph_limit_per_shard,
                        "shard_id": shard_id,
                        "num_shards": num_shards,
                        "depth": graph_depth,
                    },
                )

                task_ids.append(task_id)
                task_source_map[task_id] = "graph"
                submitted_sources.add("graph")

            # ----------------------------------------------------------
            # Phrase
            # ----------------------------------------------------------

            if (
                phrases
                and system.inverted_index
                and "phrase" in workers_to_use
            ):
                task_id = system.scheduler.submit(
                    "phrase",
                    {
                        "phrases": phrases,
                        "limit": 100 // num_shards,
                        "shard_id": shard_id,
                        "num_shards": num_shards,
                    },
                )

                task_ids.append(task_id)
                task_source_map[task_id] = "phrase"
                submitted_sources.add("phrase")

            # ----------------------------------------------------------
            # Attribute
            # ----------------------------------------------------------

            if (
                subject
                and attribute
                and "attribute" in workers_to_use
            ):
                task_id = system.scheduler.submit(
                    "attribute",
                    {
                        "subject": subject,
                        "attribute": attribute,
                        "shard_id": shard_id,
                        "num_shards": num_shards,
                    },
                )

                task_ids.append(task_id)
                task_source_map[task_id] = "attribute"
                submitted_sources.add("attribute")

            # ----------------------------------------------------------
            # Fusion
            # ----------------------------------------------------------
            if "fusion" in workers_to_use:
                task_id = system.scheduler.submit(
                    "fusion",
                    {
                        "vector": vec,
                        "tokens": query.tokens,
                        "top_k": top_k_per_shard,
                        "shard_id": shard_id,
                        "num_shards": num_shards,
                    }
                )
                task_ids.append(task_id)
                task_source_map[task_id] = "fusion"
                submitted_sources.add("fusion")

        debug(f"[Fusion] task_ids: {task_ids}")
        debug(f"[Fusion] submitted_sources: {submitted_sources}")

        debug(
            f"[MemorySystem] Submitted retrieval sources: "
            f"{sorted(submitted_sources)}"
        )

        # ---- Plugin hook: pre-scheduler ----
        if system.plugin_manager:
            try:
                system.plugin_manager.memoria_scheduler_pre(task_ids)
            except Exception as e:
                debug(f"[Plugin] scheduler_pre error: {e}")

        # --------------------------------------------------------------
        # Step 6: Execute retrieval according to policy
        # --------------------------------------------------------------

        policy = _build_retrieval_policy(
            submitted_sources
        )

        # Hard ceiling is a safety mechanism, NOT the normal completion
        # mechanism.
        retrieval_deadline_ms = getattr(
            settings,
            "QUERY_RETRIEVAL_DEADLINE_MS",
            100,
        )

        retrieval_deadline = (
            retrieval_deadline_ms / 1000.0
        )

        t_wait = time.perf_counter()

        execution = system.scheduler.execute(
            task_ids,
            policy=policy,
            deadline=retrieval_deadline,
            cancel_pending=False,
        )

        scheduler_wait_ms = (
            time.perf_counter() - t_wait
        ) * 1000

        debug(
            f"[MemorySystem] Retrieval policy="
            f"{execution.policy_name}, "
            f"reason={execution.finish_reason}, "
            f"completed="
            f"{len(execution.completed_ids)}/"
            f"{len(execution.task_ids)}, "
            f"pending="
            f"{len(execution.pending_ids)}, "
            f"wait={scheduler_wait_ms:.2f}ms"
        )

        # ---- Plugin hook: post-scheduler ----
        if system.plugin_manager:
            try:
                system.plugin_manager.memoria_scheduler_post(execution)
            except Exception as e:
                debug(f"[Plugin] scheduler_post error: {e}")

        # --------------------------------------------------------------
        # Source-level completion diagnostics
        # --------------------------------------------------------------

        completed_sources = {
            task_source_map[task_id]
            for task_id in execution.completed_ids
            if task_id in task_source_map
        }

        pending_sources = {
            task_source_map[task_id]
            for task_id in execution.pending_ids
            if task_id in task_source_map
        }

        failed_sources = {
            task_source_map[task_id]
            for task_id in execution.failed_ids
            if task_id in task_source_map
        }

        debug(
            f"[MemorySystem] Retrieval source state: "
            f"submitted={sorted(submitted_sources)}, "
            f"completed={sorted(completed_sources)}, "
            f"pending={sorted(pending_sources)}, "
            f"failed={sorted(failed_sources)}"
        )

        # --------------------------------------------------------------
        # Step 7: Consume ONLY results completed by policy termination
        # --------------------------------------------------------------

        worker_results = execution.results

        mem_ids = set()
        source_map = {}

        for result in worker_results:
            if not result:
                continue

            source = result.get("source")

            result_candidates = result.get(
                "candidates",
                [],
            )

            if source == "faiss":
                for mem_id, dist in result_candidates:
                    mem_ids.add(mem_id)

                    source_map[mem_id] = (
                        "faiss",
                        float(dist),
                        False,
                    )

            elif source == "bm25":
                for mem_id, score in result_candidates:
                    mem_ids.add(mem_id)
                    # Use raw BM25 score directly (higher is better)
                    source_map[mem_id] = (
                        "bm25",
                        float(score),
                        False,
                    )

            elif source == "graph":
                for mem_id in result_candidates:
                    mem_ids.add(mem_id)

                    source_map[mem_id] = (
                        "graph",
                        0.0,
                        True,
                    )

            elif source == "phrase":
                for mem_id, score in result_candidates:
                    mem_ids.add(mem_id)

                    source_map[mem_id] = (
                        "phrase",
                        0.0,
                        False,
                    )

            elif source == "attribute":
                for mem_id, dist in result_candidates:
                    mem_ids.add(mem_id)

                    source_map[mem_id] = (
                        "attribute",
                        float(dist),
                        False,
                    )
            elif source == "fusion":
                for mem_id, score in result_candidates:
                    mem_ids.add(mem_id)
                    source_map[mem_id] = ("fusion", float(score), False)

        # ---- Plugin hook: pre-retrieval ----
        if system.plugin_manager:
            try:
                system.plugin_manager.memoria_retrieval_pre(query, list(mem_ids))
            except Exception as e:
                debug(f"[Plugin] retrieval_pre error: {e}")

        # --------------------------------------------------------------
        # Step 8: Cap candidates BEFORE DB fetch
        # --------------------------------------------------------------

        mem_id_list = list(mem_ids)

        original_count = len(mem_id_list)

        if len(mem_id_list) > RANKING_CANDIDATE_LIMIT:
            mem_id_list = mem_id_list[
                :RANKING_CANDIDATE_LIMIT
            ]

            debug(
                f"[MemorySystem] Capped mem_ids "
                f"from {original_count} "
                f"to {len(mem_id_list)} "
                f"before DB fetch",
                category="system",
            )

        # --------------------------------------------------------------
        # Database fetch
        # --------------------------------------------------------------

        t_db = time.perf_counter()

        rows = system.db.fetch_many(
            mem_id_list
        )

        database_ms = (
            time.perf_counter() - t_db
        ) * 1000

        # --------------------------------------------------------------
        # Candidate construction
        # --------------------------------------------------------------

        worker_candidates = []

        for mem_id in mem_id_list:
            row = rows.get(mem_id)

            if row is None:
                continue

            source, dist, graph_hit = source_map.get(
                mem_id,
                (
                    None,
                    0.0,
                    False,
                ),
            )

            memory = (
                system.retrieval
                ._build_memory_record(row)
            )

            embedding = (
                system.embedding_cache.get(mem_id)
                or system.vector_store.get(mem_id)
            )

            # Create candidate
            candidate = CandidateRecord(
                memory=memory,
                distance=dist,
                embedding=embedding,
                graph_hit=graph_hit,
            )

            # ---- Set base_score based on retrieval source ----
            if source == "fusion":
                # dist is already a similarity score from fusion worker
                candidate.base_score = dist if dist is not None else 0.0
            elif source == "faiss":
                # Convert distance to similarity (lower distance = higher similarity)
                candidate.base_score = 1.0 / (1.0 + dist) if dist is not None else 0.0
            elif source == "bm25":
                # dist is already inverted (1/(score+epsilon)) in source_map
                candidate.base_score = dist if dist is not None else 0.0
            else:
                # For graph, attribute, phrase – auxiliary, set to 0
                candidate.base_score = 0.0

            worker_candidates.append(candidate)

        # --------------------------------------------------------------
        # Step 9: Merge candidates
        # --------------------------------------------------------------

        candidates = (
            relevance_candidates
            + worker_candidates
        )

        seen = set()
        unique_candidates = []

        for candidate in candidates:
            if candidate.memory.id in seen:
                continue

            seen.add(candidate.memory.id)
            unique_candidates.append(candidate)

        candidates = unique_candidates

        debug(
            f"[MemorySystem] Total candidates "
            f"for ranking: {len(candidates)} "
            f"(relevance: "
            f"{len(relevance_candidates)}, "
            f"workers: {len(worker_candidates)})"
        )

        # ---- Plugin hook: post-retrieval ----
        if system.plugin_manager:
            try:
                system.plugin_manager.memoria_retrieval_post(query, candidates)
            except Exception as e:
                debug(f"[Plugin] retrieval_post error: {e}")

    # ==================================================================
    # CROSS-ENCODER RERANKER (optional, configurable)
    # ==================================================================
    if getattr(settings, "USE_CROSS_ENCODER", False) and len(candidates) > 1:
        try:
            from sentence_transformers import CrossEncoder
            import os

            ce_path = getattr(settings, "CROSS_ENCODER_MODEL_PATH", "memory/models/cross-encoder")
            top_k = getattr(settings, "CROSS_ENCODER_TOP_K", 100)

            # Load the model from local path
            ce = CrossEncoder(ce_path)
            top_candidates = candidates[:top_k]
            pairs = [[query.text, c.memory.text] for c in top_candidates]
            ce_scores = ce.predict(pairs)

            for c, score in zip(top_candidates, ce_scores):
                c.base_score = float(score)
                c.final_score = float(score)  # also set final_score

            # Re-sort the top candidates by CE score
            top_candidates.sort(key=lambda c: c.base_score, reverse=True)

            # Merge back: top candidates first, then the rest
            candidates = top_candidates + candidates[top_k:]

            debug(f"[Cross-Encoder] Reranked {len(top_candidates)} candidates")
        except ImportError:
            debug("[Cross-Encoder] sentence_transformers not installed. Skipping.")
        except Exception as e:
            debug(f"[Cross-Encoder] Error: {e}. Skipping.")

    # ------------------------------------------------------------------
    # Step 10: Pass routing signals
    # ------------------------------------------------------------------

    if signals:
        query.metadata["routing_signals"] = signals
        query.metadata["routing_pool"] = pool

    # ------------------------------------------------------------------
    # Step 11: Type filtering
    # ------------------------------------------------------------------

    if (
        memory_type_hint
        and memory_type_hint != "general"
    ):
        filtered = [
            candidate
            for candidate in candidates
            if candidate.memory.memory_type
            == memory_type_hint
        ]

        if filtered:
            candidates = filtered

            debug(
                f"[MemorySystem] Final type filter: "
                f"{len(candidates)} candidates "
                f"of type '{memory_type_hint}'"
            )

        else:
            debug(
                f"[MemorySystem] No candidates "
                f"of type '{memory_type_hint}', "
                f"keeping all"
            )

    # ------------------------------------------------------------------
    # Retrieval timing boundary
    # ------------------------------------------------------------------

    retrieval_ms = (
        time.perf_counter() - t0_retrieval
    ) * 1000

    # ------------------------------------------------------------------
    # Ranking (TOGGLED)
    # ------------------------------------------------------------------

    t_rank = time.perf_counter()

    if getattr(settings, "RANKING_ENABLED", True):
        # Full ranking pipeline
        results, ranking_diag = system.pipeline.run(
            query,
            candidates,
        )
    else:
        # Retrieval-only: use existing scores from retriever/fusion
        # Sort candidates by base_score (or fallback to distance inverse)
        results = _sort_candidates_by_retrieval_score(candidates)
        ranking_diag = {"ranking_skipped": True}

    ranking_ms = (
        time.perf_counter() - t_rank
    ) * 1000

    # ------------------------------------------------------------------
    # Response
    # ------------------------------------------------------------------

    t_response = time.perf_counter()

    response = build_response(
        results,
        limit=100,
    )

    response_ms = (
        time.perf_counter() - t_response
    ) * 1000

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    t_feedback = time.perf_counter()

    system.query_history.record(
        text,
        response,
    )

    if response:
        top_result = response[0]

        system.feedback.record_click(
            top_result["id"],
            text,
        )

    feedback_ms = (
        time.perf_counter() - t_feedback
    ) * 1000

    # ------------------------------------------------------------------
    # Auto-store
    # ------------------------------------------------------------------

    t_auto_store = time.perf_counter()
    auto_store_stored = 0

    if hasattr(system, "auto_store") and system.auto_store:
        if settings.AUTO_STORE_MEMORIES:
            auto_store_stored = system.auto_store.process_results(
                text,
                response,
                memory_type_hint or "chat"
            )

    auto_store_ms = (
        time.perf_counter() - t_auto_store
    ) * 1000

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    total_query_ms = (
        time.perf_counter() - overall_start
    ) * 1000

    return {
        "results": response,
        "diagnostics": {
            "candidate_count": len(candidates),
            "returned_count": len(response),

            # Precise timing boundaries.
            "query_process_ms": round(
                query_process_ms,
                3,
            ),
            "embedding_ms": round(
                embedding_ms,
                3,
            ),
            "retrieval_ms": round(
                retrieval_ms,
                3,
            ),
            "database_ms": round(
                database_ms,
                3,
            ),
            "ranking_ms": round(
                ranking_ms,
                3,
            ),
            "response_ms": round(
                response_ms,
                3,
            ),
            "feedback_ms": round(
                feedback_ms,
                3,
            ),
            "auto_store_ms": round(
                auto_store_ms,
                3,
            ),

            # Preserve the existing diagnostic name so existing
            # benchmark/reporting consumers do not break.
            "formatting_ms": round(
                response_ms,
                3,
            ),

            "total_query_ms": round(
                total_query_ms,
                3,
            ),

            # Existing ranking diagnostics.
            "before_mmr": ranking_diag.get(
                "before_mmr"
            ),
            "after_mmr": ranking_diag.get(
                "after_mmr"
            ),
            "mmr_changed": ranking_diag.get(
                "mmr_changed",
                False,
            ),
            "mmr_moves": ranking_diag.get(
                "mmr_moves",
                0,
            ),

            # Existing scheduler diagnostics.
            "retrieval_policy": (
                execution.policy_name
                if execution is not None
                else None
            ),
            "retrieval_finish_reason": (
                execution.finish_reason
                if execution is not None
                else None
            ),
            "retrieval_completed": (
                len(execution.completed_ids)
                if execution is not None
                else 0
            ),
            "retrieval_pending": (
                len(execution.pending_ids)
                if execution is not None
                else 0
            ),
            "retrieval_failed": (
                len(execution.failed_ids)
                if execution is not None
                else 0
            ),
            "retrieval_wait_ms": round(
                scheduler_wait_ms,
                3,
            ),

            # New source-level scheduler diagnostics.
            "retrieval_submitted_sources": sorted(
                submitted_sources
            ),
            "retrieval_completed_sources": sorted(
                completed_sources
            ),
            "retrieval_pending_sources": sorted(
                pending_sources
            ),
            "retrieval_failed_sources": sorted(
                failed_sources
            ),

            # Auto-store diagnostics.
            "auto_store_stored": auto_store_stored,

            # Indicate if ranking was skipped
            "ranking_skipped": not getattr(settings, "RANKING_ENABLED", True),
        },
    }


# ---------------------------------------------------------------------------
# V3 fallback
# ---------------------------------------------------------------------------

def _handle_query_v3_fallback(
    system,
    query,
    vec,
    embedding_ms,
    overall_start,
    text,
    query_process_ms,
):
    """
    V3 fallback path.

    Existing V3 diagnostics contract is preserved through
    build_diagnostics_v3(). Additional timing diagnostics are appended
    afterward rather than changing that function's signature.
    """

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    t0_retrieval = time.perf_counter()

    # Existing FAISS boundary.
    t0_faiss = time.perf_counter()

    ids, distances = system.vector_store.search(
        vec
    )

    faiss_ms = (
        time.perf_counter() - t0_faiss
    ) * 1000

    # Existing DB/retrieval boundary.
    t0_db = time.perf_counter()

    candidates = system.retrieval.retrieve(
        query,
        ids,
        distances,
    )

    database_ms = (
        time.perf_counter() - t0_db
    ) * 1000

    retrieval_ms = (
        time.perf_counter() - t0_retrieval
    ) * 1000

    # ------------------------------------------------------------------
    # Cap candidates before expensive ranking
    # ------------------------------------------------------------------

    original_count = len(candidates)

    if len(candidates) > RANKING_CANDIDATE_LIMIT:
        candidates = candidates[
            :RANKING_CANDIDATE_LIMIT
        ]

        debug(
            f"[MemorySystem] Capped candidates "
            f"from {original_count} "
            f"to {len(candidates)} "
            f"for ranking "
            f"(V3 fallback)",
            category="system",
        )

    debug(
        "passing to pipeline:",
        len(candidates),
    )

    # ------------------------------------------------------------------
    # Ranking (TOGGLED)
    # ------------------------------------------------------------------

    t0_rank = time.perf_counter()

    if getattr(settings, "RANKING_ENABLED", True):
        results, ranking_diag = system.pipeline.run(
            query,
            candidates,
        )
    else:
        # Retrieval-only: use existing scores from retriever
        results = _sort_candidates_by_retrieval_score(candidates)
        ranking_diag = {"ranking_skipped": True}

    ranking_ms = (
        time.perf_counter() - t0_rank
    ) * 1000

    # ------------------------------------------------------------------
    # Response
    # ------------------------------------------------------------------

    t_response = time.perf_counter()

    response = build_response(
        results,
        limit=10,
    )

    response_ms = (
        time.perf_counter() - t_response
    ) * 1000

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    t_feedback = time.perf_counter()

    system.query_history.record(
        text,
        response,
    )

    if response:
        top_result = response[0]

        system.feedback.record_click(
            top_result["id"],
            text,
        )

    feedback_ms = (
        time.perf_counter() - t_feedback
    ) * 1000

    # ------------------------------------------------------------------
    # Auto-store (V3 fallback)
    # ------------------------------------------------------------------

    t_auto_store = time.perf_counter()
    auto_store_stored = 0

    if hasattr(system, "auto_store") and system.auto_store:
        if settings.AUTO_STORE_MEMORIES:
            auto_store_stored = system.auto_store.process_results(
                text,
                response,
                "general"
            )

    auto_store_ms = (
        time.perf_counter() - t_auto_store
    ) * 1000

    # ------------------------------------------------------------------
    # Total
    # ------------------------------------------------------------------

    total_query_ms = (
        time.perf_counter() - overall_start
    ) * 1000

    diagnostics = build_diagnostics_v3(
        candidates,
        response,
        embedding_ms,
        faiss_ms,
        database_ms,
        ranking_ms,
        response_ms,
        total_query_ms,
    )

    # Preserve the V3 builder's existing contract while adding the
    # diagnostics needed by the benchmark.
    diagnostics.update(
        {
            "query_process_ms": round(
                query_process_ms,
                3,
            ),
            "embedding_ms": round(
                embedding_ms,
                3,
            ),
            "retrieval_ms": round(
                retrieval_ms,
                3,
            ),
            "database_ms": round(
                database_ms,
                3,
            ),
            "ranking_ms": round(
                ranking_ms,
                3,
            ),
            "response_ms": round(
                response_ms,
                3,
            ),
            "feedback_ms": round(
                feedback_ms,
                3,
            ),
            "auto_store_ms": round(
                auto_store_ms,
                3,
            ),

            # Preserve existing naming.
            "formatting_ms": round(
                response_ms,
                3,
            ),

            "total_query_ms": round(
                total_query_ms,
                3,
            ),

            # V3 has no scheduler/source completion state.
            "retrieval_policy": None,
            "retrieval_finish_reason": None,
            "retrieval_completed": 0,
            "retrieval_pending": 0,
            "retrieval_failed": 0,
            "retrieval_wait_ms": 0.0,

            "retrieval_submitted_sources": [
                "faiss",
            ],
            "retrieval_completed_sources": [
                "faiss",
            ],
            "retrieval_pending_sources": [],
            "retrieval_failed_sources": [],

            # Auto-store diagnostics.
            "auto_store_stored": auto_store_stored,

            # Indicate if ranking was skipped
            "ranking_skipped": not getattr(settings, "RANKING_ENABLED", True),
        }
    )

    return {
        "results": response,
        "diagnostics": diagnostics,
    }
