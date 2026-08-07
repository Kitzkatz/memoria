from ingestion.memory_extractor import MemoryExtractor
from ranking.importance_scorer import ImportanceScorer
from retrieval.retrieval_engine import RetrievalEngine
from ranking.memory_ranker import MemoryRanker
from core.logger import debug
from core.context_builder import ContextBuilder
from cache.embedding_cache import EmbeddingCache
from ranking.mmr_reranker import MMRReranker
from ranking.attribute_booster import AttributeBooster
from ranking.score_normalizer import ScoreNormalizer
from retrieval.query_processor import QueryProcessor
from ranking.attribute_map import ATTRIBUTE_MAP
from ranking.ranking_pipeline import RankingPipeline
from graph.search import GraphSearch
from graph.edge_store import EdgeStore
from cache.config import settings
from ranking.models import CandidateRecord
from memory.feedback import FeedbackLoop
from memory.query_history import QueryHistory
from graph.numpy_graph import NumpyGraph
from memory.memory_pruner import MemoryPruner

from graph.entity_resolver import EntityResolver
from graph.relationship_builder import RelationshipBuilder

from retrieval.shard_manager import ShardManager

from ranking.tfidf import TFIDF

# ---- Ingestion workers (V4) ----
from ingestion.code_worker import CodeWorker
from ingestion.pdf_worker import PDFWorker
import time
import numpy as np

# V4 blackboard imports
from blackboard.consolidator import Consolidator
from blackboard.core import Blackboard, BlackboardEntry
from blackboard.scheduler import Scheduler
from blackboard.workers import (
    FAISSWorker, BM25Worker, GraphWorker,
    PhraseWorker, AttributeWorker   # <-- add this
)

class MemorySystem:

    def __init__(self, db, vector_store, embedder, entity_store, llm=None):
        self.db = db
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm = llm
        
        self.attribute_map = ATTRIBUTE_MAP

        self.extractor = MemoryExtractor(llm)
        self.scorer = ImportanceScorer()
        self.embedding_cache = EmbeddingCache()
        # ---- Memory Pruner (V4) ----
        self.pruner = MemoryPruner(
            db=self.db,
            vector_store=self.vector_store,
            embedding_cache=self.embedding_cache,
            threshold=getattr(settings, "PRUNE_THRESHOLD", 0.1),
            max_age_days=getattr(settings, "PRUNE_MAX_AGE_DAYS", 365),
            batch_size=getattr(settings, "PRUNE_BATCH_SIZE", 100),
            interval_seconds=getattr(settings, "PRUNE_INTERVAL_SECONDS", 3600),
            auto_start=getattr(settings, "PRUNE_AUTO_START", False),
        )
        debug(f"MemoryPruner initialized (threshold={self.pruner.threshold}, max_age={self.pruner.max_age_days}d)")

        # ---- Core components (needed before blackboard) ----
        self.query_processor = QueryProcessor()
        self.entity_store = entity_store
        self.edge_store = EdgeStore(db)
        self.entity_resolver = EntityResolver(entity_store)
        self.relationship_builder = RelationshipBuilder(self.edge_store, self.entity_store)
        self.graph_search = GraphSearch(self.edge_store, entity_store)
        self.retrieval = RetrievalEngine(
            db,
            vector_store,
            self.embedding_cache,
            self.graph_search
        )
        # ---- PDF Worker (V4) ----
        self.pdf_worker = PDFWorker(self)
        self.code_worker = CodeWorker(self)

        # ---- Sharding (V4) ----
        
        self.shard_manager = ShardManager(
            num_shards=getattr(settings, "NUM_SHARDS", 5)  # 5 types: semantic, episodic, procedural, code, science
        )
        debug(f"Shard manager initialized: {self.shard_manager.num_shards} shards (type-based)")        # ---- Numpy Graph (V4 - fast, in-memory) ----
        
        
        self.numpy_graph = NumpyGraph(db)
        debug(f"Numpy graph built: {len(self.numpy_graph.entities)} entities, {np.count_nonzero(self.numpy_graph.adj_matrix)} edges")

        # ---- Relevance Manager (V4) ----
        from memory.relevance_manager import RelevanceManager
        self.relevance_manager = RelevanceManager(db, persist_path="relevance_data.json")

        # ---- Feedback Loop (V4) ----


        # In __init__, after self.relevance_manager:
        self.feedback = FeedbackLoop(self.db, persist_path=getattr(settings, "FEEDBACK_PERSIST_PATH", "feedback_data.json"))
        self.query_history = QueryHistory(
            max_history=getattr(settings, "QUERY_HISTORY_MAX", 1000),
            persist_path=getattr(settings, "QUERY_HISTORY_PERSIST_PATH", "query_history.json")
        )
        
        # ---- TF/IDF (V4 - built from type tables) ----
        
        self.tfidf = TFIDF()
        corpus_tokens = []
        mem_types = ["semantic", "episodic", "procedural", "code", "science"]
        for mem_type in mem_types:
            rows = self.db.fetch_many_by_type(mem_type, limit=5000)
            for row in rows:
                tokens = row.get("tokens", [])
                if tokens:
                    corpus_tokens.append(tokens)
        if corpus_tokens:
            self.tfidf.build(corpus_tokens)
            debug(f"TF/IDF built on {len(corpus_tokens)} memories")
        else:
            self.tfidf = None
            debug("TF/IDF: No memories found, skipping")

        # ---- Blackboard / V4 Integration ----
        self.use_blackboard = getattr(settings, "USE_BLACKBOARD", False)
        self.bm25_ranker = None
        self.inverted_index = None
        self.consolidator = Consolidator(self.db, self.vector_store, self.embedding_cache)

        if self.use_blackboard:
            self.blackboard = Blackboard()
            self.scheduler = Scheduler(self.blackboard)

            # Build BM25 and inverted index
            if getattr(settings, "USE_BM25", False):
                from ranking.bm25_ranker import BM25
                self.bm25_ranker = BM25()
                memories = self.db.fetch_all()
                corpus = [m["tokens"] for m in memories if m.get("tokens")]
                self.bm25_ranker.build(corpus)
                debug(f"BM25 built on {len(corpus)} memories")

            if getattr(settings, "USE_INVERTED_INDEX", False):
                from retrieval.inverted_index import InvertedIndex
                self.inverted_index = InvertedIndex(self.db)
                self.inverted_index.build()

            # Create workers
            faiss_worker = FAISSWorker(self.vector_store)
            bm25_worker = BM25Worker(self.bm25_ranker, self.inverted_index) if self.bm25_ranker else None
            from blackboard.workers import GraphWorker
            graph_worker = GraphWorker(self.numpy_graph)
            attribute_worker = AttributeWorker(self.db)
            
            self.scheduler.register_worker("attribute", attribute_worker.process)
            self.scheduler.register_worker("faiss", faiss_worker.process)
            if bm25_worker:
                self.scheduler.register_worker("bm25", bm25_worker.process)
            self.scheduler.register_worker("graph", graph_worker.process)

            if getattr(settings, "USE_INVERTED_INDEX", False) and self.inverted_index:
                phrase_worker = PhraseWorker(self.inverted_index)
                self.scheduler.register_worker("phrase", phrase_worker.process)

        else:
            self.blackboard = None
            self.scheduler = None

        # ---- Ranking Pipeline ----
        self.pipeline = RankingPipeline(
            attribute_map=self.attribute_map,
            db=self.db,
            bm25_ranker=self.bm25_ranker,
            tfidf_ranker=self.tfidf,  # <-- pass TF/IDF to pipeline
            feedback_loop=self.feedback,
            ranker=None,
            normalizer=None,
            booster=None,
            context_builder=None,
            mmr=None,
            finalizer=None,
        )
    # -------------------------------------
    # STORE (unchanged)
    # -------------------------------------

    def store(self, text):
        import time
        t0 = time.perf_counter()
        record = self.extractor.extract(text)
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

        record.importance = self.scorer.score(
            record.text,
            record.metadata
        )
        debug("score:", time.perf_counter() - t0)

        t0 = time.perf_counter()
        debug(id(self.vector_store))
        vec = self.embedder.embed(record.normalized_text)
        debug("embed:", time.perf_counter() - t0)

        t0 = time.perf_counter()
        mem_id = self.db.insert(record)
        self.relationship_builder.build(
            mem_id,
            record.relationships
        )
        debug("\n[TEST DB INSERT]")
        row = self.db.fetch(mem_id)
        debug(f"[CACHE] {self.embedding_cache.count()} vectors")
        debug(row)
        debug("db:", time.perf_counter() - t0)

        self.register_embedding(mem_id, vec)

        t0 = time.perf_counter()
        debug("\n[TEST FAISS SYNC]")
        debug("FAISS COUNT:", self.vector_store.count())
        debug("DB COUNT:", self.db.count())
        debug("faiss:", time.perf_counter() - t0)

        debug(f"Stored memory {mem_id}")
        return mem_id

    # -------------------------------------
    # STORE MANY (unchanged)
    # -------------------------------------

    def store_many(self, texts):
        import time
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
            record = self.extractor.extract(text)
            record.importance = self.scorer.score(
                record.text,
                record.metadata
            )
            vector = self.embedder.embed(
                record.normalized_text
            )
            records.append(record)
            vectors.append(vector)
        debug(f"[READY] {len(records)} records")

        ids = self.db.insert_many(records)
        for mem_id, record in zip(ids, records):
            self.relationship_builder.build(
                mem_id,
                record.relationships
            )

        self.embedding_cache.add_many(
            ids,
            vectors
        )
        self.vector_store.add_many(
            ids,
            vectors,
            persist=False
        )
        debug("[DB] Insert complete")
        self.vector_store.save()

        runtime = time.perf_counter() - overall_start
        debug(
            f"[COMPLETE] {len(ids)} memories "
            f"in {runtime:.2f}s"
        )
        return ids

    # -------------------------------------
    # QUERY — with Blackboard + Phrase support
    # -------------------------------------

    def query(self, text):
        overall_start = time.perf_counter()
        query = self.query_processor.process(text)

        # Embedding
        t0_embed = time.perf_counter()
        vec = self.embedder.embed(query.normalized_text)
        embedding_ms = (time.perf_counter() - t0_embed) * 1000

        if self.use_blackboard and self.scheduler:

            t0_retrieval = time.perf_counter()
            
            # --- Step 1: Get routing configuration for memory type ---
            memory_type_hint = query.metadata.get("memory_type_hint", "general")
            
            # Use the router to get the routing configuration
            if hasattr(self, 'router') and self.router:
                route = self.router.route(memory_type_hint)
                workers_to_use = route.get("workers", ["faiss", "bm25", "graph"])
                graph_depth = route.get("graph_depth", getattr(settings, "GRAPH_DEPTH", 2))
                signals = route.get("signals", {})
                pool = route.get("pool", "memories")
                fallback_pools = route.get("fallback_pools", [])
                debug(f"[MemorySystem] Routing: type='{memory_type_hint}', workers={workers_to_use}, depth={graph_depth}")
            else:
                # Fallback to default behavior if router isn't initialized
                workers_to_use = ["faiss", "bm25", "graph", "phrase", "attribute"]
                graph_depth = getattr(settings, "GRAPH_DEPTH", 2)
                signals = {}
                pool = "memories"
                fallback_pools = []

            # --- Step 2: Check relevance pool first ---
            relevance_candidates = []
            if memory_type_hint != "general" and hasattr(self, 'relevance_manager'):
                relevance_ids = self.relevance_manager.get_top_relevant_by_type(memory_type_hint, limit=settings.TOP_K)
                if relevance_ids:
                    rows = self.db.fetch_many(relevance_ids)
                    for mem_id, row in rows.items():
                        if row:
                            memory = self.retrieval._build_memory_record(row)
                            embedding = self.embedding_cache.get(mem_id) or self.vector_store.get(mem_id)
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
                    # If no relevance IDs, fetch from the routed pool
                    if pool != "memories":
                        type_rows = self.db.fetch_many_by_type(memory_type_hint, limit=settings.TOP_K)
                    else:
                        type_rows = self.db.fetch_many(list(range(1, settings.TOP_K + 1)))  # fallback
                    for row in type_rows:
                        if row:
                            memory = self.retrieval._build_memory_record(row)
                            embedding = self.embedding_cache.get(row["id"]) or self.vector_store.get(row["id"])
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
                # --- Step 4: Get shards based on memory type hint ---
                use_sharding = getattr(settings, "USE_SHARDING", False)
                if use_sharding and hasattr(self, 'shard_manager'):
                    shards = self.shard_manager.get_shards_for_query(query, memory_type_hint)
                    num_shards = len(shards)
                    debug(f"[MemorySystem] Using type sharding: {num_shards} shard(s) for type '{memory_type_hint}'")
                else:
                    shards = [0]
                    num_shards = 1

                # --- Step 5: Submit tasks per shard based on routing ---
                task_ids = []
                top_k_per_shard = getattr(settings, "TOP_K_PER_SHARD", 300)
                graph_limit_per_shard = settings.GRAPH_TOP_K // num_shards if num_shards > 1 else settings.GRAPH_TOP_K

                for shard_id in shards:
                    # FAISS worker (if in routing)
                    if "faiss" in workers_to_use:
                        task_ids.append(
                            self.scheduler.submit(
                                "faiss",
                                {
                                    "vector": vec,
                                    "top_k": top_k_per_shard,
                                    "shard_id": shard_id,
                                    "num_shards": num_shards,
                                }
                            )
                        )

                    # BM25 worker (if in routing)
                    if self.bm25_ranker and "bm25" in workers_to_use:
                        task_ids.append(
                            self.scheduler.submit(
                                "bm25",
                                {
                                    "tokens": query.tokens,
                                    "limit": top_k_per_shard,
                                    "shard_id": shard_id,
                                    "num_shards": num_shards,
                                }
                            )
                        )

                    # Graph worker (if in routing and entities exist)
                    if query.entities and "graph" in workers_to_use:
                        task_ids.append(
                            self.scheduler.submit(
                                "graph",
                                {
                                    "entities": query.entities,
                                    "limit": graph_limit_per_shard,
                                    "shard_id": shard_id,
                                    "num_shards": num_shards,
                                    "depth": graph_depth,
                                }
                            )
                        )

                    # Phrase worker (if in routing)
                    phrases = query.metadata.get("phrases", [])
                    if phrases and self.inverted_index and "phrase" in workers_to_use:
                        task_ids.append(
                            self.scheduler.submit(
                                "phrase",
                                {
                                    "phrases": phrases,
                                    "limit": 100 // num_shards,
                                    "shard_id": shard_id,
                                    "num_shards": num_shards,
                                }
                            )
                        )

                    # Attribute worker (if in routing)
                    subject = query.metadata.get("subject")
                    attribute = query.metadata.get("attribute")
                    if subject and attribute and "attribute" in workers_to_use:
                        task_ids.append(
                            self.scheduler.submit(
                                "attribute",
                                {
                                    "subject": subject,
                                    "attribute": attribute,
                                    "shard_id": shard_id,
                                    "num_shards": num_shards,
                                }
                            )
                        )

                t_wait = time.perf_counter()
                completed = self.scheduler.wait_for_tasks(task_ids, timeout=0.05)
                debug(f"Scheduler wait {(time.perf_counter()-t_wait)*1000:.2f}ms ({'complete' if completed else 'timeout'})")

                worker_results = self.scheduler.results(task_ids)

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

                t_db = time.perf_counter()
                rows = self.db.fetch_many(list(mem_ids))
                database_ms = (time.perf_counter() - t_db) * 1000

                worker_candidates = []
                for mem_id, row in rows.items():
                    if row is None:
                        continue
                    source, dist, graph_hit = source_map.get(mem_id, (None, 0.0, False))
                    memory = self.retrieval._build_memory_record(row)
                    embedding = self.embedding_cache.get(mem_id) or self.vector_store.get(mem_id)
                    worker_candidates.append(
                        CandidateRecord(
                            memory=memory,
                            distance=dist,
                            embedding=embedding,
                            graph_hit=graph_hit,
                        )
                    )

                # --- Step 6: Merge relevance + worker candidates ---
                candidates = relevance_candidates + worker_candidates
                seen = set()
                unique_candidates = []
                for c in candidates:
                    if c.memory.id not in seen:
                        seen.add(c.memory.id)
                        unique_candidates.append(c)
                candidates = unique_candidates
                debug(f"[MemorySystem] Total unique candidates: {len(candidates)} (relevance: {len(relevance_candidates)}, workers: {len(worker_candidates)})")

            # --- Step 7: Pass signals to ranking pipeline via query metadata ---
            if signals:
                query.metadata["routing_signals"] = signals
                query.metadata["routing_pool"] = pool

            # --- Step 8: Memory Type Filtering (ensure only the right type) ---
            if memory_type_hint and memory_type_hint != "general":
                filtered = [c for c in candidates if c.memory.memory_type == memory_type_hint]
                if filtered:
                    candidates = filtered
                    debug(f"[MemorySystem] Final type filter: {len(candidates)} candidates of type '{memory_type_hint}'")
                else:
                    debug(f"[MemorySystem] No candidates of type '{memory_type_hint}', keeping all")

            faiss_ms = (time.perf_counter() - t0_retrieval) * 1000

            t_rank = time.perf_counter()
            results, ranking_diag = self.pipeline.run(query, candidates)
            ranking_ms = (time.perf_counter() - t_rank) * 1000

            response = []
            for rank, candidate in enumerate(results, start=1):
                response.append({
                    "rank": rank,
                    "id": candidate.memory.id,
                    "text": candidate.memory.text,
                    "normalized_text": candidate.memory.normalized_text,
                    "memory_type": candidate.memory.memory_type,
                    "importance": candidate.memory.importance,
                    "distance": candidate.distance,
                    "score": candidate.normalized_score,
                    "final_score": candidate.final_score,
                    "created_at": candidate.memory.created_at.isoformat(),
                    "last_accessed": candidate.memory.last_accessed.isoformat(),
                    "token_count": candidate.memory.token_count,
                    "tokens": candidate.memory.tokens,
                    "metadata": candidate.memory.metadata,
                    "entities": candidate.memory.entities,
                    "relationships": candidate.memory.relationships,
                    "graph_hit": candidate.graph_hit,
                    "diagnostics": candidate.diagnostics,
                    "mmr_score": candidate.mmr_score,
                    "diversity": candidate.diversity_score,
                })

            # ---- Record query history and feedback ----
            self.query_history.record(text, response["results"])
            if response["results"]:
                top_result = response["results"][0]
                self.feedback.record_click(top_result["id"], text)
                # Optional: track dwell time later

            formatting_ms = (time.perf_counter() - t_rank) * 1000
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

        else:
            # ---------- V3 Fallback Path ----------
            # ... keep your existing fallback code ...
            
            
            # ---------- V3 Fallback Path (unchanged) ----------
            # ... (keep the existing fallback code exactly as before) ...
            # ---------- V3 Fallback Path (unchanged) ----------
            t0_faiss = time.perf_counter()
            ids, distances = self.vector_store.search(vec)
            faiss_ms = (time.perf_counter() - t0_faiss) * 1000

            t0_db = time.perf_counter()
            candidates = self.retrieval.retrieve(query, ids, distances)
            database_ms = (time.perf_counter() - t0_db) * 1000

            debug("passing to pipeline:", len(candidates))

            t0_rank = time.perf_counter()
            results, ranking_diag = self.pipeline.run(query, candidates)
            ranking_ms = (time.perf_counter() - t0_rank) * 1000

            response = []
            for rank, candidate in enumerate(results, start=1):
                response.append({
                    "rank": rank,
                    "id": candidate.memory.id,
                    "text": candidate.memory.text,
                    "normalized_text": candidate.memory.normalized_text,
                    "memory_type": candidate.memory.memory_type,
                    "importance": candidate.memory.importance,
                    "distance": candidate.distance,
                    "score": candidate.normalized_score,
                    "final_score": candidate.final_score,
                    "created_at": candidate.memory.created_at.isoformat(),
                    "last_accessed": candidate.memory.last_accessed.isoformat(),
                    "token_count": candidate.memory.token_count,
                    "tokens": candidate.memory.tokens,
                    "metadata": candidate.memory.metadata,
                    "entities": candidate.memory.entities,
                    "relationships": candidate.memory.relationships,
                    "graph_hit": candidate.graph_hit,
                    "diagnostics": candidate.diagnostics,
                    "mmr_score": candidate.mmr_score,
                    "diversity": candidate.diversity_score,
                })

            formatting_ms = (time.perf_counter() - t0_rank) * 1000
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
                    "total_query_ms": round(total_query_ms, 3)
                }
            }

    def ingest_pdf(self, filepath: str, max_pages: int = 100):
        return self.pdf_worker.ingest_pdf(filepath, max_pages)

    def ingest_code(self, directory: str, max_files: int = 1000):
        self.code_worker.ingest_codebase(directory, max_files)
    # -------------------------------------
    # EMBEDDING REGISTRATION (unchanged)
    # -------------------------------------

    def register_embedding(self, mem_id, vector):
        debug("\nREGISTER EMBEDDING")
        debug(mem_id)
        debug(len(vector))
        debug(vector[:5])
        self.embedding_cache.add(mem_id, vector)
        self.vector_store.add(mem_id, vector, persist=True)



    def consolidate(self, threshold: float = None):
        """Run consolidation manually."""
        threshold = threshold or getattr(settings, "CONSOLIDATE_THRESHOLD", 0.5)
        self.consolidator.run(threshold)




    def prune(self, dry_run: bool = False) -> dict:
        """Manually run the pruner."""
        return self.pruner.prune_now(dry_run=dry_run)
